import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.features.engineer import find_min_d, frac_diff_ffd
from src.model.purged_kfold import PurgedKFold, get_avg_uniqueness

SYMBOL = "SOLUSDT"
FEATURES_PATH = f"data/processed/features_{SYMBOL}.parquet"
MODEL_PATH = f"data/processed/model_rf_{SYMBOL}.pkl"
OOS_PATH = f"data/processed/oos_preds_{SYMBOL}.csv"

FEATURE_COLS  = [
    # original microstructure features
    "log_return", "rel_duration", "dvi", "frac_diff",
    # technical indicators
    "bb_pct_b", "bb_width",
    "rsi",
    "macd", "macd_signal", "macd_hist",
    "adx", "adx_pdi", "adx_mdi",
    "stoch_rsi_k", "stoch_rsi_d",
]
TARGET_COL = "label"
WEIGHT_COL = "ret"

MAX_HOLD = 400
N_SPLITS = 5
EMBARGO_PCT = 0.01

def t1_positions(clean: pd.DataFrame) -> np.ndarray:
    """Map each row's t1_time to its integer position within clean via searchsorted."""
    close_ns = clean["close_time"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    t1_ns    = clean["t1_time"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    pos = np.searchsorted(close_ns, t1_ns, side="right") - 1
    return np.clip(pos, np.arange(len(clean)), len(clean) - 1)


def time_decay_weights(n: int, c: float = 1.0) -> np.ndarray:
    # AFML Ch.4.5: linear ramp from c (oldest) to 1.0 (newest), clipped at 0.
    # c=1 → flat (no decay); c=0 → 0..1; c<0 → oldest observations zeroed.
    return np.maximum(np.linspace(c, 1.0, n), 0.0)


def build_sample_weights(
    ret: pd.Series,
    avg_u: np.ndarray,
    decay: np.ndarray | None = None,
    use_time_decay: bool = False,
    use_overlap: bool = False
) -> np.ndarray:
    # AFML Eq 4.10 + Ch.4.5: w_i = decay_i × ū_i × |ret_i|, normalised to N
    w = avg_u * ret.abs().to_numpy(dtype=np.float64) if use_overlap else ret.abs().to_numpy(dtype=np.float64)
    if use_time_decay and decay is not None:
        w = w * decay
    total = w.sum()
    if total == 0:
        return np.ones(len(ret), dtype=np.float64)
    return w / total * len(ret)


def run_cv(
    df: pd.DataFrame,
    n_splits: int = N_SPLITS,
    embargo_pct: float = EMBARGO_PCT,
    max_hold: int = MAX_HOLD,
    rf_params: dict | None = None,
    feature_cols: list | None = None,
    time_decay_c: float = 0.67,
    use_time_decay: bool = False,
    use_overlap: bool = False,
) -> tuple[list, pd.DataFrame, list[dict]]:
    fc = feature_cols if feature_cols is not None else FEATURE_COLS
    # Always include log_return in clean (needed for strategy return computation)
    extra = [c for c in [TARGET_COL, WEIGHT_COL, "open_time", "close_time", "t1", "t1_time", "close", "log_return"] if c not in fc]
    clean = df[list(fc) + extra].dropna()
    X = clean[list(fc)]
    y = clean[TARGET_COL]

    cv = PurgedKFold(n_splits=n_splits, embargo_pct=embargo_pct, t1_col="t1")

    next_ret = clean["log_return"].shift(-1)
    log_px = pd.Series(np.log(clean["close"].to_numpy(dtype=np.float64)), index=clean.index)

    fold_models = []
    oos_records = []
    fold_data = []

    for fold_i, (train_idx, test_idx) in enumerate(cv.split(clean)):
        best_d, _ = find_min_d(clean["close"].iloc[train_idx])
        fd = frac_diff_ffd(log_px, best_d)

        X_tr = X.iloc[train_idx].copy()
        X_te = X.iloc[test_idx].copy()
        if "frac_diff" in fc:
            X_tr["frac_diff"] = fd.iloc[train_idx].values
            X_te["frac_diff"] = fd.iloc[test_idx].values

        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        model = RandomForestClassifier(**rf_params)
        if use_time_decay or use_overlap:
            train_clean = clean.iloc[train_idx]
            avg_u_tr = get_avg_uniqueness(len(train_clean), max_hold, t1_indices=t1_positions(train_clean))
            decay_tr = time_decay_weights(len(train_clean), c=time_decay_c)
            w_tr = build_sample_weights(train_clean[WEIGHT_COL], avg_u_tr, decay=decay_tr)
            model.fit(X_tr, y_tr, sample_weight=w_tr)
        else:
            model.fit(X_tr, y_tr)

        y_pred = model.predict(X_te)
        y_proba = model.predict_proba(X_te)
        y_train_pred = model.predict(X_tr)

        fold_data.append({
            "fold": fold_i,
            "train_size": len(X_tr),
            "test_size": len(X_te),
            "y_true": y_te.to_numpy(),
            "y_pred": y_pred,
            "bar_ret": next_ret.iloc[test_idx].to_numpy(),
            "y_train_true": y_tr.to_numpy(),
            "y_train_pred": y_train_pred,
            "timestamps": clean["close_time"].iloc[test_idx].to_numpy(),
        })

        cls_names = [f"proba_{int(c)}" for c in model.classes_]
        col_map = {int(c): name for c, name in zip(model.classes_, cls_names)}
        for loc, pred, true, prob in zip(
            clean.index[test_idx], y_pred, y_te.to_numpy(), y_proba
        ):
            record = {"idx": loc, "y_true": true, "y_pred": pred}
            for cls_int, p in zip(map(int, model.classes_), prob):
                record[col_map[cls_int]] = p
            oos_records.append(record)

        fold_models.append(model)

    oos_df = (
        pd.DataFrame(oos_records)
        .set_index("idx")
        .sort_index()
        .join(clean[[WEIGHT_COL, "close_time"]].rename(columns={WEIGHT_COL: "ret"}))
        .join(next_ret.to_frame("log_return"))
    )
    return fold_models, oos_df, fold_data


def train_final(
    df: pd.DataFrame,
    rf_params: dict,
    max_hold: int = MAX_HOLD,
    feature_cols: list | None = None,
    out_path: str = MODEL_PATH,
    time_decay_c: float = 1.0,
    use_time_decay: bool = False,
    use_overlap: bool = False,
) -> RandomForestClassifier:
    fc = feature_cols if feature_cols is not None else FEATURE_COLS
    extra = [c for c in [TARGET_COL, WEIGHT_COL, "close_time", "t1_time"] if c not in fc]
    clean = df[list(fc) + extra].dropna()
    X, y = clean[list(fc)], clean[TARGET_COL]
    model = RandomForestClassifier(**rf_params)
    if use_overlap or use_time_decay:
        avg_u = get_avg_uniqueness(len(clean), max_hold, t1_indices=t1_positions(clean))
        decay = time_decay_weights(len(clean), c=time_decay_c) if time_decay_c != 1.0 else None
        weights = build_sample_weights(clean[WEIGHT_COL], avg_u, decay=decay, use_time_decay=use_time_decay, use_overlap=use_overlap)
        model.fit(X, y, sample_weight=weights)
    else:
        model.fit(X, y)
    joblib.dump(model, out_path)
    return model