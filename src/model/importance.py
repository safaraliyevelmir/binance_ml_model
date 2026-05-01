import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from src.model.purged_kfold import PurgedKFold, get_avg_uniqueness
from src.model.train import (
    FEATURE_COLS, FEATURES_PATH, TARGET_COL, WEIGHT_COL,
    MAX_HOLD, N_SPLITS, EMBARGO_PCT, RF_PARAMS,
    build_sample_weights, t1_positions,
)

cols = FEATURE_COLS + [TARGET_COL, WEIGHT_COL, "open_time", "close_time", "t1", "t1_time"]

def mdi_importance(
    model: RandomForestClassifier,
    feature_cols: list[str],
) -> pd.DataFrame:
    imps = np.array([tree.feature_importances_ for tree in model.estimators_])
    return pd.DataFrame(
        {"mean": imps.mean(axis=0), "std": imps.std(axis=0)},
        index=feature_cols,
    ).sort_values("mean", ascending=False)


def mda_importance(
    df: pd.DataFrame,
    n_splits: int = N_SPLITS,
    embargo_pct: float = EMBARGO_PCT,
    n_repeats: int = 1,
) -> pd.DataFrame:
    clean = df[cols].dropna()
    X_all = clean[FEATURE_COLS].to_numpy(dtype=np.float64)
    y_all = clean[TARGET_COL].to_numpy()
    avg_u = get_avg_uniqueness(len(clean), MAX_HOLD, t1_indices=t1_positions(clean))
    w_all = build_sample_weights(clean[WEIGHT_COL], avg_u)

    cv = PurgedKFold(n_splits=n_splits, embargo_pct=embargo_pct, t1_col="t1")
    rng = np.random.default_rng(42)

    fold_drops: list[dict] = []

    for fold_i, (train_idx, test_idx) in enumerate(cv.split(clean)):
        print(f"MDA fold {fold_i}: train={len(train_idx):,}  test={len(test_idx):,}")
        model = RandomForestClassifier(**RF_PARAMS)
        model.fit(X_all[train_idx], y_all[train_idx], sample_weight=w_all[train_idx])

        baseline = accuracy_score(y_all[test_idx], model.predict(X_all[test_idx]))

        drops: dict[str, float] = {}
        X_test = X_all[test_idx].copy()

        for _ in range(n_repeats):
            for j, feat in enumerate(FEATURE_COLS):
                X_perm = X_test.copy()
                X_perm[:,j] = rng.permutation(X_perm[:, j])
                perm_acc = accuracy_score(y_all[test_idx], model.predict(X_perm))
                drops[feat] = drops.get(feat, 0.0) + (baseline - perm_acc)

        fold_drops.append({f: v / n_repeats for f, v in drops.items()})

    result = pd.DataFrame(fold_drops, columns=FEATURE_COLS)
    return pd.DataFrame(
        {"mean": result.mean(), "std": result.std()},
    ).sort_values("mean", ascending=False)

def sfi_importance(
    df: pd.DataFrame,
    n_splits: int = N_SPLITS,
    embargo_pct: float = EMBARGO_PCT,
) -> pd.DataFrame:
    clean = df[cols].dropna()
    y_all = clean[TARGET_COL].to_numpy()
    avg_u = get_avg_uniqueness(len(clean), MAX_HOLD, t1_indices=t1_positions(clean))
    w_all = build_sample_weights(clean[WEIGHT_COL], avg_u)
    cv = PurgedKFold(n_splits=n_splits, embargo_pct=embargo_pct, t1_col="t1")

    scores: dict[str, list[float]] = {f: [] for f in FEATURE_COLS}

    for feat in FEATURE_COLS:
        X_single = clean[[feat]].to_numpy(dtype=np.float64)
        print(f"SFI: {feat}")
        for train_idx, test_idx in cv.split(clean):
            model = RandomForestClassifier(**RF_PARAMS)
            model.fit(X_single[train_idx], y_all[train_idx], sample_weight=w_all[train_idx])
            acc = accuracy_score(y_all[test_idx], model.predict(X_single[test_idx]))
            scores[feat].append(acc)

    rows = {f: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1))}
            for f, v in scores.items()}
    return pd.DataFrame(rows).T.sort_values("mean", ascending=False)

if __name__ == "__main__":
    import joblib

    df = pd.read_parquet(FEATURES_PATH)
    df["t1"] = PurgedKFold.approximate_t1(df, max_hold=MAX_HOLD)

    model = joblib.load("data/processed/model_rf__SOLUSDT.pkl")

    mdi = mdi_importance(model, FEATURE_COLS)
    print(mdi.to_string())

    mda = mda_importance(df)
    print(mda.to_string())

    sfi = sfi_importance(df)
    print(sfi.to_string())
