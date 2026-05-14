from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, auc, classification_report, precision_recall_curve

from src.backtest_utils import (
    FEE_BPS_PER_SIDE_DEFAULT,
    SLIPPAGE_BPS_PER_SIDE_DEFAULT,
    realistic_backtest,
)
from src.main import (
    FEATURE_SETS,
    MAX_HOLD_RANGES,
    RF_PARAMS_FIXED,
    USE_OVERLAP_FIXED,
    USE_TIME_DECAY_FIXED,
    compute_bar_features,
    cusum_filter,
    safe,
)
from src.model.metrics import compute_bars_per_year, sharpe_ratio, strategy_log_returns
from src.model.train import (
    SYMBOL,
    run_cv
)
from src.pre_process.trippler_barrier import getDailyVol, label_bars

# ─── Trial list (deduped from the user's pasted Optuna rows) ────────────────
TRIALS: list[dict] = [
    {"trial_id": 273, "bar_size": 3, "cusum_h": 0.0035, "features_set": "6_indicators", "max_hold_slot": 1, "pt_sl": "0.8_0.8", "span": 50},
    {"trial_id":  37, "bar_size": 3, "cusum_h": 0.0025, "features_set": "6_indicators", "max_hold_slot": 1, "pt_sl": "0.8_0.8", "span": 40},
    {"trial_id":  21, "bar_size": 3, "cusum_h": 0.0025, "features_set": "6_indicators", "max_hold_slot": 2, "pt_sl": "0.8_0.8", "span": 40},
    {"trial_id": 142, "bar_size": 3, "cusum_h": 0.002,  "features_set": "extended",     "max_hold_slot": 2, "pt_sl": "0.8_0.8", "span": 40},
    {"trial_id": 238, "bar_size": 3, "cusum_h": 0.0035, "features_set": "6_indicators", "max_hold_slot": 2, "pt_sl": "0.8_0.8", "span": 40},
    {"trial_id": 217, "bar_size": 3, "cusum_h": 0.0035, "features_set": "6_indicators", "max_hold_slot": 1, "pt_sl": "0.8_0.8", "span": 40},
    {"trial_id": 174, "bar_size": 3, "cusum_h": 0.005,  "features_set": "6_indicators", "max_hold_slot": 1, "pt_sl": "0.8_0.8", "span": 50},
    {"trial_id":  59, "bar_size": 3, "cusum_h": 0.003,  "features_set": "extended",     "max_hold_slot": 2, "pt_sl": "0.8_0.8", "span": 50},
]

# ─── Execution-cost assumptions (Binance USDT-M Futures, taker tier) ────────
FEE_BPS_PER_SIDE: float = FEE_BPS_PER_SIDE_DEFAULT
SLIPPAGE_BPS_PER_SIDE: float = SLIPPAGE_BPS_PER_SIDE_DEFAULT

# ─── Split / CV hyper-params (must match main.py to stay comparable) ────────
TEST_RATIO: float = 0.20
N_SPLITS: int = 5
EMBARGO_PCT: float = 0.01

# ─── Output paths ───────────────────────────────────────────────────────────
OUT_ROOT = Path("data/processed/top_trials")
LOGS_DIR = Path("logs/top_trials")
LABELS_FULL_CACHE_DIR = Path("data/processed/labels_full")
MODELS_DIR = OUT_ROOT / "models"
IMPORTANCE_DIR = OUT_ROOT / "importance"
PREDS_DIR = OUT_ROOT / "preds"
CONFIGS_DIR = OUT_ROOT / "configs"

for d in (OUT_ROOT, LOGS_DIR, LABELS_FULL_CACHE_DIR, MODELS_DIR, IMPORTANCE_DIR, PREDS_DIR, CONFIGS_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ── Helpers ────────────────────────────────────────────────────────────────


def trial_tag(t: dict) -> str:
    return (
        f"trial{t['trial_id']:04d}"
        f"_bar{t['bar_size']}"
        f"_cusum{t['cusum_h']}"
        f"_{t['features_set']}"
        f"_slot{t['max_hold_slot']}"
        f"_pt{t['pt_sl']}"
        f"_span{t['span']}"
    )


def setup_trial_logger(tag: str) -> tuple[logging.Logger, Path]:
    path = LOGS_DIR / f"{tag}.log"
    log = logging.getLogger(f"top_trials.{tag}")
    log.setLevel(logging.DEBUG)
    if log.handlers:
        log.handlers.clear()
    log.propagate = False
    fmt = logging.Formatter("%(asctime)s [%(levelname)-7s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(ch)
    return log, path


def labels_full_cache_path(bar_size: int, span: int, pt: float, sl: float, max_hold: int) -> Path:
    return LABELS_FULL_CACHE_DIR / (
        f"labels_full_{SYMBOL}_bar{bar_size}_span{span}"
        f"_pt{pt}_sl{sl}_hold{max_hold}.parquet"
    )


# ── Feature importance ───────────────────────────────────────────────────


def mdi_importance(model: RandomForestClassifier, features: list[str]) -> pd.DataFrame:
    imps = np.array([t.feature_importances_ for t in model.estimators_])
    return (
        pd.DataFrame({"mean": imps.mean(0), "std": imps.std(0)}, index=features)
        .rename_axis("feature")
        .sort_values("mean", ascending=False)
    )


def mda_importance(
    model: RandomForestClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    features: list[str],
    n_repeats: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    baseline = accuracy_score(y_test, model.predict(X_test))
    rows = []
    for j, f in enumerate(features):
        drops = np.empty(n_repeats, dtype=float)
        for r in range(n_repeats):
            X_p = X_test.copy()
            X_p[:, j] = rng.permutation(X_p[:, j])
            drops[r] = baseline - accuracy_score(y_test, model.predict(X_p))
        rows.append({"feature": f, "mean": float(drops.mean()), "std": float(drops.std(ddof=1) if n_repeats > 1 else 0.0)})
    return pd.DataFrame(rows).set_index("feature").sort_values("mean", ascending=False)


def sfi_importance(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    features: list[str],
    rf_params: dict,
) -> pd.DataFrame:
    """Train a single-feature RF on trainval, score on the held-out test."""
    y_tr = df_train["label"].to_numpy()
    y_te = df_test["label"].to_numpy()
    rows = []
    for f in features:
        X_tr = df_train[[f]].to_numpy(dtype=float)
        X_te = df_test[[f]].to_numpy(dtype=float)
        m = RandomForestClassifier(**rf_params)
        m.fit(X_tr, y_tr)
        acc = accuracy_score(y_te, m.predict(X_te))
        rows.append({"feature": f, "mean": float(acc), "std": 0.0})
    return pd.DataFrame(rows).set_index("feature").sort_values("mean", ascending=False)


# ── Classification-report unpack helper ───────────────────────────────────


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rep = classification_report(
        y_true, y_pred, labels=[0, 1], target_names=["flat", "long"],
        output_dict=True, zero_division=0,
    )
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    return {
        "accuracy":    safe(rep["accuracy"]),
        "f1_macro":    safe(rep["macro avg"]["f1-score"]),
        "f1_long":     safe(rep["long"]["f1-score"]),
        "f1_flat":     safe(rep["flat"]["f1-score"]),
        "prec_long":   safe(rep["long"]["precision"]),
        "prec_flat":   safe(rep["flat"]["precision"]),
        "recall_long": safe(rep["long"]["recall"]),
        "recall_flat": safe(rep["flat"]["recall"]),
        "auc_pr":      safe(auc(recall, precision)),
    }


# ── Pipeline ───────────────────────────────────────────────────────────────


def build_full_dataset(trial: dict, log: logging.Logger) -> tuple[pd.DataFrame, pd.Timestamp, dict]:
    """
    Reproduces main.py preprocessing on the FULL bar sequence (no trainval
    slicing before labeling). Returns:
      df        : feature/label DataFrame, binary target, dropped flat,
                  CUSUM-filtered, sorted by close_time. Contains a `ret`
                  column = realized triple-barrier log return per row.
      cutoff_ts : datetime at the 80 % mark, used to split trainval/test.
      stats     : data-stage statistics for the log.
    """
    bar_size = trial["bar_size"]
    span = trial["span"]
    pt, sl = (float(x) for x in trial["pt_sl"].split("_"))
    low, _, step = MAX_HOLD_RANGES[bar_size]
    max_hold = low + trial["max_hold_slot"] * step

    t0 = time.perf_counter()
    bars_path = f"data/processed/dollar_bars_{bar_size}_{SYMBOL}.parquet"
    bars = pd.read_parquet(bars_path)
    log.info(f"  bars: {len(bars):,} rows  [{time.perf_counter()-t0:.1f}s]")

    cutoff_idx = int(len(bars) * (1.0 - TEST_RATIO))
    cutoff_ts = pd.Timestamp(bars["close_time"].iloc[cutoff_idx])
    log.info(f"  trainval/test cutoff @ row {cutoff_idx:,}  ({cutoff_ts})")

    t0 = time.perf_counter()
    features_full = compute_bar_features(bars)
    log.info(f"  features computed  [{time.perf_counter()-t0:.1f}s]")

    t0 = time.perf_counter()
    close_ts = pd.Series(bars["close"].values, index=pd.DatetimeIndex(bars["close_time"]))
    vol_full = getDailyVol(close_ts, span=span)
    vol_full = vol_full[~vol_full.index.duplicated(keep="last")]
    log.info(f"  daily-vol  [{time.perf_counter()-t0:.1f}s]")

    cache_path = labels_full_cache_path(bar_size, span, pt, sl, max_hold)
    if cache_path.exists():
        t0 = time.perf_counter()
        labels_full = pd.read_parquet(cache_path)
        log.info(f"  labels cache hit → {cache_path.name}  [{time.perf_counter()-t0:.1f}s]")
    else:
        t0 = time.perf_counter()
        vol_aligned = vol_full.reindex(pd.DatetimeIndex(bars["close_time"])).ffill()
        labels_full = label_bars(bars, vol_aligned, pt=pt, sl=sl, max_hold=max_hold)
        labels_full.to_parquet(cache_path, index=False)
        log.info(f"  labels computed → {cache_path.name}  [{time.perf_counter()-t0:.1f}s]")

    labelable = len(labels_full)
    bars_lab = bars.iloc[:labelable].reset_index(drop=True)
    feats_lab = features_full.iloc[:labelable].reset_index(drop=True)

    t0 = time.perf_counter()
    event_times = cusum_filter(close_ts, h=trial["cusum_h"])
    cusum_mask = bars_lab["close_time"].isin(event_times).values
    log.info(f"  CUSUM events: {int(cusum_mask.sum()):,}  [{time.perf_counter()-t0:.1f}s]")

    labels_df = labels_full[cusum_mask].reset_index(drop=True)
    feats_to_use = feats_lab[cusum_mask].reset_index(drop=True)

    df = feats_to_use.copy()
    df["t1_time"] = labels_df["t1_time"].values
    df["t1"] = df["t1_time"]
    df["label_raw"] = labels_df["label"].values
    df["ret"] = labels_df["return"].values

    dist_raw = pd.Series(df["label_raw"]).value_counts().sort_index().to_dict()

    # main.py: drop flat (label == 0), then binary long(=1) vs short(=0)
    df = df[df["label_raw"] != 0].copy()
    df["label"] = (df["label_raw"] == 1).astype(int)
    df.drop(columns=["label_raw"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    stats = {
        "bars_total":         int(len(bars)),
        "bars_labelable":     int(labelable),
        "cusum_events":       int(cusum_mask.sum()),
        "label_dist_raw":     dist_raw,
        "rows_after_binary":  int(len(df)),
        "label_dist_binary":  df["label"].value_counts().to_dict(),
        "max_hold":           int(max_hold),
        "pt":                 float(pt),
        "sl":                 float(sl),
    }
    log.info(f"  binary df: {len(df):,}  dist={stats['label_dist_binary']}")
    return df, cutoff_ts, stats


def run_one_trial(trial: dict, rf_params: dict) -> dict:
    tag = trial_tag(trial)
    log, log_path = setup_trial_logger(tag)
    log.info("=" * 72)
    log.info(f"Trial config: {json.dumps(trial)}")
    log.info(f"Fee model: taker {FEE_BPS_PER_SIDE} bps + slippage {SLIPPAGE_BPS_PER_SIDE} bps per side")
    log.info(f"RF params: {rf_params}")
    log.info("=" * 72)
    t_start = time.perf_counter()

    df, cutoff_ts, stats = build_full_dataset(trial, log)

    features = FEATURE_SETS[trial["features_set"]]
    df_trainval = df[df["close_time"] < cutoff_ts].reset_index(drop=True)
    df_test = df[df["close_time"] >= cutoff_ts].reset_index(drop=True)
    log.info(f"split: trainval={len(df_trainval):,}  test={len(df_test):,}")
    log.info(f"trainval dist={df_trainval['label'].value_counts().to_dict()}  "
             f"test dist={df_test['label'].value_counts().to_dict()}")

    if len(df_test) == 0 or df_test["label"].nunique() < 2:
        log.error("Held-out test slice is empty or single-class — aborting trial.")
        return {"trial_id": trial["trial_id"], "tag": tag, "error": "empty_test", "log_path": str(log_path)}

    # ── Stage 1: purged 5-fold CV on trainval ────────────────────────────
    t_cv = time.perf_counter()
    log.info(f"--- Stage 1: purged {N_SPLITS}-fold CV on trainval ---")
    _, cv_oos, fold_data = run_cv(
        df_trainval,
        n_splits=N_SPLITS,
        embargo_pct=EMBARGO_PCT,
        max_hold=stats["max_hold"],
        rf_params=rf_params,
        feature_cols=features,
        time_decay_c=1.0,
        use_time_decay=USE_TIME_DECAY_FIXED,
        use_overlap=USE_OVERLAP_FIXED,
    )
    log.info(f"CV done  [{time.perf_counter()-t_cv:.1f}s]")

    cv_metrics = classification_metrics(
        cv_oos["y_true"].to_numpy(), cv_oos["y_pred"].to_numpy(),
    )
    # cv_oos already has ret (renamed from WEIGHT_COL) — join t1_time so the
    # non-overlap backtest knows when each trade resolves.
    cv_oos_full = cv_oos.join(df_trainval[["t1_time"]], how="left")
    cv_bt = realistic_backtest(
        y_pred=cv_oos_full["y_pred"].to_numpy(),
        triple_barrier_log_ret=cv_oos_full["ret"].to_numpy(),
        close_times=cv_oos_full["close_time"].to_numpy(),
        t1_times=cv_oos_full["t1_time"].to_numpy(),
    )
    cv_oneBar_strat = strategy_log_returns(cv_oos["y_pred"].to_numpy(), cv_oos["log_return"].to_numpy())
    cv_bars_per_year = compute_bars_per_year(pd.DatetimeIndex(cv_oos["close_time"]))
    cv_one_bar_sharpe = safe(sharpe_ratio(cv_oneBar_strat, cv_bars_per_year))
    cv_one_bar_pnl = safe(float(np.exp(np.nansum(cv_oneBar_strat.to_numpy(dtype=float))) - 1.0))

    log.info(
        f"CV classif: acc={cv_metrics['accuracy']:.4f}  "
        f"f1_macro={cv_metrics['f1_macro']:.4f}  f1_long={cv_metrics['f1_long']:.4f}  "
        f"auc_pr={cv_metrics['auc_pr']:.4f}"
    )
    log.info(
        f"CV one-bar  : sharpe={cv_one_bar_sharpe:+.4f}  net_pnl={cv_one_bar_pnl:+.4f}"
    )
    log.info(
        f"CV realistic: trades={cv_bt['n_trades']}  "
        f"gross_pnl={cv_bt['gross_pnl_net']:+.4f}  net_pnl={cv_bt['net_pnl_after_fees']:+.4f}  "
        f"wr={cv_bt['win_rate_net']:.4f}  pf={cv_bt['profit_factor_net']:.4f}  "
        f"mdd_log={cv_bt['max_drawdown_log_net']:.4f}  "
        f"sharpe_trade={cv_bt['sharpe_trade_annualized']:.4f}"
    )

    for fd in fold_data:
        f_metrics = classification_metrics(fd["y_true"], fd["y_pred"])
        log.info(
            f"  fold {fd['fold']}: train={fd['train_size']:,} test={fd['test_size']:,}  "
            f"acc={f_metrics['accuracy']:.4f}  f1_long={f_metrics['f1_long']:.4f}  "
            f"f1_macro={f_metrics['f1_macro']:.4f}  auc_pr={f_metrics['auc_pr']:.4f}"
        )

    # ── Stage 2: retrain on full trainval, evaluate on held-out test ─────
    t_final = time.perf_counter()
    log.info(f"--- Stage 2: final model on trainval, score on held-out test ---")

    X_train = df_trainval[features].to_numpy(dtype=float)
    y_train = df_trainval["label"].to_numpy()
    X_test = df_test[features].to_numpy(dtype=float)
    y_test = df_test["label"].to_numpy()

    final_model = RandomForestClassifier(**rf_params)
    final_model.fit(X_train, y_train)
    log.info(f"final RF trained on {len(df_trainval):,}  [{time.perf_counter()-t_final:.1f}s]")

    y_test_pred = final_model.predict(X_test)
    test_proba = final_model.predict_proba(X_test)
    # Long class is class 1 (RandomForest.classes_ sorts ascending).
    long_class_col = int(np.where(final_model.classes_ == 1)[0][0])
    test_proba_long = test_proba[:, long_class_col]
    test_metrics = classification_metrics(y_test, y_test_pred)
    test_bt = realistic_backtest(
        y_pred=y_test_pred,
        triple_barrier_log_ret=df_test["ret"].to_numpy(),
        close_times=df_test["close_time"].to_numpy(),
        t1_times=df_test["t1_time"].to_numpy(),
    )
    test_next_log_ret = df_test["log_return"].shift(-1).fillna(0.0).to_numpy()
    test_one_bar_strat = strategy_log_returns(y_test_pred, test_next_log_ret)
    test_bars_per_year = compute_bars_per_year(pd.DatetimeIndex(df_test["close_time"]))
    test_one_bar_sharpe = safe(sharpe_ratio(test_one_bar_strat, test_bars_per_year))
    test_one_bar_pnl = safe(float(np.exp(np.nansum(test_one_bar_strat.to_numpy(dtype=float))) - 1.0))

    log.info(
        f"TEST classif: acc={test_metrics['accuracy']:.4f}  "
        f"f1_macro={test_metrics['f1_macro']:.4f}  f1_long={test_metrics['f1_long']:.4f}  "
        f"auc_pr={test_metrics['auc_pr']:.4f}"
    )
    log.info(
        f"TEST realistic: trades={test_bt['n_trades']}  "
        f"gross_pnl={test_bt['gross_pnl_net']:+.4f}  "
        f"net_pnl={test_bt['net_pnl_after_fees']:+.4f}  "
        f"wr={test_bt['win_rate_net']:.4f}  pf={test_bt['profit_factor_net']:.4f}  "
        f"mdd_log={test_bt['max_drawdown_log_net']:.4f}  "
        f"sharpe_trade={test_bt['sharpe_trade_annualized']:.4f}  "
        f"trades_per_year={test_bt['trades_per_year']:.1f}"
    )

    log.info("TEST classification report:\n" + classification_report(
        y_test, y_test_pred, labels=[0, 1], target_names=["flat", "long"], zero_division=0))

    # ── Persist artifacts: final model + held-out predictions ────────────
    model_path = MODELS_DIR / f"{tag}.pkl"
    joblib.dump(final_model, model_path)
    log.info(f"model saved → {model_path}")

    test_preds_df = pd.DataFrame({
        "close_time":         df_test["close_time"].values,
        "t1_time":            df_test["t1_time"].values,
        "y_true":             y_test,
        "y_pred":             y_test_pred,
        "proba_long":         test_proba_long,
        "triple_barrier_ret": df_test["ret"].values,
    })
    test_preds_path = PREDS_DIR / f"{tag}_test_preds.parquet"
    test_preds_df.to_parquet(test_preds_path, index=False)
    log.info(f"test preds saved → {test_preds_path}")

    # Persist CV OOS predictions too — useful for threshold tuning that
    # picks a cut on trainval rather than on the held-out slice.
    cv_proba_col = "proba_1" if "proba_1" in cv_oos_full.columns else None
    cv_preds_df = pd.DataFrame({
        "close_time":         cv_oos_full["close_time"].values,
        "t1_time":            cv_oos_full["t1_time"].values,
        "y_true":             cv_oos_full["y_true"].values,
        "y_pred":             cv_oos_full["y_pred"].values,
        "proba_long":         cv_oos_full[cv_proba_col].values if cv_proba_col else np.nan,
        "triple_barrier_ret": cv_oos_full["ret"].values,
    })
    cv_preds_path = PREDS_DIR / f"{tag}_cv_preds.parquet"
    cv_preds_df.to_parquet(cv_preds_path, index=False)
    log.info(f"CV preds saved → {cv_preds_path}")

    # Trial-level config for easy loading from the notebook.
    config_payload = {
        **trial,
        "tag": tag,
        "symbol": SYMBOL,
        "pt": stats["pt"],
        "sl": stats["sl"],
        "max_hold": stats["max_hold"],
        "features": features,
        "rf_params": {k: v for k, v in rf_params.items() if k != "n_jobs"},
        "fee_bps_per_side": FEE_BPS_PER_SIDE,
        "slippage_bps_per_side": SLIPPAGE_BPS_PER_SIDE,
        "non_overlapping": True,
        "test_ratio": TEST_RATIO,
        "n_splits": N_SPLITS,
        "embargo_pct": EMBARGO_PCT,
        "paths": {
            "model":      str(Path(model_path).resolve()),
            "test_preds": str(Path(test_preds_path).resolve()),
            "cv_preds":   str(Path(cv_preds_path).resolve()),
            "log":        str(Path(log_path).resolve()),
        },
    }
    config_path = CONFIGS_DIR / f"{tag}.json"
    config_path.write_text(json.dumps(config_payload, indent=2, default=str))
    log.info(f"config saved → {config_path}")

    # ── Stage 3: feature importance ──────────────────────────────────────
    log.info("--- Stage 3: feature importance ---")
    t_imp = time.perf_counter()

    mdi = mdi_importance(final_model, features)
    log.info(f"MDI computed  [{time.perf_counter()-t_imp:.1f}s]")
    t_imp = time.perf_counter()

    mda = mda_importance(final_model, X_test, y_test, features, n_repeats=3)
    log.info(f"MDA (n_repeats=3) computed  [{time.perf_counter()-t_imp:.1f}s]")
    t_imp = time.perf_counter()

    sfi = sfi_importance(df_trainval, df_test, features, rf_params)
    log.info(f"SFI computed  [{time.perf_counter()-t_imp:.1f}s]")

    log.info("MDI top:\n" + mdi.head(15).to_string())
    log.info("MDA top:\n" + mda.head(15).to_string())
    log.info("SFI top:\n" + sfi.head(15).to_string())

    imp_path = IMPORTANCE_DIR / f"{tag}_importance.csv"
    merged = (
        mdi.rename(columns={"mean": "mdi_mean", "std": "mdi_std"})
        .join(mda.rename(columns={"mean": "mda_mean", "std": "mda_std"}), how="outer")
        .join(sfi.rename(columns={"mean": "sfi_mean", "std": "sfi_std"}), how="outer")
        .sort_values("mdi_mean", ascending=False)
    )
    merged.to_csv(imp_path)
    log.info(f"importance table saved → {imp_path}")

    total_wall = time.perf_counter() - t_start
    log.info(f"=== trial done in {total_wall:.1f}s ===")

    # ── Master CSV row ──────────────────────────────────────────────────
    def topk(s: pd.Series, k: int = 5) -> str:
        return ",".join(s.head(k).index.tolist())

    row = {
        "tag": tag,
        "trial_id": trial["trial_id"],
        "bar_size": trial["bar_size"],
        "cusum_h": trial["cusum_h"],
        "features_set": trial["features_set"],
        "features_count": len(features),
        "max_hold_slot": trial["max_hold_slot"],
        "max_hold": stats["max_hold"],
        "pt_sl": trial["pt_sl"],
        "pt": stats["pt"],
        "sl": stats["sl"],
        "span": trial["span"],
        "bars_total": stats["bars_total"],
        "rows_after_binary": stats["rows_after_binary"],
        "n_trainval": int(len(df_trainval)),
        "n_test": int(len(df_test)),
        "trainval_balance_long": float((df_trainval["label"] == 1).mean()),
        "test_balance_long": float((df_test["label"] == 1).mean()),
        "fee_bps_per_side": FEE_BPS_PER_SIDE,
        "slippage_bps_per_side": SLIPPAGE_BPS_PER_SIDE,
        "round_trip_cost_bps": cv_bt["round_trip_cost_bps"],
        # CV
        "cv_accuracy": cv_metrics["accuracy"],
        "cv_f1_macro": cv_metrics["f1_macro"],
        "cv_f1_long": cv_metrics["f1_long"],
        "cv_auc_pr": cv_metrics["auc_pr"],
        "cv_one_bar_sharpe": cv_one_bar_sharpe,
        "cv_one_bar_net_pnl": cv_one_bar_pnl,
        "cv_n_trades": cv_bt["n_trades"],
        "cv_gross_pnl": cv_bt["gross_pnl_net"],
        "cv_net_pnl_after_fees": cv_bt["net_pnl_after_fees"],
        "cv_win_rate": cv_bt["win_rate_net"],
        "cv_profit_factor": cv_bt["profit_factor_net"],
        "cv_mdd_log": cv_bt["max_drawdown_log_net"],
        "cv_sharpe_trade": cv_bt["sharpe_trade_annualized"],
        # Held-out test
        "test_accuracy": test_metrics["accuracy"],
        "test_f1_macro": test_metrics["f1_macro"],
        "test_f1_long": test_metrics["f1_long"],
        "test_f1_flat": test_metrics["f1_flat"],
        "test_prec_long": test_metrics["prec_long"],
        "test_recall_long": test_metrics["recall_long"],
        "test_auc_pr": test_metrics["auc_pr"],
        "test_one_bar_sharpe": test_one_bar_sharpe,
        "test_one_bar_net_pnl": test_one_bar_pnl,
        "test_n_trades": test_bt["n_trades"],
        "test_gross_pnl": test_bt["gross_pnl_net"],
        "test_net_pnl_after_fees": test_bt["net_pnl_after_fees"],
        "test_win_rate": test_bt["win_rate_net"],
        "test_profit_factor": test_bt["profit_factor_net"],
        "test_mdd_log": test_bt["max_drawdown_log_net"],
        "test_sharpe_trade": test_bt["sharpe_trade_annualized"],
        "test_trades_per_year": test_bt["trades_per_year"],
        # Importance summaries
        "top5_mdi": topk(mdi["mean"]),
        "top5_mda": topk(mda["mean"]),
        "top5_sfi": topk(sfi["mean"]),
        # Paths
        "log_path": str(log_path),
        "model_path": str(model_path),
        "importance_path": str(imp_path),
        "test_preds_path": str(test_preds_path),
        "cv_preds_path": str(cv_preds_path),
        "config_path": str(config_path),
        "wall_seconds": round(total_wall, 1),
    }
    return row


# ── Driver ────────────────────────────────────────────────────────────────


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    rf_n_jobs = os.cpu_count() or 1
    rf_params = {**RF_PARAMS_FIXED, "n_jobs": rf_n_jobs}

    master_csv = OUT_ROOT / f"top_trials_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    rows: list[dict] = []

    print(f"Running {len(TRIALS)} trials. Master CSV → {master_csv}")
    for i, trial in enumerate(TRIALS, 1):
        print(f"\n[{i}/{len(TRIALS)}] {trial_tag(trial)}")
        try:
            row = run_one_trial(trial, rf_params)
        except Exception as e:  # noqa: BLE001
            row = {"trial_id": trial["trial_id"], "tag": trial_tag(trial), "error": repr(e)}
            print(f"  ERROR: {e!r}")
        rows.append(row)
        # Persist after every trial so a crash doesn't lose completed work.
        pd.DataFrame(rows).to_csv(master_csv, index=False)
        print(f"  → row appended ({len(rows)}/{len(TRIALS)} done)")

    print(f"\nDone. Master results: {master_csv}")
    print(f"Per-trial logs in: {LOGS_DIR}/")


if __name__ == "__main__":
    main()
