import joblib
import logging
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import classification_report, precision_recall_curve, auc

from src.features.engineer import (
    adx_features,
    bollinger_bands,
    frac_diff_ffd,
    log_returns,
    macd_features,
    rsi_feature,
    stochastic_rsi,
)
from src.model.metrics import (
    sharpe_ratio,
    strategy_log_returns,
    compute_bars_per_year
)
from src.model.train import SYMBOL, run_cv, train_final
from src.pre_process.trippler_barrier import getDailyVol, label_bars

CUSUM_H = 0.003

OPTUNA_FEATURE_COLS = [
    "frac_diff",
    "bb_pct_b", "bb_width",
    "rsi",
    "macd", "macd_signal", "macd_hist",
    "adx", "adx_pdi", "adx_mdi",
    "stoch_rsi_k", "stoch_rsi_d",
]

MAX_HOLD_RANGES: dict[int, tuple[int, int, int]] = {
    3: (1000, 5000, 1000),
    5:  (500, 2500, 500),
    # 10: (250, 1250, 250),
    15: (150,  750, 150),
    # 20: (125,  625, 125),
    25: (100,  500, 100),
    50: (50,   250,  50),
    60: (25,   125,  25),
}

FIXED_RF_PARAMS: dict = {
    "n_estimators": 500,
    "min_samples_leaf": 50,
    "class_weight": "balanced",
    "n_jobs":  -1,
    "random_state": 42,
    "max_depth": 4
}

SEARCH_SPACE: dict[str, list] = {
    "bar_size": [3, 5, 15, 25, 35, 50, 60],
    "pt_sl": ["1.2_1.0", "1.0_1.0", "1.0_1.2"],
    "span":   list(range(10, 70, 20)),
    "max_hold_slot": [0, 1, 2, 3, 4],
    "use_cusum":     [True],
    "use_time_decay": [True, False],
    "use_overlap": [True,False]
}

N_TRIALS: int = (
    len(SEARCH_SPACE["bar_size"])
    * len(SEARCH_SPACE["pt_sl"])
    * len(SEARCH_SPACE["span"])
    * len(SEARCH_SPACE["max_hold_slot"])
    * len(SEARCH_SPACE["use_cusum"])
    * len(SEARCH_SPACE["use_time_decay"])
    * len(SEARCH_SPACE["use_overlap"])
)  # 7 × 3 × 4 × 5 × 1 × 2 × 2 = 1680

STUDY_NAME = "dollar_bar_gridsearch"

TEST_RATIO: float = 0.20


def safe(v: float) -> float:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return 0.0
    return float(v)


def setup_logging(run_name: str) -> logging.Logger:
    Path("logs").mkdir(exist_ok=True)
    log = logging.getLogger("optuna_search")
    log.setLevel(logging.DEBUG)
    if log.handlers:
        log.handlers.clear()
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    fh = logging.FileHandler(f"logs/{run_name}.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    log.addHandler(ch)
    log.addHandler(fh)
    return log


def cusum_filter(close: pd.Series, h: float = CUSUM_H) -> pd.Index:
    """Symmetric CUSUM filter (AFML Ch.2). close must have DatetimeIndex."""
    s_pos, s_neg = 0.0, 0.0
    events: list[object] = []
    log_diff = np.log(close).diff()
    for t, x in zip(log_diff.index[1:], log_diff.values[1:]):
        if np.isnan(x):
            continue
        s_pos = max(0.0, s_pos + x)
        s_neg = min(0.0, s_neg + x)
        if s_pos > h or s_neg < -h:
            s_pos, s_neg = 0.0, 0.0
            events.append(t)
    return pd.Index(events)


def compute_bar_features(bars: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all indicators on the FULL contiguous bar sequence.
    Must be called BEFORE any CUSUM filtering so rolling windows have no gaps.
    Returns a DataFrame with the same RangeIndex as bars.
    """
    lr = log_returns(bars)
    log_price = pd.Series(np.log(bars["close"].to_numpy(dtype=np.float64)), index=bars.index)
    fd = frac_diff_ffd(log_price, d=1.0)  # placeholder: run_cv overrides per fold with ADF-selected d
    bb_pct_b, bb_width = bollinger_bands(bars)
    rsi_val = rsi_feature(bars)
    macd_line, macd_sig, macd_hist = macd_features(bars)
    adx_val, adx_pdi, adx_mdi = adx_features(bars)
    stoch_k, stoch_d = stochastic_rsi(bars)

    return pd.DataFrame({
        "open_time":    bars["open_time"].values,
        "close_time":   bars["close_time"].values,
        "close":        bars["close"].values,
        "log_return":   lr.values,
        "frac_diff":    fd.values,
        "bb_pct_b":     bb_pct_b.values,
        "bb_width":     bb_width.values,
        "rsi":          rsi_val.values,
        "macd":         macd_line.values,
        "macd_signal":  macd_sig.values,
        "macd_hist":    macd_hist.values,
        "adx":          adx_val.values,
        "adx_pdi":      adx_pdi.values,
        "adx_mdi":      adx_mdi.values,
        "stoch_rsi_k":  stoch_k.values,
        "stoch_rsi_d":  stoch_d.values,
    }, index=pd.RangeIndex(len(bars)))


# ── Optuna Grid Search ──────────────────────────────────────────────────────────

def objective(trial: optuna.Trial, log: logging.Logger, rf_params: dict) -> float:
    t_trial = time.perf_counter()
    # ── Suggest parameters ──
    bar_size = trial.suggest_categorical("bar_size", SEARCH_SPACE["bar_size"])
    tp_sl_key = trial.suggest_categorical("pt_sl", SEARCH_SPACE["pt_sl"])
    span = trial.suggest_int("span", 10, 70, step=20)
    max_hold_slot = trial.suggest_int("max_hold_slot", 0, 4)
    use_cusum = trial.suggest_categorical("use_cusum", SEARCH_SPACE["use_cusum"])
    use_time_decay = trial.suggest_categorical("use_time_decay", SEARCH_SPACE["use_time_decay"])
    use_overlap = trial.suggest_categorical("use_overlap", SEARCH_SPACE["use_overlap"])

    low, high, step = MAX_HOLD_RANGES[bar_size]
    max_hold = low + max_hold_slot * step
    
    tp_sl = tp_sl_key.split("_")
    pt, sl = float(tp_sl[0]), float(tp_sl[1])

    log.debug(
        f"Trial {trial.number:>6} | bar={bar_size}M  pt/sl={tp_sl_key}"
        f" span={span}  hold={max_hold}  cusum={use_cusum}  time_decay={use_time_decay}  overlap={use_overlap}"
    )

    # ── 1. Load bars — slice to train/val; test set is never touched during Optuna ──
    t0 = time.perf_counter()
    bars = pd.read_parquet(f"data/processed/dollar_bars_{bar_size}_{SYMBOL}.parquet")
    n_test = int(len(bars) * TEST_RATIO)
    bars = bars.iloc[: len(bars) - n_test].reset_index(drop=True)
    log.debug(f"  bars loaded: {len(bars):,} trainval  (test withheld: {n_test:,})  [{time.perf_counter()-t0:.1f}s]")

    # ── 2. Features on FULL bars (before any filtering — avoids rolling-window gaps) ──
    features_full = compute_bar_features(bars)

    # ── 3. Daily vol on full bars (EWM is causal — no leakage) ──
    close_ts = pd.Series(bars["close"].values, index=pd.DatetimeIndex(bars["close_time"]))
    vol_full = getDailyVol(close_ts, span=span)
    vol_full = vol_full[~vol_full.index.duplicated(keep="last")]

    # ── 4. CUSUM filter — select which bars to label (after features are computed) ──
    event_times = cusum_filter(close_ts)
    cusum_mask = bars["close_time"].isin(event_times).values
    bars_to_label = bars[cusum_mask].reset_index(drop=True)
    feats_to_use  = features_full[cusum_mask].reset_index(drop=True)
    # vol: reindex to filtered bar timestamps, forward-fill any gaps
    vol_to_label = vol_full.reindex(
        pd.DatetimeIndex(bars_to_label["close_time"])
    ).ffill()
    log.debug(f"CUSUM: {len(bars_to_label):,} bars selected  [{time.perf_counter()-t0:.1f}s]")

    # ── 5. Label selected bars using tick data ──
    t_label = time.perf_counter()
    labels_df = label_bars(bars_to_label, vol_to_label, pt=pt, sl=sl, max_hold=max_hold)
    log.debug(
        f"  labels: {len(labels_df):,}"
        f"  dist={labels_df['label'].value_counts().to_dict()}"
        f"  [{time.perf_counter()-t_label:.1f}s]"
    )

    # ── 6. Merge features + labels (features already computed on full sequence) ──
    labelable = len(labels_df)
    df = feats_to_use.iloc[:labelable].copy()
    df["t1_time"] = labels_df["t1_time"].values
    df["t1"] = df["t1_time"]
    df["label"] = labels_df["label"].values
    df["ret"] = labels_df["return"].values

    # ── 7. Long-only binary: drop hold(0), remap short(-1) → 0 ──
    df = df[df["label"] != 0].copy()
    df["label"] = (df["label"] == 1).astype(int)
    log.debug(f"Binary: {len(df):,}  dist={df['label'].value_counts().to_dict()}")

    # ── 8. 5-fold Purged CV ──
    t_cv = time.perf_counter()
    _, oos_df, fold_data = run_cv(
        df,
        max_hold=max_hold,
        rf_params=rf_params,
        feature_cols=OPTUNA_FEATURE_COLS,
        use_time_decay=use_time_decay,
        use_overlap=use_overlap,
    )
    fold_results = []
    # Per-fold log
    for fd in fold_data:
        fold_report = classification_report(
            fd["y_true"], fd["y_pred"],
            labels=[0, 1],
            target_names=["flat", "long"],
            output_dict=True,
            zero_division=0,
        )
        train_acc = float((fd["y_train_true"] == fd["y_train_pred"]).mean())
        fold_strat = pd.Series(fd["y_pred"].astype(float) * fd["bar_ret"])
        bars_per_year = compute_bars_per_year(pd.DatetimeIndex(fd["timestamps"]))

        train_accuracy = train_acc
        accuracy = safe(fold_report["accuracy"])
        f1_long = safe(fold_report["long"]["f1-score"])
        f1_flat = safe(fold_report["flat"]["f1-score"])
        recall_long = safe(fold_report["long"]["recall"])
        recall_flat = safe(fold_report["flat"]["recall"])
        prec_long = safe(fold_report["long"]["precision"])
        prec_flat = safe(fold_report["flat"]["precision"])
        f1_macro = safe(fold_report["macro avg"]["f1-score"])
        sharpe = safe(sharpe_ratio(fold_strat, bars_per_year))
        overfit_gap = train_accuracy - accuracy
        precision, recall, _ = precision_recall_curve(
            fd["y_true"], fd["y_pred"]
        )
        auc_pr = safe(auc(recall, precision))


        fold_results.append({
            "train_acc": train_accuracy,
            "accuracy": accuracy,
            "f1_long": f1_long,
            "f1_flat": f1_flat,
            "recall_long": recall_long,
            "recall_flat": recall_flat,
            "prec_long": prec_long,
            "prec_flat": prec_flat,
            "f1_macro": f1_macro,
            "sharpe": sharpe,
            "overfit_gap": overfit_gap,
            "auc_pr": auc_pr
        })

        for k, v in fold_results[-1].items():
            trial.set_user_attr(f"fold{fd['fold']}_{k}", v)

        log.debug(
            f"fold {fd['fold']}:"
            f"train={fd['train_size']:,}  test={fd['test_size']:,}"
            f"train_acc={train_acc:.4f}  acc={accuracy:.4f}"
            f"f1_macro={f1_macro:.4f}  f1_long={f1_long:.4f}  f1_flat={f1_flat:.4f}"
            f"prec_long={prec_long:.4f}  rec_long={recall_long:.4f}"
            f"prec_flat={prec_flat:.4f}  rec_flat={recall_flat:.4f}"
            f"sharpe={sharpe:.4f}  auc_pr={auc_pr:.4f}  overfit_gap={overfit_gap:.4f}"    
        )             
        
    log.debug(f"CV done [{time.perf_counter()-t_cv:.1f}s]")

    # ── 9. OOS metrics ──
    strat_ret = strategy_log_returns(
        oos_df["y_pred"].to_numpy(),
        oos_df["log_return"].to_numpy(),
    )
    oos_bars_per_year = compute_bars_per_year(pd.DatetimeIndex(oos_df["close_time"]))
    oos_sharpe = safe(sharpe_ratio(strat_ret, oos_bars_per_year))

    oos_report = classification_report(
        oos_df["y_true"].to_numpy(),
        oos_df["y_pred"].to_numpy(),
        labels=[0, 1],
        target_names=["flat", "long"],
        output_dict=True,
        zero_division=0,
    )
    precision, recall, _ = precision_recall_curve(
        oos_df["y_true"].to_numpy(),
        oos_df["y_pred"].to_numpy() 
    )
    auc_pr = safe(auc(recall, precision))
    macro_f1 = safe(oos_report["macro avg"]["f1-score"])
    accuracy = safe(oos_report["accuracy"])
    f1_long = safe(oos_report["long"]["f1-score"])
    f1_flat = safe(oos_report["flat"]["f1-score"])
    recall_long = safe(oos_report["long"]["recall"])
    recall_flat = safe(oos_report["flat"]["recall"])
    prec_long = safe(oos_report["long"]["precision"])
    prec_flat = safe(oos_report["flat"]["precision"])

    trial.set_user_attr("oos_auc_pr", auc_pr)
    trial.set_user_attr("oos_sharpe", oos_sharpe)
    trial.set_user_attr("oos_f1_macro", macro_f1)
    trial.set_user_attr("oos_accuracy", accuracy)
    trial.set_user_attr("oos_f1_long", f1_long)
    trial.set_user_attr("oos_f1_flat", f1_flat)
    trial.set_user_attr("oos_recall_long", recall_long)
    trial.set_user_attr("oos_recall_flat", recall_flat)
    trial.set_user_attr("oos_prec_long", prec_long)
    trial.set_user_attr("oos_prec_flat", prec_flat)
    trial.set_user_attr("max_hold", max_hold)
    trial.set_user_attr("n_bars", len(df))

    # ── 10. Save model for this trial (all models saved — task req 8) ──
    t_save = time.perf_counter()
    Path("data/processed/models").mkdir(parents=True, exist_ok=True)
    model_path = f"data/processed/models/trial_{trial.number:06d}_bar{bar_size}_{SYMBOL}.pkl"
    train_final(
        df, rf_params,
        max_hold=max_hold,
        feature_cols=OPTUNA_FEATURE_COLS,
        out_path=model_path,
        use_time_decay=use_time_decay,
        use_overlap=use_overlap
    )
    log.debug(f"Model saved → {model_path}  [{time.perf_counter()-t_save:.1f}s]")

    wall = time.perf_counter() - t_trial
    log.info(
        f"Trial {trial.number:>6} | bar={bar_size}M  pt/sl={tp_sl_key}"
        f"span={span}  hold={max_hold}  cusum={use_cusum}"
        f" | sharpe={oos_sharpe:+.4f}  f1_macro={macro_f1:.4f}"
        f"  acc={accuracy:.4f}  f1_long={f1_long:.4f}"
        f" prec_long={prec_long:.4f}  rec_long={recall_long:.4f}"
        f"  | {wall:.0f}s"
    )
    return oos_sharpe


def optuna_main(
    n_trials: int = N_TRIALS,
    study_name: str = STUDY_NAME,
    optuna_n_jobs: int = 1,
    rf_n_jobs: int | None = None,
    storage: str | None = None,
) -> None:
    import os
    # Pin BLAS/OpenBLAS/MKL to 1 thread to prevent over-subscription when
    # RF's joblib workers each spawn their own BLAS threads.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    run_name = f"optuna_{study_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log = setup_logging(run_name)

    cpu_count = os.cpu_count() or 1
    if rf_n_jobs is None:
        rf_n_jobs = max(1, cpu_count // max(1, optuna_n_jobs))
    rf_params_runtime = {**FIXED_RF_PARAMS, "n_jobs": rf_n_jobs}

    log.info("=" * 72)
    log.info(f"Optuna Grid Search  —  {SYMBOL}")
    log.info(f"Study: {study_name}  |  Trials: {n_trials:,}  |  Space: {N_TRIALS:,}")
    log.info(f"Features ({len(OPTUNA_FEATURE_COLS)}): {OPTUNA_FEATURE_COLS}")
    log.info(f"CUSUM h={CUSUM_H}  |  Binary long-only labels")
    log.info(
        f"RF (fixed): n_est={FIXED_RF_PARAMS['n_estimators']}"
        f"  leaf={FIXED_RF_PARAMS['min_samples_leaf']}"
    )
    log.info(
        f"Parallel: optuna_n_jobs={optuna_n_jobs}"
        f"  rf_n_jobs={rf_n_jobs}"
        f"  cpu_count={cpu_count}"
        f"  storage={storage or 'in-memory'}"
    )
    log.info("=" * 72)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        sampler=optuna.samplers.GridSampler(SEARCH_SPACE),
        storage=storage,
        load_if_exists=True,
    )

    flow_start = time.perf_counter()

    study.optimize(
        lambda trial: objective(trial, log, rf_params_runtime),
        n_trials=n_trials,
        n_jobs=optuna_n_jobs,
        show_progress_bar=True,
    )

    total = time.perf_counter() - flow_start

    log.info("=" * 72)
    log.info(f"Search complete in {total / 60:.1f} min  ({int(total)}s)")
    best = study.best_trial
    log.info(f"Best trial #{best.number}  |  OOS Sharpe = {best.value:.4f}")
    log.info(f"  Params  : {best.params}")
    log.info(f"  Metrics : {best.user_attrs}")
    out_csv = "optuna_trials.csv"
    study.trials_dataframe().to_csv(out_csv, index=False)
    log.info(f"Trial history → {out_csv}  ({len(study.trials)} rows)")
    log.info("=" * 72)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Optuna grid search")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Parallel Optuna trials (use 1 with --multiprocess)")
    parser.add_argument("--rf-jobs", type=int, default=None,
                        help="RF cores per worker (default: cpu_count // n-jobs)")
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    parser.add_argument("--study-name", type=str, default=STUDY_NAME)
    parser.add_argument("--storage", type=str, default=None,
                        help="e.g. sqlite:///optuna.db  (required for --multiprocess)")
    args = parser.parse_args()
    optuna_main(
        n_trials=args.n_trials,
        study_name=args.study_name,
        optuna_n_jobs=args.n_jobs,
        rf_n_jobs=args.rf_jobs,
        storage=args.storage,
    )
