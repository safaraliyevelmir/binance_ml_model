import logging
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import classification_report

from src.features.engineer import (
    adx_features,
    bollinger_bands,
    dollar_volume_imbalance,
    find_min_d,
    log_returns,
    macd_features,
    relative_duration,
    rsi_feature,
    stochastic_rsi,
)
from src.model.metrics import (
    full_report,
    max_drawdown,
    plot_equity_curve,
    profit_factor,
    sharpe_ratio,
    strategy_log_returns,
    win_rate,
)
from src.model.train import (
    MAX_HOLD,
    OOS_PATH,
    RF_PARAMS,
    SYMBOL,
    run_cv,
    train_final,
)
from src.pre_process.trippler_barrier import PT_SL, SPAN, getDailyVol, label_bars

# ── Constants ──────────────────────────────────────────────────────────────────

EQUITY_CURVE_PATH = f"data/processed/equity_curve_{SYMBOL}.png"
DEFAULT_BAR_SIZE = 25

CUSUM_H = 0.003

# Task req 6: only 5 indicators + fracdiff price
OPTUNA_FEATURE_COLS = [
    "frac_diff",
    "bb_pct_b", "bb_width",
    "rsi",
    "macd", "macd_signal", "macd_hist",
    "adx", "adx_pdi", "adx_mdi",
    "stoch_rsi_k", "stoch_rsi_d",
]

# Task req 4: dynamic max_hold per bar threshold (low, high, step)
MAX_HOLD_RANGES: dict[int, tuple[int, int, int]] = {
    5:  (500, 2500, 500),
    10: (250, 1250, 250),
    15: (150,  750, 150),
    20: (125,  625, 125),
    25: (100,  500, 100),
    50: (50,   250,  50),
}

# RF params fixed per López de Prado (AFML): max_features=1, no RF grid search
FIXED_RF_PARAMS: dict = {
    "n_estimators":    500,
    "max_features":    1,
    "max_depth":       None,
    "min_samples_leaf": 50,
    "class_weight":   "balanced",
    "n_jobs":         -1,
    "random_state":   42,
}

# GridSampler search space — max_hold encoded as slot 0..4, resolved per bar_size
SEARCH_SPACE: dict[str, list] = {
    "bar_size":      [5, 10, 15, 20, 25, 50],
    "pt_sl":         ["1.2_1.0", "1.0_1.0"],
    "span":          list(range(10, 101, 10)),
    "max_hold_slot": [0, 1, 2, 3, 4],
    "use_cusum":     [True, False],
}

N_TRIALS: int = (
    len(SEARCH_SPACE["bar_size"])
    * len(SEARCH_SPACE["pt_sl"])
    * len(SEARCH_SPACE["span"])
    * len(SEARCH_SPACE["max_hold_slot"])
    * len(SEARCH_SPACE["use_cusum"])
)  # = 6 × 2 × 10 × 5 × 2 = 1,200

STUDY_NAME = "dollar_bar_gridsearch"


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


def fold_metrics(fd: dict) -> dict:
    y_true = fd["y_true"]
    y_pred = fd["y_pred"]
    ret = fd["bar_ret"]

    test_report = classification_report(
        y_true, y_pred,
        labels=[-1, 0, 1],
        target_names=["short", "hold", "long"],
        output_dict=True,
        zero_division=0,
    )
    train_report = classification_report(
        fd["y_train_true"], fd["y_train_pred"],
        labels=[-1, 0, 1],
        target_names=["short", "hold", "long"],
        output_dict=True,
        zero_division=0,
    )
    strat = pd.Series(y_pred.astype(float) * ret)

    return {
        "train_size":       float(fd["train_size"]),
        "test_size":        float(fd["test_size"]),
        "test_accuracy":    safe(test_report["accuracy"]),
        "test_macro_f1":    safe(test_report["macro avg"]["f1-score"]),
        "test_weighted_f1": safe(test_report["weighted avg"]["f1-score"]),
        "test_f1_short":    safe(test_report["short"]["f1-score"]),
        "test_f1_hold":     safe(test_report["hold"]["f1-score"]),
        "test_f1_long":     safe(test_report["long"]["f1-score"]),
        "train_accuracy":   safe(train_report["accuracy"]),
        "train_macro_f1":   safe(train_report["macro avg"]["f1-score"]),
        "sharpe":           safe(sharpe_ratio(strat)),
        "max_drawdown":     safe(max_drawdown(strat)),
        "profit_factor":    safe(profit_factor(strat)),
        "win_rate":         safe(win_rate(strat)),
        "total_return":     safe(float(strat.sum())),
        "active_pct":       safe(float((y_pred != 0).mean())),
    }


def build_features_inline(bars: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    lr = log_returns(bars)
    rel_dur = relative_duration(bars)
    dvi = dollar_volume_imbalance(bars)
    _, fd = find_min_d(bars["close"])
    bb_pct_b, bb_width = bollinger_bands(bars)
    rsi_val = rsi_feature(bars)
    macd_line, macd_sig, macd_hist = macd_features(bars)
    adx_val, adx_pdi, adx_mdi = adx_features(bars)
    stoch_k, stoch_d = stochastic_rsi(bars)

    feats = pd.DataFrame({
        "open_time":    bars["open_time"],
        "close_time":   bars["close_time"],
        "t1_time":      labels["t1_time"],
        "close":        bars["close"],
        "log_return":   lr,
        "rel_duration": rel_dur,
        "dvi":          dvi,
        "frac_diff":    fd,
        "bb_pct_b":     bb_pct_b,
        "bb_width":     bb_width,
        "rsi":          rsi_val,
        "macd":         macd_line,
        "macd_signal":  macd_sig,
        "macd_hist":    macd_hist,
        "adx":          adx_val,
        "adx_pdi":      adx_pdi,
        "adx_mdi":      adx_mdi,
        "stoch_rsi_k":  stoch_k,
        "stoch_rsi_d":  stoch_d,
    })

    feats = feats.loc[labels.index].copy()
    feats["label"] = labels["label"].values
    feats["ret"]   = labels["return"].values
    return feats


# ── Optuna Grid Search ──────────────────────────────────────────────────────────

def objective(trial: optuna.Trial, log: logging.Logger, rf_params: dict | None = None) -> float:
    rf_params = rf_params or FIXED_RF_PARAMS
    t_trial = time.perf_counter()

    # ── Suggest parameters (data params only — RF is fixed per AFML) ──
    bar_size = trial.suggest_categorical("bar_size", SEARCH_SPACE["bar_size"])
    pt_sl_key = trial.suggest_categorical("pt_sl", SEARCH_SPACE["pt_sl"])
    span = trial.suggest_int("span", 10, 100, step=10)
    max_hold_slot = trial.suggest_int("max_hold_slot", 0, 4)
    use_cusum = trial.suggest_categorical("use_cusum", SEARCH_SPACE["use_cusum"])

    low, high, step = MAX_HOLD_RANGES[bar_size]
    max_hold = low + max_hold_slot * step
    pt, sl = (1.2, 1.0) if pt_sl_key == "1.2_1.0" else (1.0, 1.0)

    log.debug(
        f"Trial {trial.number:>6} | bar={bar_size}M  pt/sl={pt_sl_key}  span={span}"
        f"  hold={max_hold}  cusum={use_cusum}"
    )

    # ── Load bars ──
    t0 = time.perf_counter()
    bars = pd.read_parquet(f"data/processed/dollar_bars_{bar_size}_{SYMBOL}.parquet")
    log.debug(f"bars loaded: {len(bars):,}  [{time.perf_counter()-t0:.1f}s]")

    # ── CUSUM filter (task req 5) ──
    if use_cusum:
        close_ts = pd.Series(
            bars["close"].values,
            index=pd.DatetimeIndex(bars["close_time"]),
        )
        event_times = cusum_filter(close_ts)
        bars = bars[bars["close_time"].isin(event_times)].reset_index(drop=True)
        log.debug(f"  CUSUM: {len(bars):,} bars remain  [{time.perf_counter()-t0:.1f}s]")
        if len(bars) < 200:
            log.warning(f"Trial {trial.number}: too few bars after CUSUM, pruned")
            return float("-inf")

    # ── Volatility + labels (task reqs 2,3,4) ──
    close_ts = bars["close"].copy()
    close_ts.index = pd.DatetimeIndex(bars["close_time"])
    vol = getDailyVol(close_ts, span=span)

    t1 = time.perf_counter()
    labels_df = label_bars(bars, vol, pt=pt, sl=sl, max_hold=max_hold)
    log.debug(f"Labels: {len(labels_df):,}  dist={labels_df['label'].value_counts().to_dict()}  [{time.perf_counter()-t1:.1f}s]")

    # ── Features (task req 6) ──
    t2 = time.perf_counter()
    df = build_features_inline(bars, labels_df)
    df["t1"] = df["t1_time"]
    log.debug(f"Features: {len(df):,} rows  [{time.perf_counter()-t2:.1f}s]")

    # ── Long-only binary labels: drop hold(0), remap short(-1)→0 ──
    df = df[df["label"] != 0].copy()
    df["label"] = (df["label"] == 1).astype(int)
    log.debug(f"Binary: {len(df):,} rows  dist={df['label'].value_counts().to_dict()}")

    # ── Cross-validation (task req 7) ──
    t3 = time.perf_counter()
    _, oos_df, _ = run_cv(
        df,
        max_hold=max_hold,
        rf_params=rf_params,
        feature_cols=OPTUNA_FEATURE_COLS,
    )
    log.debug(f"CV done  [{time.perf_counter()-t3:.1f}s]")

    # ── OOS metrics ──
    strat_ret = strategy_log_returns(
        oos_df["y_pred"].to_numpy(),
        oos_df["log_return"].to_numpy(),
    )
    oos_sharpe = safe(sharpe_ratio(strat_ret))

    oos_report = classification_report(
        oos_df["y_true"].to_numpy(),
        oos_df["y_pred"].to_numpy(),
        labels=[0, 1],
        target_names=["flat", "long"],
        output_dict=True,
        zero_division=0,
    )
    macro_f1 = safe(oos_report["macro avg"]["f1-score"])
    accuracy  = safe(oos_report["accuracy"])
    f1_long   = safe(oos_report["long"]["f1-score"])
    f1_flat   = safe(oos_report["flat"]["f1-score"])

    # Store as Optuna user attrs so they appear in trials_dataframe()
    trial.set_user_attr("oos_sharpe",    oos_sharpe)
    trial.set_user_attr("oos_f1_macro",  macro_f1)
    trial.set_user_attr("oos_accuracy",  accuracy)
    trial.set_user_attr("oos_f1_long",   f1_long)
    trial.set_user_attr("oos_f1_flat",   f1_flat)
    trial.set_user_attr("max_hold",      max_hold)
    trial.set_user_attr("n_bars",        len(df))

    # ── Save model (task req 8: all models saved) ──
    t4 = time.perf_counter()
    Path("data/processed/models").mkdir(parents=True, exist_ok=True)
    model_path = f"data/processed/models/trial_{trial.number:06d}_bar{bar_size}_{SYMBOL}.pkl"
    train_final(
        df, rf_params,
        max_hold=max_hold,
        feature_cols=OPTUNA_FEATURE_COLS,
        out_path=model_path,
    )
    log.debug(f"  model saved → {model_path}  [{time.perf_counter()-t4:.1f}s]")

    wall = time.perf_counter() - t_trial
    log.info(
        f"Trial {trial.number:>6} | bar={bar_size}M  pt/sl={pt_sl_key}  span={span}"
        f"  hold={max_hold}  cusum={use_cusum}"
        f"  | sharpe={oos_sharpe:+.4f}  f1_macro={macro_f1:.4f}  acc={accuracy:.4f}"
        f"  f1_long={f1_long:.4f}"
        f"  | {wall:.0f}s"
    )

    return oos_sharpe


def optuna_main(
    n_trials: int = N_TRIALS,
    study_name: str = STUDY_NAME,
    optuna_n_jobs: int = 1,
) -> None:
    import os
    run_name = f"optuna_{study_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log = setup_logging(run_name)

    # Divide CPU cores evenly: each Optuna worker gets its share for RF
    cpu_count = os.cpu_count() or 1
    rf_jobs_per_worker = max(1, cpu_count // max(1, optuna_n_jobs))
    rf_params_runtime = {**FIXED_RF_PARAMS, "n_jobs": rf_jobs_per_worker}

    log.info("=" * 72)
    log.info(f"Optuna Grid Search  —  {SYMBOL}")
    log.info(f"Study: {study_name}  |  Trials: {n_trials:,}  |  Total space: {N_TRIALS:,}")
    log.info(f"Features ({len(OPTUNA_FEATURE_COLS)}): {OPTUNA_FEATURE_COLS}")
    log.info(f"CUSUM h={CUSUM_H}  |  Binary long-only labels")
    log.info(
        f"RF (fixed): n_est={FIXED_RF_PARAMS['n_estimators']}"
        f"  max_feat={FIXED_RF_PARAMS['max_features']}"
        f"  leaf={FIXED_RF_PARAMS['min_samples_leaf']}"
    )
    log.info(
        f"Parallel: optuna_n_jobs={optuna_n_jobs}"
        f"  rf_jobs_per_worker={rf_jobs_per_worker}"
        f"  cpu_count={cpu_count}"
    )
    log.info("=" * 72)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        sampler=optuna.samplers.GridSampler(SEARCH_SPACE),
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

    out_csv = "data/processed/optuna_trials.csv"
    study.trials_dataframe().to_csv(out_csv, index=False)
    log.info(f"Trial history → {out_csv}  ({len(study.trials)} rows)")
    log.info("=" * 72)


# ── Full single pipeline (no MLflow) ──────────────────────────────────────────

def main() -> None:
    run_name = f"pipeline_{SYMBOL}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log = setup_logging(run_name)

    t_start = time.perf_counter()
    log.info(f"Pipeline start — {SYMBOL}")

    bars = pd.read_parquet(f"data/processed/dollar_bars_{DEFAULT_BAR_SIZE}_{SYMBOL}.parquet")
    log.info(f"Bars loaded: {len(bars):,}")

    close_ts = bars["close"].copy()
    close_ts.index = pd.DatetimeIndex(bars["close_time"])
    vol = getDailyVol(close_ts, span=SPAN)

    pt, sl = PT_SL
    log.info(f"Labeling: PT={pt}  SL={sl}  MAX_HOLD={MAX_HOLD}")
    labels_df = label_bars(bars, vol, pt=pt, sl=sl, max_hold=MAX_HOLD)
    labels_df.to_parquet(f"data/processed/labels_{SYMBOL}.parquet")
    labels_df = pd.read_parquet(f"data/processed/labels_{SYMBOL}.parquet")

    df = build_features_inline(bars, labels_df)
    df["t1"] = df["t1_time"]

    label_dist = df["label"].value_counts().sort_index().to_dict()
    log.info(f"Bars: {len(df):,}  Labels: {label_dist}")

    fold_models, oos_df, fold_data = run_cv(df, rf_params=RF_PARAMS)

    all_fm = [fold_metrics(fd) for fd in fold_data]
    for i, fm in enumerate(all_fm):
        log.info(
            f"Fold {i}: acc={fm['test_accuracy']:.4f}"
            f"  macro_f1={fm['test_macro_f1']:.4f}"
            f"  sharpe={fm['sharpe']:.4f}"
        )

    oos_metrics = full_report(
        y_true=oos_df["y_true"].to_numpy(),
        y_pred=oos_df["y_pred"].to_numpy(),
        bar_ret=oos_df["log_return"].to_numpy(),
    )
    log.info(f"OOS Sharpe: {oos_metrics['sharpe']:.4f}")

    oos_df.to_csv(OOS_PATH)
    strat_ret = strategy_log_returns(
        oos_df["y_pred"].to_numpy(), oos_df["log_return"].to_numpy()
    )
    plot_equity_curve(strat_ret, title=f"OOS Equity Curve — {SYMBOL}")

    train_final(df, RF_PARAMS)
    log.info(f"Pipeline done in {(time.perf_counter() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    optuna_main()
