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
    atr_norm,
    bollinger_bands,
    buy_ratio,
    co_range,
    cyclical_time_features,
    dist_to_sma,
    frac_diff_ffd,
    hl_range,
    lagged_log_returns,
    log_returns,
    macd_features,
    ofi_ema,
    rate_of_change,
    realized_vol,
    rsi_feature,
    stochastic_rsi,
    trade_intensity,
    volume_zscore,
)
from src.model.metrics import (
    sharpe_ratio,
    strategy_log_returns,
    compute_bars_per_year
)
from src.model.train import SYMBOL, run_cv, train_final
from src.pre_process.trippler_barrier import getDailyVol, label_bars

# ─── Feature lists ───────────────────────────────────────────────────────────
# All features below are CAUSAL: at row t they depend only on bar data from
# rows ≤ t (backward rolling, .shift(k>0), or per-bar values known at close).
# No t1_time, no labels, no future bars are touched.

# List 1: original 6 indicator families (12 columns).
FEATURES_6_INDICATORS: list[str] = [
    "frac_diff",
    "bb_pct_b", "bb_width",
    "rsi",
    "macd", "macd_signal", "macd_hist",
    "adx", "adx_pdi", "adx_mdi",
    "stoch_rsi_k", "stoch_rsi_d",
]

# List 2: extended feature set. Adds microstructure, volatility, momentum,
# range, lagged returns, and cyclical time on top of the 6 indicator families.
FEATURES_EXTENDED: list[str] = FEATURES_6_INDICATORS + [
    # microstructure (per-bar; ofi/buy_volume/sell_volume/trades are bar-local)
    "ofi", "ofi_ema",
    "buy_ratio", "vol_zscore", "trade_intensity",
    # volatility & bar shape
    "atr_norm", "realized_vol_20",
    "hl_range", "co_range",
    # momentum & mean-reversion (backward windows only)
    "roc_5", "roc_20",
    "dist_to_sma20", "dist_to_sma50",
    # lagged returns
    "log_return_lag1", "log_return_lag2",
    # cyclical time-of-day / day-of-week
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]

# Selector — hardcoded so a run is reproducible without CLI flags.
ACTIVE_FEATURES: list[str] = FEATURES_EXTENDED

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

LABELS_CACHE_DIR = Path("data/processed/labels")


def labels_cache_path(
    symbol: str, bar_size: int, span: int,
    pt: float, sl: float, max_hold: int,
) -> Path:
    return LABELS_CACHE_DIR / (
        f"labels_{symbol}_bar{bar_size}_span{span}"
        f"_pt{pt}_sl{sl}_hold{max_hold}.parquet"
    )

# Phase-2-aligned defaults. max_depth is now a search axis (see SEARCH_SPACE).
FIXED_RF_PARAMS: dict = {
    "n_estimators": 500,
    "min_samples_leaf": 50,
    "class_weight": "balanced",
    "n_jobs":  -1,
    "random_state": 42,
}

# Phase-2-aligned grid (2×1×3×2×2×2 = 48). Sampled with replacement via RandomSampler.
SEARCH_SPACE: dict[str, list] = {
    "bar_size":      [3, 15],
    "pt_sl":         ["1.0_1.0"],
    "cusum_h":       [0.003, 0.005, 0.008],
    "span":          [10, 30],
    "max_hold_slot": [1, 2],
    "max_depth":     [3, 4],
}

# Random sampling over the 72-cell grid. 144 ≈ 2× cells, giving ~2 hits/cell on average.
N_TRIALS: int = 144

STUDY_NAME = "dollar_bar_grid72_random_v1"

TEST_RATIO: float = 0.20

# ─── Runtime knobs (hardcoded; no CLI) ───────────────────────────────────────
# Each `python -m src.main` invocation is a SINGLE worker: it runs trials
# sequentially within its own process. To run several workers in parallel,
# either use `python launch.py` (ProcessPool) or open N terminals and call
# `python -m src.main` in each. Workers share state via the sqlite study.
RUN_OPTUNA_N_JOBS: int = 1        # threads within a worker (keep at 1 — see launch.py)
RUN_RF_N_JOBS: int | None = None  # None → all CPUs (good for single-worker mode)
RUN_STORAGE: str | None = "sqlite:///optuna.db"


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


def cusum_filter(close: pd.Series, h: float) -> pd.Index:
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
    Compute the superset of features (6 indicators + extended set) on the FULL
    contiguous bar sequence. Must be called BEFORE any CUSUM filtering so
    rolling windows have no gaps. Returns a DataFrame with the same RangeIndex
    as bars.

    Leakage note: every column produced here is causal — backward-rolling
    windows, .shift(k>0), or bar-local values known at close. No future bars
    are referenced.
    """
    lr = log_returns(bars)
    log_price = pd.Series(np.log(bars["close"].to_numpy(dtype=np.float64)), index=bars.index)
    fd = frac_diff_ffd(log_price, d=1.0)  # placeholder: run_cv overrides per fold with ADF-selected d

    # 6 indicator families
    bb_pct_b, bb_width = bollinger_bands(bars)
    rsi_val = rsi_feature(bars)
    macd_line, macd_sig, macd_hist = macd_features(bars)
    adx_val, adx_pdi, adx_mdi = adx_features(bars)
    stoch_k, stoch_d = stochastic_rsi(bars)

    # Extended set (all causal)
    atr = atr_norm(bars, period=14)
    rv20 = realized_vol(bars, window=20)
    roc5 = rate_of_change(bars, window=5)
    roc20 = rate_of_change(bars, window=20)
    d_sma20 = dist_to_sma(bars, window=20)
    d_sma50 = dist_to_sma(bars, window=50)
    hl_r = hl_range(bars)
    co_r = co_range(bars)
    br = buy_ratio(bars)
    vz = volume_zscore(bars, window=50)
    ti = trade_intensity(bars, window=50)
    oema = ofi_ema(bars, span=10)
    lags = lagged_log_returns(bars, lags=(1, 2))
    cyc = cyclical_time_features(bars)

    out = pd.DataFrame({
        "open_time":    bars["open_time"].values,
        "close_time":   bars["close_time"].values,
        "close":        bars["close"].values,
        "log_return":   lr.values,
        # 6 indicator families
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
        # extended: microstructure
        "ofi":              bars["ofi"].values,
        "ofi_ema":          oema.values,
        "buy_ratio":        br.values,
        "vol_zscore":       vz.values,
        "trade_intensity":  ti.values,
        # extended: volatility / range
        "atr_norm":         atr.values,
        "realized_vol_20":  rv20.values,
        "hl_range":         hl_r.values,
        "co_range":         co_r.values,
        # extended: momentum / mean-reversion
        "roc_5":            roc5.values,
        "roc_20":           roc20.values,
        "dist_to_sma20":    d_sma20.values,
        "dist_to_sma50":    d_sma50.values,
        # extended: lagged returns
        "log_return_lag1":  lags["log_return_lag1"].values,
        "log_return_lag2":  lags["log_return_lag2"].values,
        # extended: cyclical time
        "hour_sin":         cyc["hour_sin"].values,
        "hour_cos":         cyc["hour_cos"].values,
        "dow_sin":          cyc["dow_sin"].values,
        "dow_cos":          cyc["dow_cos"].values,
    }, index=pd.RangeIndex(len(bars)))
    return out


# ── Optuna Grid Search ──────────────────────────────────────────────────────────

def objective(trial: optuna.Trial, log: logging.Logger, rf_params: dict) -> float:
    t_trial = time.perf_counter()
    # ── Suggested axes (6-axis grid sampled randomly) ──
    bar_size      = trial.suggest_categorical("bar_size",      SEARCH_SPACE["bar_size"])
    pt_sl_key     = trial.suggest_categorical("pt_sl",         SEARCH_SPACE["pt_sl"])
    cusum_h       = trial.suggest_categorical("cusum_h",       SEARCH_SPACE["cusum_h"])
    span          = trial.suggest_categorical("span",          SEARCH_SPACE["span"])
    max_hold_slot = trial.suggest_categorical("max_hold_slot", SEARCH_SPACE["max_hold_slot"])
    max_depth     = trial.suggest_categorical("max_depth",     SEARCH_SPACE["max_depth"])
    pt, sl = (float(x) for x in pt_sl_key.split("_"))
    tp_sl_key = pt_sl_key

    # PDF Action 1: weighting OFF (decay/overlap caused trivial-long in Phase-2 analysis).
    use_time_decay = False
    use_overlap    = False
    time_decay_c   = 1.0  # no-op; kept to satisfy run_cv() signature

    trial_rf_params = {**rf_params, "max_depth": max_depth}

    low, _, step = MAX_HOLD_RANGES[bar_size]
    max_hold = low + max_hold_slot * step

    log.debug(
        f"Trial {trial.number:>6} | bar={bar_size}M  pt/sl={tp_sl_key}"
        f"  span={span}  hold={max_hold}  cusum_h={cusum_h}"
        f"  max_depth={max_depth}"
        f"  td={use_time_decay}  overlap={use_overlap}"
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

    # ── 4. Label ALL bars (cached by bar_size/span/max_hold; cusum applied later) ──
    t_label = time.perf_counter()
    LABELS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = labels_cache_path(SYMBOL, bar_size, span, pt, sl, max_hold)
    labels_cache_hit = cache_path.exists()
    if labels_cache_hit:
        labels_full = pd.read_parquet(cache_path)
        cache_status = f"cache hit → {cache_path.name}"
    else:
        vol_aligned = vol_full.reindex(pd.DatetimeIndex(bars["close_time"])).ffill()
        labels_full = label_bars(bars, vol_aligned, pt=pt, sl=sl, max_hold=max_hold)
        labels_full.to_parquet(cache_path, index=False)
        cache_status = f"cached → {cache_path.name}"
    log.debug(
        f"  labels (all bars): {len(labels_full):,}"
        f"  dist={labels_full['label'].value_counts().to_dict()}"
        f"  ({cache_status})"
        f"  [{time.perf_counter()-t_label:.1f}s]"
    )

    # ── 5. CUSUM filter applied to labelable bars (positional, dedup-safe) ──
    labelable = len(labels_full)
    bars_labelable = bars.iloc[:labelable].reset_index(drop=True)
    feats_labelable = features_full.iloc[:labelable].reset_index(drop=True)
    event_times = cusum_filter(close_ts, h=cusum_h)
    cusum_mask = bars_labelable["close_time"].isin(event_times).values
    labels_df = labels_full[cusum_mask].reset_index(drop=True)
    feats_to_use = feats_labelable[cusum_mask].reset_index(drop=True)
    log.debug(f"CUSUM: {len(labels_df):,} bars selected  [{time.perf_counter()-t0:.1f}s]")

    # ── 6. Merge features + labels (positional alignment, both cusum-filtered) ──
    df = feats_to_use.copy()
    df["t1_time"] = labels_df["t1_time"].values
    df["t1"] = df["t1_time"]
    df["label"] = labels_df["label"].values
    df["ret"] = labels_df["return"].values

    # ── 7. Long-vs-short binary: drop flat(0); long(+1) → 1, short(-1) → 0 ──
    df = df[df["label"] != 0].copy()
    df["label"] = (df["label"] == 1).astype(int)
    log.debug(f"Binary long-vs-short: {len(df):,}  dist={df['label'].value_counts().to_dict()}")

    # ── 8. 5-fold Purged CV ──
    t_cv = time.perf_counter()
    _, oos_df, fold_data = run_cv(
        df,
        max_hold=max_hold,
        rf_params=trial_rf_params,
        feature_cols=ACTIVE_FEATURES,
        time_decay_c=time_decay_c,
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
    # Net PnL: cumulative compounded return over OOS log returns (no fees).
    oos_net_pnl = safe(float(np.exp(np.nansum(np.asarray(strat_ret, dtype=float))) - 1.0))

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

    features_set_name = (
        "6_indicators" if ACTIVE_FEATURES is FEATURES_6_INDICATORS
        else "extended" if ACTIVE_FEATURES is FEATURES_EXTENDED
        else "custom"
    )
    trial.set_user_attr("features_set", features_set_name)
    trial.set_user_attr("features_count", len(ACTIVE_FEATURES))
    trial.set_user_attr("features_list", ",".join(ACTIVE_FEATURES))
    trial.set_user_attr("oos_auc_pr", auc_pr)
    trial.set_user_attr("oos_sharpe", oos_sharpe)
    trial.set_user_attr("oos_net_pnl", oos_net_pnl)
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
    trial.set_user_attr("bar_size", bar_size)
    trial.set_user_attr("pt", pt)
    trial.set_user_attr("sl", sl)
    trial.set_user_attr("pt_sl", tp_sl_key)
    trial.set_user_attr("use_cusum", True)
    trial.set_user_attr("use_time_decay", use_time_decay)
    trial.set_user_attr("symbol", SYMBOL)
    trial.set_user_attr("labels_cache_hit", labels_cache_hit)

    # ── 10. Save model for this trial (all models saved — task req 8) ──
    t_save = time.perf_counter()
    Path("data/processed/models").mkdir(parents=True, exist_ok=True)
    model_path = f"data/processed/models/trial_{trial.number:06d}_bar{bar_size}_{SYMBOL}.pkl"
    train_final(
        df, trial_rf_params,
        max_hold=max_hold,
        feature_cols=ACTIVE_FEATURES,
        out_path=model_path,
        time_decay_c=time_decay_c,
        use_time_decay=use_time_decay,
        use_overlap=use_overlap,
    )
    log.debug(f"Model saved → {model_path}  [{time.perf_counter()-t_save:.1f}s]")

    wall = time.perf_counter() - t_trial
    log.info(
        f"Trial {trial.number:>6} | bar={bar_size}M  pt/sl={tp_sl_key}"
        f" span={span}  hold={max_hold}  cusum_h={cusum_h}"
        f"  td={use_time_decay}  td_c={time_decay_c}"
        f" | sharpe={oos_sharpe:+.4f}  net_pnl={oos_net_pnl:+.4f}"
        f"  f1_macro={macro_f1:.4f}  acc={accuracy:.4f}  f1_long={f1_long:.4f}"
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
    sampler_seed: int | None = 42,
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
    log.info(f"Optuna Random Search (72-cell grid)  —  {SYMBOL}")
    log.info(f"Study: {study_name}  |  Trials: {n_trials:,}")
    log.info(f"Features ({len(ACTIVE_FEATURES)}): {ACTIVE_FEATURES}")
    log.info(
        f"bar_size ∈ {SEARCH_SPACE['bar_size']}  |  "
        f"pt_sl ∈ {SEARCH_SPACE['pt_sl']}"
    )
    log.info(
        f"cusum_h ∈ {SEARCH_SPACE['cusum_h']}  |  "
        f"span ∈ {SEARCH_SPACE['span']}  |  "
        f"max_hold_slot ∈ {SEARCH_SPACE['max_hold_slot']}  |  "
        f"max_depth ∈ {SEARCH_SPACE['max_depth']}"
    )
    log.info(
        f"Fixed: use_time_decay=False  use_overlap=False  |  "
        f"Binary long-vs-short labels (flat dropped)"
    )
    log.info(
        f"RF: n_est={FIXED_RF_PARAMS['n_estimators']}  "
        f"min_samples_leaf={FIXED_RF_PARAMS['min_samples_leaf']}"
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
        sampler=optuna.samplers.RandomSampler(seed=sampler_seed),
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
    optuna_main(
        n_trials=N_TRIALS,
        study_name=STUDY_NAME,
        optuna_n_jobs=RUN_OPTUNA_N_JOBS,
        rf_n_jobs=RUN_RF_N_JOBS,
        storage=RUN_STORAGE,
    )
