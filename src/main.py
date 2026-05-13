import contextlib
import io
import json
import logging
import math
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
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
from src.backtest_utils import (
    FEE_BPS_PER_SIDE_DEFAULT,
    SLIPPAGE_BPS_PER_SIDE_DEFAULT,
    realistic_backtest,
)
from src.model.metrics import (
    sharpe_ratio,
    strategy_log_returns,
    compute_bars_per_year
)
from src.model.train import SYMBOL, run_cv
from src.pre_process.trippler_barrier import getDailyVol, label_bars

# ─── Feature lists ───────────────────────────────────────────────────────────
# All features below are CAUSAL: at row t they depend only on bar data from
# rows ≤ t (backward rolling, .shift(k>0), or per-bar values known at close).
# No t1_time, no labels, no future bars are touched.

# Original 6 indicator families (12 columns). Only feature set used by training.
FEATURES_6_INDICATORS: list[str] = [
    "frac_diff",
    "bb_pct_b", "bb_width",
    "rsi",
    "macd", "macd_signal", "macd_hist",
    "adx", "adx_pdi", "adx_mdi",
    "stoch_rsi_k", "stoch_rsi_d",
]

MAX_HOLD_RANGES: dict[int, tuple[int, int, int]] = {
    # (low, high, step). objective() uses: max_hold = low + max_hold_slot * step,
    # with slot ∈ [1..10] → 10 distinct values per bar_size.
    3:  (500, 5000, 500),   # 1000..5500
    5:  (300, 3000, 300),   #  600..3300
    15: (100, 1000, 100),   #  200..1100
    25: (60,   600,  60),   #  120..660
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

RF_PARAMS_FIXED: dict = {
    "n_estimators": 500,
    "max_depth": 4,
    "min_samples_leaf": 50,
    "class_weight": "balanced",
    "random_state": 42,
}

USE_TIME_DECAY_FIXED: bool = False
USE_OVERLAP_FIXED: bool = False

SEARCH_SPACE: dict[str, list] = {
    # v2 — bar 3/5 dropped (over-trading); pt_sl restricted to TP≤SL configs +
    # mid symmetric (extreme symmetrics dropped); cusum/span/hold trimmed to
    # mid-range. class_weight removed from search and fixed at "balanced".
    "bar_size":          [15, 25],                                                   # 2
    "pt_sl":             [
        "0.7_0.7", "0.9_0.9", "1.0_1.0", "1.2_1.2",  # mid symmetric
        "0.7_1.4", "0.8_1.2", "1.0_1.5",             # TP<SL asymmetric (v1 winners)
    ],                                                                               # 7
    "cusum_h":           [0.0035, 0.004, 0.005, 0.006, 0.007],                       # 5
    "span":              [30, 50, 80],                                               # 3
    "max_hold_slot":     [3, 5, 7],                                                  # 3
    "max_depth":         [3, 5],                                                     # 2 — probability calibration
    "min_samples_leaf":  [20, 50],                                                   # 2 — flexibility
}

# Theoretical space: 2 × 7 × 5 × 3 × 3 × 2 × 2 = 2,520 cells. TPE samples N_TRIALS.
N_TRIALS: int = 400

STUDY_NAME = "solusdt_v2_scoring_fixed"

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


def get_label_share(pt_sl_str: str) -> float:
    """Brownian-motion baseline P(TP hit first under a driftless random walk).

    With PT = pt·σ above and SL = sl·σ below the entry, optional-stopping on a
    Brownian motion gives  P(TP first) = sl / (pt + sl).  This is the prior
    probability of the +1 (long) label *before* the model sees any features,
    so it's the right baseline for both `precision_long` and `auc_pr`:
    a no-skill classifier sits at precision ≈ label_share, AUC-PR ≈ label_share.
    """
    tp, sl = (float(x) for x in pt_sl_str.split("_"))
    return sl / (tp + sl)


# ── Scoring v2 weights & hard-filter thresholds ────────────────────────────
# Rewards: AUC-PR edge above the Brownian baseline (signal), precision edge
# (selectivity), pf near break-even. Penalises: bad trade frequency, train/test
# gap, fold instability. Hard filter prunes degenerate trials.
W_SIGNAL:    float = 1.5   # signal_score = max(0, auc_edge * 5)         (edge vs label_share)
W_SELECT:    float = 1.0   # selectivity_score = max(0, precision_edge * 10)
W_PF:        float = 0.5   # pf_score = clip((pf - 0.7)/0.3, 0, 1.5)
W_FREQ:      float = 0.8   # freq_penalty: optimal 500–3000 trades/year
W_OVERFIT:   float = 1.5   # overfit_penalty = max(0, train_acc - test_acc) * 10
W_STABILITY: float = 0.5   # stability_penalty = max(0, fold_f1_std - 0.10) * 2

PRUNE_RECALL_LO:    float = 0.15    # trivial flat
PRUNE_RECALL_HI:    float = 0.90    # trivial long
PRUNE_EDGE_LO:      float = -0.02   # prune if prec_long < label_share + (-0.02)
PRUNE_AUC_EDGE_LO:  float = -0.01   # prune if auc_pr   < label_share + (-0.01)
PRUNE_MIN_TRADES:   int   = 100     # statistically meaningless


def _signal_score(auc_pr: float, baseline: float) -> float:
    """AUC-PR edge above the random-classifier baseline (= positive prevalence).
    0.08 edge → 0.40 score."""
    return max(0.0, (auc_pr - baseline) * 5.0)

def _selectivity_score(prec_long: float, baseline: float) -> float:
    """Precision edge above the Brownian baseline. 0.05 edge → 0.50 score."""
    return max(0.0, (prec_long - baseline) * 10.0)

def _pf_score(pf: float) -> float:
    if not math.isfinite(pf):
        return 0.0
    return min(1.5, max(0.0, (pf - 0.7) / 0.3))

def _freq_penalty(trades_per_year: float) -> float:
    tpy = safe(trades_per_year)
    if 500.0 <= tpy <= 3000.0:
        return 0.0
    if tpy < 500.0:
        return min(1.0, (500.0 - tpy) / 300.0)
    return min(1.5, (tpy - 3000.0) / 2000.0)

def _overfit_penalty(train_acc_mean: float, test_acc_mean: float) -> float:
    return max(0.0, train_acc_mean - test_acc_mean) * 10.0

def _stability_penalty(fold_f1_std: float) -> float:
    return max(0.0, fold_f1_std - 0.10) * 2.0


def composite_score(
    pt_sl: str,
    auc_pr: float,
    prec_long: float,
    recall_long: float,
    profit_factor: float,
    trades_per_year: float,
    n_trades: int,
    fold_train_accs: list[float],
    fold_test_accs: list[float],
    fold_f1_longs: list[float],
) -> tuple[float | None, dict, str | None]:
    """Scoring v2 — calibrated against the per-trial Brownian baseline.

    The +1 (long) label prevalence depends on the pt_sl ratio:
        label_share = sl / (pt + sl)
    A no-skill classifier hits precision ≈ label_share and AUC-PR ≈ label_share.
    We score the *edge* above that baseline, so e.g. precision = 0.66 on a
    pt_sl="0.7_1.4" trial (baseline = 0.667) is worth ~0, while the same
    precision on pt_sl="1.0_1.0" (baseline = 0.50) is worth +0.16 of edge.

    Returns ``(score, breakdown, prune_reason)``.
      - If ``prune_reason`` is not ``None``, the caller MUST raise
        ``optuna.TrialPruned(prune_reason)``; ``score`` will be ``None``.

    Components:
        + W_SIGNAL    * max(0, auc_edge * 5)        ← auc_pr  − label_share
        + W_SELECT    * max(0, precision_edge * 10) ← prec_long − label_share
        + W_PF        * clip((pf − 0.7)/0.3, 0, 1.5)
        - W_FREQ      * freq_penalty                ← optimal 500–3000 tpy
        - W_OVERFIT   * (train_acc − test_acc) * 10
        - W_STABILITY * max(0, std(fold f1) − 0.10) * 2

    Hard filter (edge-based; baseline = label_share):
        recall_long  ∉ [0.15, 0.90]                       trivial flat / trivial long
        prec_long    < label_share + PRUNE_EDGE_LO        below precision baseline
        auc_pr       < label_share + PRUNE_AUC_EDGE_LO    below AUC-PR baseline
        n_trades     < 100                                statistically meaningless
    """
    prec_long_s   = safe(prec_long)
    recall_long_s = safe(recall_long)
    auc_pr_s      = safe(auc_pr)
    pf_raw        = float(profit_factor) if profit_factor is not None else float("nan")
    tpy           = safe(trades_per_year)
    n_trades      = int(n_trades)
    label_share   = get_label_share(pt_sl)        # per-trial Brownian baseline

    # ── Hard filter (edge-based against per-trial baseline) ──
    if recall_long_s < PRUNE_RECALL_LO:
        return None, {}, f"recall_long={recall_long_s:.3f} < {PRUNE_RECALL_LO}"
    if recall_long_s > PRUNE_RECALL_HI:
        return None, {}, f"recall_long={recall_long_s:.3f} > {PRUNE_RECALL_HI}"
    prec_floor = label_share + PRUNE_EDGE_LO
    if prec_long_s < prec_floor:
        return None, {}, (
            f"prec_long={prec_long_s:.3f} < label_share+edge_lo={prec_floor:.3f} "
            f"(label_share={label_share:.3f})"
        )
    auc_floor = label_share + PRUNE_AUC_EDGE_LO
    if auc_pr_s < auc_floor:
        return None, {}, (
            f"auc_pr={auc_pr_s:.3f} < label_share+auc_edge_lo={auc_floor:.3f} "
            f"(label_share={label_share:.3f})"
        )
    if n_trades < PRUNE_MIN_TRADES:
        return None, {}, f"n_trades={n_trades} < {PRUNE_MIN_TRADES}"

    # ── Components (edge-based) ──
    precision_edge = prec_long_s - label_share
    auc_edge       = auc_pr_s    - label_share
    signal      = _signal_score(auc_pr_s, label_share)
    selectivity = _selectivity_score(prec_long_s, label_share)
    pf_s        = _pf_score(pf_raw)
    freq_pen    = _freq_penalty(tpy)
    overfit_pen = _overfit_penalty(
        float(np.mean(fold_train_accs)) if fold_train_accs else 0.0,
        float(np.mean(fold_test_accs))  if fold_test_accs  else 0.0,
    )
    fold_std    = float(np.std(fold_f1_longs, ddof=0)) if len(fold_f1_longs) > 1 else 0.0
    stab_pen    = _stability_penalty(fold_std)

    score = (
        + W_SIGNAL     * signal
        + W_SELECT     * selectivity
        + W_PF         * pf_s
        - W_FREQ       * freq_pen
        - W_OVERFIT    * overfit_pen
        - W_STABILITY  * stab_pen
    )
    breakdown = {
        "label_share":       label_share,
        "precision_edge":    precision_edge,
        "auc_edge":          auc_edge,
        "signal_score":      signal,
        "selectivity_score": selectivity,
        "pf_score":          pf_s,
        "freq_penalty":      freq_pen,
        "overfit_penalty":   overfit_pen,
        "stability_penalty": stab_pen,
        "fold_f1_std":       fold_std,
        "auc_pr":            auc_pr_s,
        "prec_long":         prec_long_s,
        "recall_long":       recall_long_s,
        "profit_factor":     pf_raw if math.isfinite(pf_raw) else 0.0,
        "trades_per_year":   tpy,
        "score_total":       float(score),
    }
    return float(score), breakdown, None


def setup_logging(run_name: str) -> tuple[logging.Logger, Path, Path]:
    """Create the run logger. Returns (logger, log_path, jsonl_path).

    Two sinks:
      - console (StreamHandler, INFO): clean dense per-trial line; banners.
      - file    (FileHandler,   DEBUG): everything above + per-fold breakdowns
        + captured `label_bars` stdout chatter.

    JSONL path (one trial-record per line) is returned alongside so the caller
    can hand it to TrialReporter.
    """
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    log_path   = logs_dir / f"{run_name}.log"
    jsonl_path = logs_dir / f"{run_name}.jsonl"

    log = logging.getLogger("optuna_search")
    log.setLevel(logging.DEBUG)
    if log.handlers:
        log.handlers.clear()
    log.propagate = False

    console_fmt = logging.Formatter("%(message)s")
    file_fmt    = logging.Formatter(
        "%(asctime)s [%(levelname)-5s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(console_fmt)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(file_fmt)
    log.addHandler(ch)
    log.addHandler(fh)
    return log, log_path, jsonl_path


def _fmt_duration(seconds: float) -> str:
    """Compact human duration: 47s, 3m12s, 11h22m."""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s"


class TrialReporter:
    """Per-trial logging + JSONL emission + running best/ETA tracking.

    `report(...)` is called once per completed trial. It:
      • updates best-so-far and trial timing
      • prints a single dense INFO line (console + file)
      • appends a structured row to the JSONL file (machine-readable)
    """

    def __init__(self, log: logging.Logger, jsonl_path: Path, n_trials: int):
        self.log = log
        self.n_trials = n_trials
        self.start = time.perf_counter()
        self.best_score = float("-inf")
        self.best_trial = -1
        self.completed = 0
        self.jsonl_path = jsonl_path
        self._fh = open(jsonl_path, "a", encoding="utf-8")

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass

    def report(
        self,
        *,
        trial_number: int,
        params: dict,
        score: float,
        breakdown: dict,
        oos: dict,
        bt: dict,
        wall_s: float,
    ) -> None:
        self.completed += 1
        if score > self.best_score:
            self.best_score = score
            self.best_trial = trial_number

        elapsed = time.perf_counter() - self.start
        mean_t  = elapsed / max(1, self.completed)
        eta     = (self.n_trials - self.completed) * mean_t

        self.log.info(
            f"[Trial {trial_number:>4}/{self.n_trials}] "
            f"bar={params['bar_size']:>2}M "
            f"pt/sl={params['pt_sl']:>7} "
            f"span={params['span']:>2} "
            f"hold={params['max_hold']:>4} "
            f"cusum={params['cusum_h']:.4f} "
            f"│ score={score:+.4f} "
            f"(sig={breakdown['signal_score']:.2f} "
            f"sel={breakdown['selectivity_score']:.2f} "
            f"pf={breakdown['pf_score']:.2f} "
            f"freq={breakdown['freq_penalty']:.2f} "
            f"over={breakdown['overfit_penalty']:.2f} "
            f"stab={breakdown['stability_penalty']:.2f}) "
            f"│ aucpr={oos['auc_pr']:.2f} "
            f"p={oos['prec_long']:.2f} "
            f"r={oos['recall_long']:.2f} "
            f"│ bt n={bt['n_trades']:>5} "
            f"pnl={safe(bt['net_pnl_after_fees']):+.3f} "
            f"pf={safe(bt['profit_factor_net']):.2f} "
            f"tpy={safe(bt['trades_per_year']):.0f} "
            f"│ {_fmt_duration(wall_s):>6} "
            f"ETA {_fmt_duration(eta):>6} "
            f"★best={self.best_score:+.3f}(#{self.best_trial})"
        )

        rec = {
            "trial":     int(trial_number),
            "wall_s":    float(wall_s),
            "elapsed_s": float(elapsed),
            "params":    params,
            "score":     float(score),
            "breakdown": {k: float(v) for k, v in breakdown.items()},
            "oos":       {k: float(v) for k, v in oos.items()},
            "bt_real": {
                "n_trades":         int(bt["n_trades"]),
                "net_pnl":          safe(bt["net_pnl_after_fees"]),
                "gross_pnl":        safe(bt["gross_pnl_net"]),
                "win_rate":         safe(bt["win_rate_net"]),
                "profit_factor":    safe(bt["profit_factor_net"]),
                "mdd_log":          safe(bt["max_drawdown_log_net"]),
                "sharpe_trade":     safe(bt["sharpe_trade_annualized"]),
                "trades_per_year":  safe(bt["trades_per_year"]),
            },
            "best_so_far": {"score": self.best_score, "trial": self.best_trial},
        }
        self._fh.write(json.dumps(rec) + "\n")
        self._fh.flush()


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
    Compute the 6 indicator families (frac_diff + BB + RSI + MACD + ADX +
    Stoch RSI) on the FULL contiguous bar sequence. Must be called BEFORE any
    CUSUM filtering so rolling windows have no gaps.

    Leakage note: every column is causal — backward-rolling windows,
    .shift(k>0), or bar-local values known at close. No future bars referenced.
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


# ── Optuna TPE Search ──────────────────────────────────────────────────────────

def objective(
    trial: optuna.Trial,
    log: logging.Logger,
    rf_params: dict,
    reporter: "TrialReporter | None" = None,
) -> float:
    t_trial = time.perf_counter()
    bar_size         = trial.suggest_categorical("bar_size",         SEARCH_SPACE["bar_size"])
    pt_sl_key        = trial.suggest_categorical("pt_sl",            SEARCH_SPACE["pt_sl"])
    cusum_h          = trial.suggest_categorical("cusum_h",          SEARCH_SPACE["cusum_h"])
    span             = trial.suggest_categorical("span",             SEARCH_SPACE["span"])
    max_hold_slot    = trial.suggest_categorical("max_hold_slot",    SEARCH_SPACE["max_hold_slot"])
    max_depth        = trial.suggest_categorical("max_depth",        SEARCH_SPACE["max_depth"])
    min_samples_leaf = trial.suggest_categorical("min_samples_leaf", SEARCH_SPACE["min_samples_leaf"])
    pt, sl = (float(x) for x in pt_sl_key.split("_"))
    tp_sl_key = pt_sl_key

    active_features = FEATURES_6_INDICATORS
    use_time_decay  = USE_TIME_DECAY_FIXED
    use_overlap     = USE_OVERLAP_FIXED
    time_decay_c    = 1.0  # no-op; kept to satisfy run_cv() signature

    low, _, step = MAX_HOLD_RANGES[bar_size]
    max_hold = low + max_hold_slot * step

    # Per-trial RF params: start from the baseline (n_jobs, class_weight,
    # n_estimators, random_state) and override the two searched RF dims.
    rf_params_trial = {
        **rf_params,
        "max_depth":        max_depth,
        "min_samples_leaf": min_samples_leaf,
    }

    log.debug(
        f"Trial {trial.number:>6} | bar={bar_size}M  pt/sl={tp_sl_key}"
        f"  span={span}  hold={max_hold}  cusum_h={cusum_h}"
        f"  depth={max_depth}  leaf={min_samples_leaf}"
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
        # Capture label_bars' progress prints into DEBUG (file only); keep console clean.
        _buf = io.StringIO()
        with contextlib.redirect_stdout(_buf):
            labels_full = label_bars(bars, vol_aligned, pt=pt, sl=sl, max_hold=max_hold)
        for _line in _buf.getvalue().splitlines():
            if _line.strip():
                log.debug(f"  label_bars: {_line}")
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
        rf_params=rf_params_trial,
        feature_cols=active_features,
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

    # ── 9b. Realistic OOS backtest (non-overlap PT/SL exits + fees) ──
    # `df` carries t1_time (gappy index after label-filter); rejoin onto oos_df.
    oos_df_bt = oos_df.join(df[["t1_time"]], how="left")
    bt_real = realistic_backtest(
        y_pred=oos_df_bt["y_pred"].to_numpy(),
        triple_barrier_log_ret=oos_df_bt["ret"].to_numpy(),
        close_times=oos_df_bt["close_time"].to_numpy(),
        t1_times=oos_df_bt["t1_time"].to_numpy(),
    )

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
    trial.set_user_attr("pt_sl", tp_sl_key)
    trial.set_user_attr("labels_cache_hit", labels_cache_hit)

    # ── Realistic non-overlap backtest user_attrs (fees + PT/SL exits) ──
    trial.set_user_attr("oos_real_n_trades",        bt_real["n_trades"])
    trial.set_user_attr("oos_real_net_pnl",         safe(bt_real["net_pnl_after_fees"]))
    trial.set_user_attr("oos_real_gross_pnl",       safe(bt_real["gross_pnl_net"]))
    trial.set_user_attr("oos_real_win_rate",        safe(bt_real["win_rate_net"]))
    trial.set_user_attr("oos_real_profit_factor",   safe(bt_real["profit_factor_net"]))
    trial.set_user_attr("oos_real_mdd_log",         safe(bt_real["max_drawdown_log_net"]))
    trial.set_user_attr("oos_real_sharpe_trade",    safe(bt_real["sharpe_trade_annualized"]))
    trial.set_user_attr("oos_real_trades_per_year", safe(bt_real["trades_per_year"]))
    trial.set_user_attr("oos_real_round_trip_bps",  bt_real["round_trip_cost_bps"])
    trial.set_user_attr("fee_bps_per_side",         FEE_BPS_PER_SIDE_DEFAULT)
    trial.set_user_attr("slippage_bps_per_side",    SLIPPAGE_BPS_PER_SIDE_DEFAULT)

    # ── 10. Composite score v2 (+ hard filter / TrialPruned) ──
    fold_train_accs = [fr["train_acc"] for fr in fold_results]
    fold_test_accs  = [fr["accuracy"]  for fr in fold_results]
    fold_f1_longs   = [fr["f1_long"]   for fr in fold_results]

    score, score_breakdown, prune_reason = composite_score(
        pt_sl           = tp_sl_key,
        auc_pr          = auc_pr,
        prec_long       = prec_long,
        recall_long     = recall_long,
        profit_factor   = bt_real["profit_factor_net"],
        trades_per_year = bt_real["trades_per_year"],
        n_trades        = int(bt_real["n_trades"]),
        fold_train_accs = fold_train_accs,
        fold_test_accs  = fold_test_accs,
        fold_f1_longs   = fold_f1_longs,
    )

    if prune_reason is not None:
        # Record reason on the trial so it's visible in the DB / CSV, then prune.
        trial.set_user_attr("prune_reason",   prune_reason)
        trial.set_user_attr("oos_prec_long",  prec_long)
        trial.set_user_attr("oos_recall_long",recall_long)
        trial.set_user_attr("oos_auc_pr",     auc_pr)
        trial.set_user_attr("oos_real_n_trades", int(bt_real["n_trades"]))
        wall_prune = time.perf_counter() - t_trial
        log.info(
            f"[Trial {trial.number:>4}] PRUNED  ({prune_reason})  "
            f"bar={bar_size}M  pt/sl={tp_sl_key}  span={span}  "
            f"hold={max_hold}  cusum={cusum_h}  | {wall_prune:.0f}s"
        )
        raise optuna.TrialPruned(prune_reason)

    for k, v in score_breakdown.items():
        trial.set_user_attr(k, v)
    trial.set_user_attr("composite_score", score)

    wall = time.perf_counter() - t_trial
    if reporter is not None:
        reporter.report(
            trial_number = trial.number,
            params = {
                "bar_size":         bar_size,
                "pt_sl":            tp_sl_key,
                "span":             span,
                "cusum_h":          cusum_h,
                "max_hold":         max_hold,
                "max_depth":        max_depth,
                "min_samples_leaf": min_samples_leaf,
            },
            score     = score,
            breakdown = score_breakdown,
            oos = {
                "f1_long":     f1_long,
                "prec_long":   prec_long,
                "recall_long": recall_long,
                "sharpe":      oos_sharpe,
                "net_pnl":     oos_net_pnl,
                "auc_pr":      auc_pr,
            },
            bt = bt_real,
            wall_s = wall,
        )
    else:
        log.info(
            f"[Trial {trial.number:>4}] score={score:+.4f}  bar={bar_size}M"
            f"  pt/sl={tp_sl_key}  span={span}  hold={max_hold}  cusum={cusum_h}"
            f"  | {wall:.0f}s"
        )
    return score


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
    log, log_path, jsonl_path = setup_logging(run_name)

    # Silence Optuna's experimental-flag warnings (we know multivariate/group are experimental).
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    cpu_count = os.cpu_count() or 1
    if rf_n_jobs is None:
        rf_n_jobs = max(1, cpu_count // max(1, optuna_n_jobs))
    rf_params_runtime = {**RF_PARAMS_FIXED, "n_jobs": rf_n_jobs}

    bar = "═" * 78
    sub = "─" * 78
    space_size = 1
    for v in SEARCH_SPACE.values():
        space_size *= len(v)
    log.info(bar)
    log.info(f"  OPTUNA TPE SEARCH v2 — {SYMBOL}   ({datetime.now():%Y-%m-%d %H:%M:%S})")
    log.info(bar)
    log.info(f"  study:        {study_name}")
    log.info(f"  trials:       {n_trials:,}   (space = {space_size:,} cells)")
    log.info(f"  sampler:      TPE  seed={sampler_seed}  n_startup_trials=50  multivariate=True  constant_liar=True")
    log.info(f"  storage:      {storage or 'in-memory'}   (load_if_exists=True — workers share)")
    log.info(f"  parallel:     optuna_n_jobs={optuna_n_jobs}  rf_n_jobs={rf_n_jobs}  cpu={cpu_count}")
    log.info(f"  log file:     {log_path}")
    log.info(f"  jsonl file:   {jsonl_path}")
    log.info(sub)
    log.info(f"  search dims:")
    log.info(f"    bar_size         ∈ {SEARCH_SPACE['bar_size']}")
    log.info(f"    pt_sl            ∈ {SEARCH_SPACE['pt_sl']}")
    log.info(f"    cusum_h          ∈ {SEARCH_SPACE['cusum_h']}")
    log.info(f"    span             ∈ {SEARCH_SPACE['span']}")
    log.info(f"    max_hold_slot    ∈ {SEARCH_SPACE['max_hold_slot']}")
    log.info(f"    max_depth        ∈ {SEARCH_SPACE['max_depth']}")
    log.info(f"    min_samples_leaf ∈ {SEARCH_SPACE['min_samples_leaf']}")
    log.info(sub)
    log.info(
        f"  score = +{W_SIGNAL}·signal +{W_SELECT}·selectivity +{W_PF}·pf "
        f"-{W_FREQ}·freq -{W_OVERFIT}·overfit -{W_STABILITY}·stability"
    )
    log.info(
        f"  baseline:     label_share = sl/(tp+sl) — signal/selectivity scored "
        f"as EDGE above this per-trial Brownian prior"
    )
    log.info(
        f"  hard filter (TrialPruned, baseline = sl/(pt+sl)):"
    )
    log.info(
        f"    recall_long ∉ [{PRUNE_RECALL_LO}, {PRUNE_RECALL_HI}]  │  "
        f"prec_long < baseline + ({PRUNE_EDGE_LO})  │  "
        f"auc_pr < baseline + ({PRUNE_AUC_EDGE_LO})  │  "
        f"n_trades < {PRUNE_MIN_TRADES}"
    )
    log.info(
        f"  features:     6_indicators (12 cols, frac_diff + BB + RSI + MACD + ADX + Stoch RSI)"
    )
    log.info(
        f"  RF base:      n_est={RF_PARAMS_FIXED['n_estimators']}  "
        f"class_weight={RF_PARAMS_FIXED['class_weight']}  "
        f"random_state={RF_PARAMS_FIXED['random_state']}  "
        f"(max_depth / min_samples_leaf are searched)"
    )
    log.info(bar)

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        sampler=TPESampler(
            seed=sampler_seed,
            n_startup_trials=50,
            multivariate=True,
            constant_liar=True,
        ),
        storage=storage,
        load_if_exists=False,
    )

    # ── Warm-start: seed the TPE with a known-good point from the v1 run.
    # Only trial 18 from v1 fits the new search space (the others used
    # bar_size 3/5 or cusum/span values now pruned out). Skip silently if
    # the values somehow drift out of the current space.
    # Warm-start with v1 winners that still fit the (much smaller) v2 space.
    # Each dict must contain every searched key. The filter below silently
    # drops points whose values fall outside the new space. Trial 6's params
    # were not logged, so it can't be enqueued.
    WARM_START_TRIALS = [
        # Trial 18 (v1):  bar=15 pt_sl=0.7_1.4 cusum=0.006 span=80 max_hold=300 (slot 2)
        {"bar_size": 15, "pt_sl": "0.7_1.4", "cusum_h": 0.006,
         "span": 80, "max_hold_slot": 2,
         "max_depth": 3, "min_samples_leaf": 50},
        # Trial 30 (v1):  bar=25 pt_sl=0.8_1.2 cusum=0.0015 span=50 max_hold=540 (slot 8)
        {"bar_size": 25, "pt_sl": "0.8_1.2", "cusum_h": 0.0015,
         "span": 50, "max_hold_slot": 8,
         "max_depth": 3, "min_samples_leaf": 50},
        # Trial 6 (v1):  params unknown — omitted intentionally.
    ]
    enqueued, skipped = 0, []
    for w in WARM_START_TRIALS:
        bad = [k for k in w if w[k] not in SEARCH_SPACE[k]]
        if bad:
            skipped.append((w, bad))
        else:
            study.enqueue_trial(w)
            enqueued += 1
    log.info(f"  warm-start:   enqueued {enqueued}/{len(WARM_START_TRIALS)} v1-trial points "
             f"({len(skipped)} skipped — keys out of v2 space)")
    for w, bad in skipped:
        log.debug(f"    skipped warm-start (out-of-space keys {bad}): {w}")
    log.info(bar)

    reporter = TrialReporter(log, jsonl_path, n_trials)
    flow_start = time.perf_counter()
    try:
        study.optimize(
            lambda trial: objective(trial, log, rf_params_runtime, reporter),
            n_trials=n_trials,
            n_jobs=optuna_n_jobs,
            show_progress_bar=False,
        )
    finally:
        reporter.close()

    total = time.perf_counter() - flow_start

    # ── End-of-run summary ──
    out_csv = "optuna_trials.csv"
    study.trials_dataframe().to_csv(out_csv, index=False)

    completed_trials = [t for t in study.trials if t.value is not None]
    top = sorted(completed_trials, key=lambda t: t.value, reverse=True)[:5]

    log.info(bar)
    log.info(f"  SEARCH COMPLETE  —  {len(completed_trials)}/{n_trials} trials in {_fmt_duration(total)}")
    log.info(bar)
    if top:
        best = top[0]
        ua   = best.user_attrs
        log.info(f"  ★ best: trial #{best.number}   composite_score = {best.value:+.4f}")
        log.info(f"    params:   {best.params}")
        log.info(
            f"    OOS:      f1_long={ua.get('oos_f1_long', 0):.4f}  "
            f"prec_long={ua.get('oos_prec_long', 0):.4f}  "
            f"recall_long={ua.get('oos_recall_long', 0):.4f}  "
            f"sharpe={ua.get('oos_sharpe', 0):+.4f}"
        )
        log.info(
            f"    bt_real:  n_trades={ua.get('oos_real_n_trades', 0)}  "
            f"net_pnl={ua.get('oos_real_net_pnl', 0):+.4f}  "
            f"win_rate={ua.get('oos_real_win_rate', 0):.3f}  "
            f"pf={ua.get('oos_real_profit_factor', 0):.3f}  "
            f"sharpe_trade={ua.get('oos_real_sharpe_trade', 0):+.4f}"
        )
        log.info(sub)
        log.info(f"  top 5 by composite_score:")
        for t in top:
            log.info(
                f"    #{t.number:>5}  score={t.value:+.4f}  "
                f"bar={t.params.get('bar_size'):>2}M  pt/sl={t.params.get('pt_sl'):>7}  "
                f"span={t.params.get('span'):>2}  hold_slot={t.params.get('max_hold_slot'):>2}  "
                f"cusum={t.params.get('cusum_h'):.4f}"
            )
    log.info(sub)
    log.info(f"  outputs:")
    log.info(f"    {log_path}        (full run log, DEBUG level)")
    log.info(f"    {jsonl_path}      (per-trial JSONL)")
    log.info(f"    {out_csv}         (Optuna trials dataframe)")
    log.info(bar)


if __name__ == "__main__":
    optuna_main(
        n_trials=N_TRIALS,
        study_name=STUDY_NAME,
        optuna_n_jobs=RUN_OPTUNA_N_JOBS,
        rf_n_jobs=RUN_RF_N_JOBS,
        storage=RUN_STORAGE,
    )
