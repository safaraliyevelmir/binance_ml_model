"""Meta-labeling pipeline — AFML Ch 3.6.

Loads a frozen primary model bundle (from src/main.py train_batch), applies
calibration to its OOS probas, computes a 35-feature meta pool, trains a
meta RandomForest with walk-forward validation, and runs Optuna hyperparam
search over (calibration method, feature subset, RF params, meta threshold,
primary threshold).

Usage (hardcoded at bottom):
    python -m src.meta
"""
from __future__ import annotations

import io
import json
import logging
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV  # noqa: F401  (future: sigmoid_cv5)
from sklearn.metrics import precision_recall_curve, auc
from sklearn.model_selection import KFold

from src.main import (
    FEATURES_6_INDICATORS,
    LABELS_CACHE_DIR,
    SYMBOL,
    compute_bar_features,
    compute_lagged_bar_features,
    cusum_filter,
    isotonic_cv_calibrate,
    labels_cache_path,
    setup_logging,
)


# ── 35-Feature Meta Pool ───────────────────────────────────────────────────────
# All features are V2-lagged (computed via .shift(1) where applicable).
# See compute_meta_features() for the build flow.

META_VOLATILITY = [
    "atr_ratio", "atr_normalized",
    "rstd_24", "rstd_96", "rstd_192", "rstd_12_192",
    "vol_of_vol_48", "gk_vol_24",
]

META_MOMENTUM = [
    "ret_6", "ret_24", "ret_96", "mom_zscore_24",
    "log_return", "bar_direction",
]

META_MICROSTRUCTURE = [
    "dvol_z_48", "vol_z_48",
    "body_ratio", "vwap_dev_24", "vwap_dev_96", "vwap_dev_96_z",
]

META_DISTRIBUTION = [
    "ret_skew_48", "ret_kurt_96",
    "sign_entropy_48", "hurst_100",
]

META_STRUCTURAL = [
    "var_ratio_8_96", "var_ratio_4_48", "cusum_breaks_100",
]

META_TIME = [
    "hour_sin", "hour_cos", "dow_cos", "mins_to_funding_norm",
]

META_PRIMARY_AWARE = [
    "proba_long_calib", "proba_margin", "proba_entropy", "proba_extremity",
]

META_HISTORY = ["hit_rate_50_causal"]  # 1

META_FEATURE_POOL_35: list[str] = (
    META_VOLATILITY + META_MOMENTUM + META_MICROSTRUCTURE
    + META_DISTRIBUTION + META_STRUCTURAL + META_TIME
    + META_PRIMARY_AWARE + META_HISTORY
)
# Actually 36 (8+6+6+4+3+4+4+1) — name kept as "_35" for historical reference.
# To run Optuna with EXACTLY 35, drop one feature here (e.g. proba_extremity is
# monotonic in proba_margin).
assert len(META_FEATURE_POOL_35) >= 30, f"Pool too small: {len(META_FEATURE_POOL_35)}"

# Time/causal features that are safe even without shift(1) — they don't peek
# at current-bar close. Used only inside compute_meta_features().
NO_LAG_FEATURES = {
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "mins_to_funding_norm", "hit_rate_50_causal",
}


# ── Cost model — execution scenarios ──────────────────────────────────────────

FEE_BPS_PER_SIDE_TAKER = 4.0
FEE_BPS_PER_SIDE_MAKER = 2.0
SLIPPAGE_BPS_PER_SIDE  = 1.0

RT_COST_TAKER = 2 * (FEE_BPS_PER_SIDE_TAKER + SLIPPAGE_BPS_PER_SIDE) / 1e4   # 10 bps
RT_COST_MAKER = 2 * (FEE_BPS_PER_SIDE_MAKER + SLIPPAGE_BPS_PER_SIDE) / 1e4   # 6 bps


# ── Causal hit-rate (López AFML t1_time-aware) ────────────────────────────────

def causal_hit_rate(df: pd.DataFrame, close_col: str = "close_time",
                    t1_col: str = "t1_time", y_col: str = "meta_y",
                    window: int = 50, min_periods: int = 20) -> np.ndarray:
    """Rolling hit rate using ONLY trades whose barriers resolved before current
    decision time. df must be sorted by close_time."""
    n = len(df)
    ct = pd.to_datetime(df[close_col]).values
    t1 = pd.to_datetime(df[t1_col]).values
    y  = df[y_col].astype(float).values
    out = np.full(n, np.nan)
    for i in range(n):
        resolved = np.where(t1[:i] < ct[i])[0]
        if len(resolved) >= min_periods:
            out[i] = y[resolved[-window:]].mean()
    return out


# ── Calibration variants ──────────────────────────────────────────────────────

def calibrate_cv_oos(cv_oos: pd.DataFrame, method: str = "isotonic_posthoc"
                     ) -> tuple[IsotonicRegression, pd.DataFrame]:
    """Calibrate primary's cv_oos probas. Returns (calibrator_for_test, cv_oos_calibrated).

    Methods:
      - "isotonic_posthoc": fit isotonic on full cv_oos, apply to itself (slight self-fit bias)
      - "isotonic_cv5":     5-fold chrono CV inside cv_oos (no self-fit bias)
      - "none":             identity — return raw proba
    """
    cv_oos = cv_oos.sort_values("close_time").reset_index(drop=True).copy()
    if "proba_1" not in cv_oos.columns:
        raise ValueError("cv_oos must have 'proba_1'")
    cv_oos["proba_1_raw"] = cv_oos["proba_1"].values

    if method == "none":
        cv_oos["proba_long_calib"] = cv_oos["proba_1"].values
        # Identity "calibrator" — predict_proba returns raw
        class _Identity:
            def transform(self, x): return np.asarray(x, dtype=float)
        return _Identity(), cv_oos

    if method == "isotonic_posthoc":
        return isotonic_cv_calibrate(cv_oos, n_splits=2)  # tiny fold for posthoc-ish behavior

    if method == "isotonic_cv5":
        return isotonic_cv_calibrate(cv_oos, n_splits=5)

    raise ValueError(f"Unknown calibration method: {method}")


# ── Meta feature construction ────────────────────────────────────────────────

def _add_regime_features(features_full: pd.DataFrame, bars: pd.DataFrame,
                          compute_only: list[str] | None = None) -> pd.DataFrame:
    """Add regime features. If `compute_only` is given, ONLY compute those.

    Skipping expensive features (hurst_100, cusum_breaks_100, sign_entropy_48)
    gives ~3-5× speedup when meta only needs a subset.

    Assumes features_full is already V2-lagged at the 12-indicator level.
    All NEW features added here are also computed in V2-lagged form.
    """
    def _want(name):
        return compute_only is None or name in compute_only
    ff = features_full.copy()
    _log_p   = np.log(ff["close"])  # close is NOT lagged (identifier)
    _log_r1  = _log_p.diff()
    _log_r8  = _log_p.diff(8)
    _log_r4  = _log_p.diff(4)

    _h = bars["high"].astype(float).reset_index(drop=True)
    _l = bars["low"].astype(float).reset_index(drop=True)
    _c = bars["close"].astype(float).reset_index(drop=True)
    _o = bars["open"].astype(float).reset_index(drop=True)
    _v = bars["volume"].astype(float).reset_index(drop=True)
    _qv = (bars["quote_volume"].astype(float).reset_index(drop=True)
           if "quote_volume" in bars.columns else (_v * _c))

    # ── Volatility ──
    _tr = pd.concat([(_h - _l), (_h - _c.shift(1)).abs(), (_l - _c.shift(1)).abs()], axis=1).max(axis=1)
    _atr_24 = _tr.rolling(24).mean()
    _atr_96 = _tr.rolling(96).mean()
    ff["atr_ratio"]      = (_atr_24 / _atr_96.replace(0, np.nan)).values
    ff["atr_normalized"] = (_atr_24 / _c.replace(0, np.nan)).values
    ff["rstd_24"]   = _log_r1.rolling(24).std().values
    ff["rstd_96"]   = _log_r1.rolling(96).std().values
    ff["rstd_192"]  = _log_r1.rolling(192).std().values
    ff["rstd_12_192"] = (_log_r1.rolling(12).std() / _log_r1.rolling(192).std().replace(0, np.nan)).values
    ff["vol_of_vol_48"] = _log_r1.rolling(12).std().rolling(48).std().values
    # Garman-Klass
    _hl = np.log(_h / _l) ** 2
    _co = np.log(_c / _o) ** 2
    _gk = 0.5 * _hl - (2.0 * np.log(2.0) - 1.0) * _co
    ff["gk_vol_24"] = np.sqrt(_gk.rolling(24).mean().clip(lower=0)).values

    # ── Momentum ──
    ff["ret_6"]  = (_log_p - _log_p.shift(6)).values
    ff["ret_24"] = (_log_p - _log_p.shift(24)).values
    ff["ret_96"] = (_log_p - _log_p.shift(96)).values
    _r24_mean = pd.Series(ff["ret_24"]).rolling(96, min_periods=48).mean()
    _r24_std  = pd.Series(ff["ret_24"]).rolling(96, min_periods=48).std()
    ff["mom_zscore_24"] = ((pd.Series(ff["ret_24"]) - _r24_mean) / _r24_std.replace(0, np.nan)).values
    # log_return is already in features_full from compute_bar_features
    ff["bar_direction"] = np.sign(_c - _o).values

    # ── Microstructure ──
    ff["dvol_z_48"] = ((_qv - _qv.rolling(48).mean()) / _qv.rolling(48).std().replace(0, np.nan)).values
    ff["vol_z_48"]  = ((_v  - _v.rolling(48).mean())  / _v.rolling(48).std().replace(0, np.nan)).values
    _rng = (_h - _l).replace(0, np.nan)
    ff["body_ratio"] = ((_c - _o).abs() / _rng).clip(0, 1).values
    _tp = (_h + _l + _c) / 3.0
    _vwap_24 = (_tp * _v).rolling(24).sum() / _v.rolling(24).sum().replace(0, np.nan)
    _vwap_96 = (_tp * _v).rolling(96).sum() / _v.rolling(96).sum().replace(0, np.nan)
    ff["vwap_dev_24"] = ((_c - _vwap_24) / _vwap_24).values
    ff["vwap_dev_96"] = ((_c - _vwap_96) / _vwap_96).values
    _vd96 = pd.Series(ff["vwap_dev_96"])
    _mean_192 = _vd96.rolling(192, min_periods=96).mean()
    _std_192  = _vd96.rolling(192, min_periods=96).std()
    ff["vwap_dev_96_z"] = ((_vd96 - _mean_192) / _std_192.replace(0, np.nan)).values

    # ── Distribution shape ──
    ff["ret_skew_48"] = _log_r1.rolling(48).skew().values
    ff["ret_kurt_96"] = _log_r1.rolling(96).kurt().values

    if _want("sign_entropy_48"):
        def _sign_entropy(r: pd.Series, w: int) -> np.ndarray:
            signs = np.sign(r).fillna(0)
            def _e(arr):
                if len(arr) == 0: return np.nan
                _, c = np.unique(arr, return_counts=True)
                p = c / c.sum()
                return -np.sum(p * np.log(p + 1e-12))
            return signs.rolling(w).apply(_e, raw=True).values
        ff["sign_entropy_48"] = _sign_entropy(_log_r1, 48)

    if _want("hurst_100"):
        def _rolling_hurst(x: pd.Series, w: int = 100, max_lag: int = 20):
            def _h_calc(arr):
                if len(arr) < max_lag + 5 or np.any(~np.isfinite(arr)):
                    return np.nan
                lags = np.arange(2, min(max_lag, len(arr) // 2))
                try:
                    tau = [np.sqrt(np.std(np.subtract(arr[lag:], arr[:-lag]))) for lag in lags]
                    poly = np.polyfit(np.log(lags), np.log(tau), 1)
                    return poly[0] * 2.0
                except (ValueError, np.linalg.LinAlgError):
                    return np.nan
            return x.rolling(w).apply(_h_calc, raw=True)
        ff["hurst_100"] = _rolling_hurst(_log_p, 100, 20).values

    # ── Structural ──
    if _want("var_ratio_8_96"):
        ff["var_ratio_8_96"] = (_log_r8.rolling(96, min_periods=48).var() /
                                 (8.0 * _log_r1.rolling(96, min_periods=48).var().replace(0, np.nan))).values
    if _want("var_ratio_4_48"):
        ff["var_ratio_4_48"] = (_log_r4.rolling(48, min_periods=24).var() /
                                 (4.0 * _log_r1.rolling(48, min_periods=24).var().replace(0, np.nan))).values

    if _want("cusum_breaks_100"):
        def _cusum_breaks(r: pd.Series, w: int = 100, h_mult: float = 2.0):
            sd = r.rolling(w).std()
            h = h_mult * sd
            s_pos = np.zeros(len(r)); s_neg = np.zeros(len(r))
            breaks = np.zeros(len(r), dtype=int)
            rr = r.fillna(0).to_numpy()
            h_arr = h.fillna(np.inf).to_numpy()
            for i in range(1, len(rr)):
                s_pos[i] = max(0.0, s_pos[i-1] + rr[i])
                s_neg[i] = min(0.0, s_neg[i-1] + rr[i])
                if s_pos[i] > h_arr[i]:
                    breaks[i] = 1; s_pos[i] = 0.0
                elif s_neg[i] < -h_arr[i]:
                    breaks[i] = 1; s_neg[i] = 0.0
            return pd.Series(breaks, index=r.index).rolling(w).sum().values
        ff["cusum_breaks_100"] = _cusum_breaks(_log_r1, 100)

    # ── Time ──
    ot = pd.to_datetime(bars["open_time"], unit="ms" if bars["open_time"].dtype.kind in "iu" else None)
    hour = ot.dt.hour + ot.dt.minute / 60.0
    ff["hour_sin"] = np.sin(2 * np.pi * hour / 24.0).values
    ff["hour_cos"] = np.cos(2 * np.pi * hour / 24.0).values
    dow = ot.dt.dayofweek
    ff["dow_cos"]  = np.cos(2 * np.pi * dow / 7.0).values
    mins_into_funding = ((ot.dt.hour % 8) * 60 + ot.dt.minute).astype(float)
    ff["mins_to_funding_norm"] = ((480.0 - mins_into_funding) / 480.0).values

    # ── V2 lag — shift(1) on ALL added features except time/no-lag ──
    added_cols = [c for c in ff.columns if c in META_FEATURE_POOL_35 and c not in NO_LAG_FEATURES]
    for c in added_cols:
        ff[c] = ff[c].shift(1)

    return ff


def build_meta_dataset(primary_bundle: dict, primary_threshold: float,
                       calibration: str = "isotonic_cv5",
                       compute_only: list[str] | None = None,
                       ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """From a primary bundle, build (meta_train_df, meta_test_df, info).

    - Re-loads bars at the primary's bar_size
    - V2-lagged primary features + adds 35-pool features
    - Calibrates primary's cv_oos probas
    - Filters cv_oos rows where calibrated proba > primary_threshold (canonical AFML)
    - Builds test set analog from df_test + final_model proba

    Returns:
        meta_train: rows from cv_oos (primary longs), with 35 features + meta_y + ret + t1_time
        meta_test:  rows from df_test (primary longs), with 35 features + meta_y_true + ret + t1_time
        info: dict with diagnostic stats
    """
    cfg = primary_bundle["config"]
    bar_size = cfg["bar_size"]

    bars = pd.read_parquet(f"data/processed/dollar_bars_{bar_size}_{SYMBOL}.parquet")
    # Defensive: drop duplicate close_time rows (can happen at bar_size=3 dense bars)
    bars = bars.drop_duplicates(subset="close_time", keep="last").reset_index(drop=True)
    features_full = compute_lagged_bar_features(bars)
    features_full = _add_regime_features(features_full, bars, compute_only=compute_only)
    # Also dedupe features_full close_time
    features_full = features_full.drop_duplicates(subset="close_time", keep="last").reset_index(drop=True)

    # ── Calibrate primary probabilities ──
    # First dedupe RAW cv_oos before calibration (calibration assumes unique rows)
    cv_oos_input = primary_bundle["cv_oos"].drop_duplicates(subset="close_time", keep="last").reset_index(drop=True)
    calibrator, cv_oos_calib = calibrate_cv_oos(cv_oos_input, method=calibration)
    # Defensive: dedupe again after calibration
    cv_oos_calib = cv_oos_calib.drop_duplicates(subset="close_time", keep="last").reset_index(drop=True)

    # Dedupe df_trainval and df_test from bundle (both may have duplicate close_times at bar=3/5)
    df_trainval_dedup = primary_bundle.get("df_trainval")
    if df_trainval_dedup is not None:
        df_trainval_dedup = df_trainval_dedup.drop_duplicates(subset="close_time", keep="last").reset_index(drop=True)

    # ── meta_train: filter cv_oos by primary_threshold ──
    meta_train = cv_oos_calib[cv_oos_calib["proba_1"] > primary_threshold].copy()
    meta_train["meta_y"] = (meta_train["y_true"] == 1).astype(int)

    # Join close_time → features. If compute_only is set, only use computed cols.
    DERIVED_NON_REGIME = {"proba_long_calib", "proba_margin", "proba_entropy",
                          "proba_extremity", "hit_rate_50_causal"}
    if compute_only is not None:
        regime_cols = [c for c in compute_only
                       if c not in DERIVED_NON_REGIME and c in features_full.columns]
    else:
        regime_cols = [c for c in META_FEATURE_POOL_35 if c not in DERIVED_NON_REGIME]
    state_cols = ["close_time"] + regime_cols
    state = features_full[state_cols].copy()
    meta_train = meta_train.merge(state, on="close_time", how="left", suffixes=("", "_state"))

    # Join t1_time (3-way fallback for backward compat with old bundles)
    if "t1_time" in meta_train.columns:
        pass  # already present from cv_oos (some run_cv versions include it)
    elif df_trainval_dedup is not None:
        meta_train = meta_train.merge(
            df_trainval_dedup[["close_time", "t1_time"]],
            on="close_time", how="left",
        )
    else:
        # Fallback: re-load labels parquet from cache, merge on close_time
        cache_path = labels_cache_path(SYMBOL, cfg["bar_size"], cfg["span"],
                                        cfg["pt"], cfg["sl"], cfg["max_hold"])
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Old bundle without df_trainval AND no labels cache at {cache_path}. "
                f"Re-run train_batch for this primary OR generate labels first."
            )
        labels_full = pd.read_parquet(cache_path)
        meta_train = meta_train.merge(
            labels_full[["close_time", "t1_time"]],
            on="close_time", how="left",
        )
    if meta_train["t1_time"].isna().any():
        n_miss = meta_train["t1_time"].isna().sum()
        raise ValueError(f"t1_time join left {n_miss} NaN rows — close_time mismatch")

    # Primary-derived features
    p = meta_train["proba_1"].clip(1e-6, 1 - 1e-6)
    meta_train["proba_long_calib"] = p
    meta_train["proba_margin"]     = (p - 0.5).abs()
    meta_train["proba_entropy"]    = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    meta_train["proba_extremity"]  = (p - 0.5) ** 2

    # Causal hit_rate (sort by close_time first)
    meta_train = meta_train.sort_values("close_time").reset_index(drop=True)
    meta_train["hit_rate_50_causal"] = causal_hit_rate(meta_train, y_col="meta_y", window=50, min_periods=20)

    # ── meta_test: from df_test + final_model proba (calibrated) ──
    final_model = primary_bundle["final_model"]
    long_col = primary_bundle["long_col"]
    test_proba_raw = primary_bundle["test_proba_raw"]
    # Defensive: dedupe df_test AND test_proba_raw together (row-aligned)
    df_test_orig = primary_bundle["df_test"]
    keep_mask = ~df_test_orig.duplicated(subset="close_time", keep="last")
    df_test = df_test_orig[keep_mask].reset_index(drop=True)
    test_proba_raw = test_proba_raw[keep_mask.values]
    test_proba_calib = calibrator.transform(test_proba_raw)

    meta_test = df_test[["close_time", "t1_time", "ret", "label"]].copy().reset_index(drop=True)
    meta_test["proba_1"] = test_proba_calib
    meta_test = meta_test[meta_test["proba_1"] > primary_threshold].copy().reset_index(drop=True)
    meta_test["meta_y_true"] = (meta_test["label"] == 1).astype(int)
    meta_test = meta_test.merge(state, on="close_time", how="left")

    p_t = meta_test["proba_1"].clip(1e-6, 1 - 1e-6)
    meta_test["proba_long_calib"] = p_t
    meta_test["proba_margin"]     = (p_t - 0.5).abs()
    meta_test["proba_entropy"]    = -(p_t * np.log(p_t) + (1 - p_t) * np.log(1 - p_t))
    meta_test["proba_extremity"]  = (p_t - 0.5) ** 2

    # For hit_rate on test, use train history as warmup
    history = meta_train[["close_time", "t1_time", "meta_y"]].rename(columns={"meta_y": "outcome"})
    test_for_hist = meta_test[["close_time", "t1_time", "meta_y_true"]].rename(columns={"meta_y_true": "outcome"})
    combined = pd.concat([
        history.assign(is_test=False),
        test_for_hist.assign(is_test=True),
    ], ignore_index=True).sort_values("close_time").reset_index(drop=True)
    combined["hit_rate_50_causal"] = causal_hit_rate(combined, y_col="outcome", window=50, min_periods=20)
    test_hr_map = combined[combined["is_test"]].set_index("close_time")["hit_rate_50_causal"]
    meta_test["hit_rate_50_causal"] = meta_test["close_time"].map(test_hr_map)

    # Drop rows with NaN in features (early bars before rolling windows fill)
    # dropna only on features actually present (compute_only may have skipped some)
    dropna_features = compute_only if compute_only is not None else META_FEATURE_POOL_35
    dropna_features = [c for c in dropna_features if c in meta_train.columns]
    meta_train = meta_train.dropna(subset=dropna_features + ["meta_y"]).reset_index(drop=True)
    dropna_test = [c for c in dropna_features if c in meta_test.columns]
    meta_test  = meta_test.dropna(subset=dropna_test + ["meta_y_true"]).reset_index(drop=True)

    info = {
        "calibration":       calibration,
        "primary_threshold": primary_threshold,
        "n_meta_train":      len(meta_train),
        "n_meta_test":       len(meta_test),
        "meta_y_base_rate":  float(meta_train["meta_y"].mean()),
        "test_baseline_prec": float(meta_test["meta_y_true"].mean()),
    }
    return meta_train, meta_test, info


# ── Walk-Forward Validation (4 windows inside meta_train) ─────────────────────

def walkforward_eval(meta_train: pd.DataFrame, features: list[str],
                     rf_params: dict, n_windows: int = 4,
                     min_n_per_window: int = 200) -> dict:
    """Train meta on chronological windows. THRESHOLD-FREE AUC-PR metrics.

    Per-window: auc_pr_train, auc_pr_val, base_rate, auc_edge.
    Threshold selection is DELIBERATELY done OUTSIDE this function via
    threshold_sweep_on_meta_test(). This separates 'signal quality' (Optuna's
    job) from 'operating point' (the user's choice, post-Optuna).
    """
    n = len(meta_train)
    if n < min_n_per_window * 2:
        return {"valid": False, "reason": "meta_train too small"}

    meta_train = meta_train.sort_values("close_time").reset_index(drop=True)
    X = meta_train[features].to_numpy(dtype=float)
    y = meta_train["meta_y"].to_numpy()
    w = meta_train["ret"].abs().to_numpy()

    train_ends = [0.50, 0.65, 0.80, 0.85][:n_windows]
    val_starts = [t for t in train_ends]
    val_ends   = [min(1.0, t + 0.15) for t in val_starts]

    results = []
    for tr_end, va_st, va_end in zip(train_ends, val_starts, val_ends):
        tr_idx = slice(0, int(n * tr_end))
        va_idx = slice(int(n * va_st), int(n * va_end))
        Xtr, ytr, wtr = X[tr_idx], y[tr_idx], w[tr_idx]
        Xva, yva = X[va_idx], y[va_idx]
        if len(Xtr) < min_n_per_window or len(Xva) < min_n_per_window:
            continue

        mdl = RandomForestClassifier(**rf_params)
        mdl.fit(Xtr, ytr, sample_weight=wtr)
        long_col = int(np.where(mdl.classes_ == 1)[0][0])

        # AUC-PR on train (for overfit gap)
        proba_tr = mdl.predict_proba(Xtr)[:, long_col]
        p_tr, r_tr, _ = precision_recall_curve(ytr, proba_tr)
        auc_pr_train = float(auc(r_tr, p_tr))

        # AUC-PR on validation (the real signal quality)
        proba_va = mdl.predict_proba(Xva)[:, long_col]
        p_va, r_va, _ = precision_recall_curve(yva, proba_va)
        auc_pr_val = float(auc(r_va, p_va))

        base_rate = float(yva.mean())
        auc_edge  = auc_pr_val - base_rate
        gap       = auc_pr_train - auc_pr_val

        results.append({
            "auc_pr_train": auc_pr_train,
            "auc_pr_val":   auc_pr_val,
            "base_rate":    base_rate,
            "auc_edge":     auc_edge,
            "gap":          gap,
            "n_val":        int(len(yva)),
            "valid":        True,
        })

    if not results:
        return {"valid": False, "reason": "no valid windows"}

    edges  = np.array([r["auc_edge"] for r in results])
    gaps   = np.array([r["gap"]      for r in results])
    aucs_v = np.array([r["auc_pr_val"]   for r in results])
    aucs_t = np.array([r["auc_pr_train"] for r in results])
    bases  = np.array([r["base_rate"] for r in results])

    return {
        "valid":             True,
        "windows":           results,
        "n_windows_valid":   len(results),
        "mean_auc_pr_val":   float(aucs_v.mean()),
        "mean_auc_pr_train": float(aucs_t.mean()),
        "mean_base_rate":    float(bases.mean()),
        "mean_auc_edge":     float(edges.mean()),
        "auc_edge_std":      float(edges.std()),
        "positive_edge_fraction": float((edges > 0).mean()),
        "mean_gap":          float(gaps.mean()),
    }


def composite_score_pure_model(wf: dict) -> float:
    """Threshold-AGNOSTIC signal-quality score (AUC-PR based).

    Components (each normalized 0-1):
      SIGNAL    = clip(mean_auc_edge * 5, 0, 1)         # +0.20 AUC edge → 1.0
      STABILITY = 0.5*pos_edge_fraction + 0.5/(1+CV)    # CV = std/|mean| of edges
      OVERFIT   = clip(mean_gap * 5, 0, 1)              # +0.10 train-val AUC gap → 0.5

    Composite:
      score = 0.6*SIGNAL + 0.3*STABILITY - 0.1*OVERFIT

    Hard gates:
      mean_auc_edge <= 0  → -1e6  (no real signal)
      positive_edge_fraction < 0.5  → -1e6  (signal not stable across windows)

    Note: NO threshold inside scoring. Operating point chosen AFTER Optuna
    via threshold_sweep_on_meta_test(). This is the AFML Ch 11 separation
    of 'signal quality' from 'execution decision'.
    """
    if not wf.get("valid"):
        return -1e6
    if wf["mean_auc_edge"] <= 0:
        return -1e6
    if wf["positive_edge_fraction"] < 0.5:
        return -1e6

    # SIGNAL
    signal = float(np.clip(wf["mean_auc_edge"] * 5, 0.0, 1.0))

    # STABILITY
    mean_e = wf["mean_auc_edge"]
    std_e  = wf["auc_edge_std"]
    cv = std_e / (abs(mean_e) + 1e-9)
    stability = 0.5 * wf["positive_edge_fraction"] + 0.5 / (1.0 + cv)

    # OVERFIT
    overfit_penalty = float(np.clip(wf["mean_gap"] * 5, 0.0, 1.0))

    score = 0.6 * signal + 0.3 * stability - 0.1 * overfit_penalty
    return float(score)


# ── Stage 2 — Threshold Sweep (post-Optuna, on meta_test) ────────────────────

def threshold_sweep_on_meta_test(meta_train: pd.DataFrame, meta_test: pd.DataFrame,
                                  features: list[str], rf_params: dict,
                                  thresholds: np.ndarray | None = None) -> pd.DataFrame:
    """Train meta on FULL meta_train, sweep thresholds on meta_test.

    Returns a DataFrame with per-threshold metrics so the user can pick an
    operating point. This is the post-Optuna 'operating decision' stage.
    """
    if thresholds is None:
        thresholds = np.arange(0.30, 0.71, 0.01)

    Xtr = meta_train[features].to_numpy(dtype=float)
    ytr = meta_train["meta_y"].to_numpy()
    wtr = meta_train["ret"].abs().to_numpy()
    mdl = RandomForestClassifier(**rf_params).fit(Xtr, ytr, sample_weight=wtr)
    long_col = int(np.where(mdl.classes_ == 1)[0][0])

    Xte = meta_test[features].to_numpy(dtype=float)
    yte = meta_test["meta_y_true"].to_numpy()
    retval = meta_test["ret"].to_numpy()
    proba = mdl.predict_proba(Xte)[:, long_col]
    baseline_prec = yte.mean()

    rows = []
    for thr in thresholds:
        keep = proba > thr
        n = int(keep.sum())
        if n == 0:
            rows.append({"threshold": float(thr), "n_keep": 0, "valid": False})
            continue
        prec       = float(yte[keep].mean())
        prec_lift  = float(prec - baseline_prec)
        rec_share  = float(n / len(yte))
        gross_pnl  = float(retval[keep].sum())
        net_maker  = float(gross_pnl - n * RT_COST_MAKER)
        net_taker  = float(gross_pnl - n * RT_COST_TAKER)
        rows.append({
            "threshold":        float(thr),
            "n_keep":           n,
            "precision":        prec,
            "prec_lift":        prec_lift,
            "rec_share":        rec_share,
            "gross_pnl_log":    gross_pnl,
            "net_maker_log":    net_maker,
            "net_taker_log":    net_taker,
            "net_maker_bps_per_trade": float(net_maker / n * 1e4),
            "net_taker_bps_per_trade": float(net_taker / n * 1e4),
            "valid":            True,
        })

    df = pd.DataFrame(rows)
    df.attrs["baseline_prec"] = float(baseline_prec)
    df.attrs["meta_test_n"]   = int(len(yte))
    return df


# ── Optuna Search Space + Objective ───────────────────────────────────────────

def trial_features(trial: optuna.Trial) -> list[str]:
    """Optuna picks a subset of META_FEATURE_POOL_35 (always-on cores + binary opts)."""
    # Always-on cores (proven across iter1-7)
    CORE = ["atr_ratio", "rstd_192"]
    selected = list(CORE)
    optional = [f for f in META_FEATURE_POOL_35 if f not in CORE]
    for feat in optional:
        if trial.suggest_categorical(f"use_{feat}", [True, False]):
            selected.append(feat)
    # Cap at 15 features to prevent overfit / preserve diversity
    if len(selected) > 15:
        selected = CORE + [f for f in selected if f not in CORE][:13]
    if len(selected) < 3:
        selected = CORE + ["dvol_z_48"]
    return selected


def meta_objective_factory(primary_bundle: dict, log: logging.Logger | None = None):
    """Build an Optuna objective bound to a specific primary bundle.

    NO meta_threshold in search space — threshold is chosen post-Optuna via
    threshold_sweep_on_meta_test() on the best trial.
    """
    def objective(trial: optuna.Trial) -> float:
        # Wrap EVERYTHING in try/except so a single bad trial never crashes the study
        try:
            # Search space — narrowed based on quick_meta_eval findings
            # (drop primary_threshold > 0.50 which usually fails hard gate;
            #  keep "none" calibration for diversity but isotonic_cv5 dominant)
            calibration = trial.suggest_categorical("calibration", ["none", "isotonic_posthoc", "isotonic_cv5"])
            primary_threshold = trial.suggest_float("primary_threshold", 0.40, 0.50, step=0.01)
            meta_max_depth        = trial.suggest_categorical("meta_max_depth",        [3, 4, 5])
            meta_min_samples_leaf = trial.suggest_categorical("meta_min_samples_leaf", [50, 100, 200])
            meta_n_estimators     = trial.suggest_categorical("meta_n_estimators",     [200, 500])

            # Feature subset (binary per feature)
            features = trial_features(trial)
            trial.set_user_attr("n_features", len(features))
            trial.set_user_attr("features", features)

            rf_params = {
                "n_estimators":      meta_n_estimators,
                "max_depth":         meta_max_depth,
                "min_samples_leaf":  meta_min_samples_leaf,
                "class_weight":      "balanced",
                "random_state":      42,
                "n_jobs":            -1,
            }

            try:
                meta_train, meta_test, info = build_meta_dataset(
                    primary_bundle, primary_threshold, calibration=calibration,
                    compute_only=META_FEATURE_POOL_35,   # full pool — Optuna picks subset
                )
            except Exception as e:
                if log: log.warning(f"  trial {trial.number}: build failed: {e}")
                return -1e6

            if info["n_meta_train"] < 500:                              return -1e6
            if not (0.20 <= info["meta_y_base_rate"] <= 0.80):           return -1e6

            wf = walkforward_eval(meta_train, features, rf_params)
            score = composite_score_pure_model(wf)

            # Diagnostic user_attrs (for post-Optuna analysis)
            if wf.get("valid"):
                trial.set_user_attr("mean_auc_pr_val",   wf["mean_auc_pr_val"])
                trial.set_user_attr("mean_auc_pr_train", wf["mean_auc_pr_train"])
                trial.set_user_attr("mean_base_rate",    wf["mean_base_rate"])
                trial.set_user_attr("mean_auc_edge",     wf["mean_auc_edge"])
                trial.set_user_attr("auc_edge_std",      wf["auc_edge_std"])
                trial.set_user_attr("positive_edge_fraction", wf["positive_edge_fraction"])
                trial.set_user_attr("mean_gap",          wf["mean_gap"])
            trial.set_user_attr("n_meta_train", info["n_meta_train"])
            trial.set_user_attr("n_meta_test",  info["n_meta_test"])
            return score
        except Exception as e:
            # Catch-all: never crash the study
            if log: log.exception(f"  trial {trial.number}: unexpected error: {e}")
            return -1e6
    return objective


# ── Final Held-Out Test Evaluation (post-Optuna, threshold sweep) ────────────

def evaluate_on_test(primary_bundle: dict, params: dict, log: logging.Logger) -> dict:
    """Re-train meta with `params` on full meta_train, run threshold sweep on
    meta_test. Returns AUC-PR (threshold-free) + full sweep table.

    The user picks operating threshold from the sweep table (Stage 2).
    """
    calibration       = params["calibration"]
    primary_threshold = params["primary_threshold"]
    rf_params = {
        "n_estimators":      params["meta_n_estimators"],
        "max_depth":         params["meta_max_depth"],
        "min_samples_leaf":  params["meta_min_samples_leaf"],
        "class_weight":      "balanced",
        "random_state":      42,
        "n_jobs":            -1,
    }
    features = params["features"]

    meta_train, meta_test, info = build_meta_dataset(
        primary_bundle, primary_threshold, calibration=calibration,
    )

    # ── Threshold-free AUC-PR on meta_test ──
    Xtr = meta_train[features].to_numpy(dtype=float)
    ytr = meta_train["meta_y"].to_numpy()
    wtr = meta_train["ret"].abs().to_numpy()
    mdl = RandomForestClassifier(**rf_params).fit(Xtr, ytr, sample_weight=wtr)
    long_col = int(np.where(mdl.classes_ == 1)[0][0])

    Xte = meta_test[features].to_numpy(dtype=float)
    yte = meta_test["meta_y_true"].to_numpy()
    proba_te = mdl.predict_proba(Xte)[:, long_col]
    p_te, r_te, _ = precision_recall_curve(yte, proba_te)
    test_auc_pr = float(auc(r_te, p_te))
    base_rate_test = float(yte.mean())

    # ── Threshold sweep on meta_test ──
    sweep_df = threshold_sweep_on_meta_test(meta_train, meta_test, features, rf_params)

    # Pick a few "interesting" thresholds for the summary log
    valid = sweep_df[sweep_df["valid"]].copy()
    if len(valid) > 0:
        # Best by net_maker per trade
        best_maker_thr = valid.loc[valid["net_maker_bps_per_trade"].idxmax()]
        # Best by precision_lift (rec_share >= 0.30 constraint)
        valid_rec = valid[valid["rec_share"] >= 0.30]
        best_prec_thr = (valid_rec.loc[valid_rec["prec_lift"].idxmax()]
                         if len(valid_rec) > 0 else valid.iloc[0])
    else:
        best_maker_thr = best_prec_thr = None

    return {
        "n_meta_train":     info["n_meta_train"],
        "n_meta_test":      info["n_meta_test"],
        "base_rate_test":   base_rate_test,
        "test_auc_pr":      test_auc_pr,
        "test_auc_edge":    float(test_auc_pr - base_rate_test),
        "sweep":            sweep_df.to_dict(orient="records"),
        "best_maker_threshold":  None if best_maker_thr is None else float(best_maker_thr["threshold"]),
        "best_maker_bps":        None if best_maker_thr is None else float(best_maker_thr["net_maker_bps_per_trade"]),
        "best_prec_threshold":   None if best_prec_thr  is None else float(best_prec_thr["threshold"]),
        "best_prec_lift":        None if best_prec_thr  is None else float(best_prec_thr["prec_lift"]),
        "params":           params,
    }


# ── Main runner ──────────────────────────────────────────────────────────────

def run_meta_study(primary_pkl_path: Path, output_dir: Path,
                   n_trials: int = 200) -> dict:
    """Run Optuna meta study on one primary, evaluate top trials on test, save."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log, _, _ = setup_logging(run_name=f"meta_{primary_pkl_path.stem}_{datetime.now():%Y%m%d_%H%M%S}")

    # Route Optuna's per-trial logger to our file handlers (else file stays empty)
    optuna_logger = optuna.logging.get_logger("optuna")
    for h in log.handlers:
        if isinstance(h, logging.FileHandler):
            optuna_logger.addHandler(h)

    log.info(f"\n=== Meta study for {primary_pkl_path.name} ===")
    bundle = joblib.load(primary_pkl_path)
    cfg = bundle["config"]
    primary_tag = (
        f"pick{cfg.get('pick_id','?'):>02}_"
        f"t{cfg.get('trial_number','?')}_"
        f"{cfg.get('group_tag','?')}_"
        f"bar{cfg.get('bar_size','?')}_"
        f"{cfg.get('pt_sl','?')}"
    )
    log.info(f"  primary_tag: {primary_tag}")
    log.info(f"  primary cfg: {cfg}")

    objective = meta_objective_factory(bundle, log=log)

    # ── Persistent storage so we can resume on crash + inspect post-hoc ──
    storage_path = output_dir / f"optuna_{primary_pkl_path.stem}.db"
    storage_url = f"sqlite:///{storage_path}"
    log.info(f"  storage: {storage_url}  (resumable on crash)")

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(
            seed=42,
            n_startup_trials=30,      # more exploration before exploitation
            n_ei_candidates=24,        # candidates per acquisition step
            multivariate=True,         # consider param correlations
        ),
        study_name=f"meta_{primary_pkl_path.stem}",
        storage=storage_url,
        load_if_exists=True,           # resume if .db already has trials
    )

    # Per-trial concise summary callback (writes to our file too)
    def _trial_log_callback(study_, trial):
        attrs = trial.user_attrs
        val = trial.value if trial.value is not None else float("nan")
        features = attrs.get("features", [])
        log.info(
            f"  [{primary_tag}]  T{trial.number:>3d}  score={val:>+8.4f}  "
            f"auc_edge={attrs.get('mean_auc_edge', 0):+.4f}  "
            f"pos_frac={attrs.get('positive_edge_fraction', 0):.2f}  "
            f"gap={attrs.get('mean_gap', 0):+.3f}  "
            f"n_feat={len(features)}  "
            f"cal={str(trial.params.get('calibration', '?'))[:8]:<8}  "
            f"prim_thr={trial.params.get('primary_threshold', 0):.2f}  "
            f"depth={trial.params.get('meta_max_depth', '?')}  "
            f"leaf={trial.params.get('meta_min_samples_leaf', '?')}"
        )
        if features:
            log.info(f"        features({len(features)}): {','.join(features)}")

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False,
                   callbacks=[_trial_log_callback])

    # ── Top 5 trials → threshold sweep on held-out test ──
    top_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else -1e9, reverse=True)[:5]
    log.info(f"\n=== TOP 5 trials (threshold-free composite score) ===")
    test_evals = []
    for i, t in enumerate(top_trials, 1):
        attrs = t.user_attrs
        log.info(
            f"  #{i}  trial {t.number}  score={t.value:.4f}  "
            f"auc_pr={attrs.get('mean_auc_pr_val', 0):.4f}  "
            f"auc_edge={attrs.get('mean_auc_edge', 0):+.4f}  "
            f"pos_frac={attrs.get('positive_edge_fraction', 0):.2f}  "
            f"gap={attrs.get('mean_gap', 0):.3f}"
        )
        params = dict(t.params)
        params["features"] = attrs.get("features", [])
        try:
            te = evaluate_on_test(bundle, params, log)
            te["trial_number"] = t.number
            te["wf_score"]     = t.value
            test_evals.append(te)
            log.info(
                f"       TEST AUC-PR: {te['test_auc_pr']:.4f}  edge={te['test_auc_edge']:+.4f}  "
                f"(baseline={te['base_rate_test']:.4f})"
            )
            if te["best_maker_threshold"] is not None:
                log.info(
                    f"       Best operating point (maker, net bps/trade): "
                    f"thr={te['best_maker_threshold']:.2f}  {te['best_maker_bps']:+.2f} bps/trade"
                )
            if te["best_prec_threshold"] is not None:
                log.info(
                    f"       Best operating point (prec_lift @ recall>=0.30): "
                    f"thr={te['best_prec_threshold']:.2f}  lift={te['best_prec_lift']:+.4f}"
                )
            # Print threshold sweep table for this trial
            sweep_rows = [r for r in te["sweep"] if r.get("valid")]
            if sweep_rows:
                log.info("       Threshold sweep (held-out test, top trial):")
                log.info(f"       {'thr':>5} {'n_keep':>7} {'prec':>7} {'lift':>+7} "
                         f"{'rec':>6} {'maker':>+7} {'taker':>+7}")
                for r in sweep_rows:
                    log.info(
                        f"       {r['threshold']:>5.2f} {r['n_keep']:>7d} {r['precision']:>7.4f} "
                        f"{r['prec_lift']:>+7.4f} {r['rec_share']:>6.3f} "
                        f"{r['net_maker_bps_per_trade']:>+7.2f} {r['net_taker_bps_per_trade']:>+7.2f}"
                    )
        except Exception as e:
            log.warning(f"       test eval failed: {e}")

    # Save study summary
    summary_path = output_dir / f"meta_summary_{primary_pkl_path.stem}.json"
    with open(summary_path, "w") as f:
        json.dump({
            "primary":       primary_pkl_path.name,
            "n_trials":      n_trials,
            "best_wf_score": study.best_value,
            "best_params":   dict(study.best_params),
            "best_features": study.best_trial.user_attrs.get("features", []),
            "top_5_test_evals": test_evals,
        }, f, indent=2, default=str)
    log.info(f"\n  summary → {summary_path}")
    return {"primary": primary_pkl_path.name, "best_score": study.best_value,
            "best_params": dict(study.best_params), "test_evals": test_evals}


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    MODELS_DIR   = PROJECT_ROOT / "models" / "primary"
    OUTPUT_DIR   = PROJECT_ROOT / "models" / "meta"

    # ── OVERNIGHT RUN — Top 5 primaries (from quick_meta_eval) ──
    # Each gets its own sqlite db, log file, summary JSON. Resumable on crash.
    # TPE: 30 startup trials + multivariate sampler (focused on right area).
    # 5 × 200 trials × ~30s/trial = ~8.5 hours total estimated.
    PRIMARIES_TO_RUN = [
        "primary_08_t930.pkl",    # ⭐ best credible: bar=25, 1.0_1.0, +14.24bps × 382 trades
        "primary_09_t1623.pkl",   # same family as #08: bar=25, 1.0_1.0, +34.13bps × 137 (verify)
        "primary_15_t615.pkl",    # diff family long-biased: bar=25, 0.8_1.2, +6.22bps × 324
        "primary_13_t1305.pkl",   # diff bar long-biased: bar=15, 1.0_1.5, +5.86bps × 400
        "primary_04_t551.pkl",    # ⚠ +111bps suspicious — verify real or sample artifact
    ]
    N_TRIALS_PER = 200

    print(f"=== OVERNIGHT META STUDY — {len(PRIMARIES_TO_RUN)} primaries × {N_TRIALS_PER} trials ===")
    print(f"  Estimated total: ~{len(PRIMARIES_TO_RUN) * N_TRIALS_PER * 30 / 3600:.1f} hours")
    print(f"  Output dir: {OUTPUT_DIR}")
    print()

    results_all = []
    for i, pkl_name in enumerate(PRIMARIES_TO_RUN, 1):
        pkl_path = MODELS_DIR / pkl_name
        print(f"\n{'='*80}")
        print(f"[{i}/{len(PRIMARIES_TO_RUN)}] Starting meta study for {pkl_name}")
        print(f"{'='*80}")
        if not pkl_path.exists():
            print(f"  SKIP: {pkl_path} not found")
            results_all.append({"primary": pkl_name, "status": "SKIP_not_found"})
            continue
        try:
            result = run_meta_study(pkl_path, OUTPUT_DIR, n_trials=N_TRIALS_PER)
            result["status"] = "OK"
            results_all.append(result)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            results_all.append({"primary": pkl_name, "status": "FAILED", "error": str(e)})
            # Continue to next primary even on crash

    # Aggregated overnight summary
    overnight_summary_path = OUTPUT_DIR / "overnight_run_summary.json"
    with open(overnight_summary_path, "w") as f:
        json.dump(results_all, f, indent=2, default=str)
    print(f"\n=== OVERNIGHT RUN COMPLETE ===")
    print(f"  Aggregated summary → {overnight_summary_path}")
    n_ok = sum(1 for r in results_all if r.get("status") == "OK")
    print(f"  Successful: {n_ok}/{len(PRIMARIES_TO_RUN)}")
