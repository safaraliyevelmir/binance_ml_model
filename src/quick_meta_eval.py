from __future__ import annotations

import warnings
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, auc

warnings.filterwarnings("ignore")

from src.meta import build_meta_dataset

# ── Trial #43 recipe ──
META_CONFIG = {
    "calibration":       "isotonic_cv5",
    "primary_threshold": 0.45,   # slightly higher than 0.42 — works for more primaries
}
META_FEATURES = [
    "atr_ratio", "rstd_192",
    "atr_normalized",
    "rstd_24", "rstd_96", "rstd_12_192",
    "vol_of_vol_48",
    "ret_6", "ret_24",
    "mom_zscore_24",
    "log_return",
    "vol_z_48",
    "vwap_dev_24", "vwap_dev_96", "vwap_dev_96_z",
]
META_RF_PARAMS = {
    "n_estimators":      200,
    "max_depth":         5,
    "min_samples_leaf":  100,
    "class_weight":      "balanced",
    "random_state":      42,
    "n_jobs":            -1,
}
RT_COST_MAKER = 6 / 1e4
RT_COST_TAKER = 10 / 1e4
THRESHOLDS    = np.arange(0.30, 0.71, 0.01)
MIN_KEEP      = 50


def eval_one_primary(pkl_path: Path) -> dict:
    bundle = joblib.load(pkl_path)
    cfg = bundle["config"]
    tag = f"#{cfg.get('pick_id'):02d}_t{cfg.get('trial_number')}_{cfg.get('group_tag')}"

    try:
        # Only compute features we use → 3-5× speedup (skips Hurst, CUSUM breaks, etc.)
        meta_train, meta_test, info = build_meta_dataset(
            bundle,
            primary_threshold=META_CONFIG["primary_threshold"],
            calibration=META_CONFIG["calibration"],
            compute_only=META_FEATURES,
        )
    except Exception as e:
        return {"tag": tag, "status": f"FAIL_build: {e}"}

    if len(meta_train) < 500 or len(meta_test) < 100:
        return {"tag": tag, "status": f"FAIL_small (train={len(meta_train)}, test={len(meta_test)})"}

    # Train meta
    Xtr = meta_train[META_FEATURES].to_numpy(dtype=float)
    ytr = meta_train["meta_y"].to_numpy()
    wtr = meta_train["ret"].abs().to_numpy()
    mdl = RandomForestClassifier(**META_RF_PARAMS).fit(Xtr, ytr, sample_weight=wtr)
    long_col = int(np.where(mdl.classes_ == 1)[0][0])

    # Predict on test
    Xte = meta_test[META_FEATURES].to_numpy(dtype=float)
    yte = meta_test["meta_y_true"].to_numpy()
    retval = meta_test["ret"].to_numpy()
    proba = mdl.predict_proba(Xte)[:, long_col]

    baseline_prec = float(yte.mean())
    p_te, r_te, _ = precision_recall_curve(yte, proba)
    test_auc_pr = float(auc(r_te, p_te))
    test_auc_edge = test_auc_pr - baseline_prec

    # Threshold sweep
    best_maker_thr      = None
    best_maker_bps      = -1e9
    best_maker_n        = 0
    best_maker_prec     = 0.0
    best_maker_lift     = 0.0
    best_maker_total    = 0.0

    for thr in THRESHOLDS:
        keep = proba > thr
        n = int(keep.sum())
        if n < MIN_KEEP:
            continue
        gross = float(retval[keep].sum())
        net_maker = gross - n * RT_COST_MAKER
        per_trade_bps = net_maker / n * 1e4
        if per_trade_bps > best_maker_bps:
            best_maker_thr   = float(thr)
            best_maker_bps   = per_trade_bps
            best_maker_n     = n
            best_maker_prec  = float(yte[keep].mean())
            best_maker_lift  = best_maker_prec - baseline_prec
            best_maker_total = net_maker

    return {
        "tag":            tag,
        "pick_id":        int(cfg.get("pick_id", 0)),
        "trial":          int(cfg.get("trial_number", 0)),
        "group":          cfg.get("group_tag", "?"),
        "bar":            int(cfg.get("bar_size", 0)),
        "pt_sl":          cfg.get("pt_sl", "?"),
        "n_meta_train":   len(meta_train),
        "n_meta_test":    len(meta_test),
        "baseline_prec":  baseline_prec,
        "test_auc_pr":    test_auc_pr,
        "test_auc_edge":  test_auc_edge,
        "best_maker_thr":   best_maker_thr,
        "best_maker_bps":   best_maker_bps if best_maker_thr else None,
        "best_maker_n":     best_maker_n if best_maker_thr else 0,
        "best_maker_prec":  best_maker_prec if best_maker_thr else None,
        "best_maker_lift":  best_maker_lift if best_maker_thr else None,
        "best_maker_total": best_maker_total if best_maker_thr else None,
        "rec_share":        (best_maker_n / len(meta_test)) if best_maker_thr else None,
        "status":         "OK",
    }


def main():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    MODELS_DIR   = PROJECT_ROOT / "models" / "primary"
    OUTPUT_PATH  = PROJECT_ROOT / "models" / "meta" / "quick_leaderboard.csv"

    pkls = sorted(MODELS_DIR.glob("primary_*.pkl"))
    if not pkls:
        print(f"No primaries found in {MODELS_DIR}")
        return

    print(f"\n=== Quick meta-eval across {len(pkls)} primaries ===")
    print(f"  Config: {META_CONFIG}")
    print(f"  Features: {len(META_FEATURES)}")
    print(f"  Output: {OUTPUT_PATH}")
    print()

    results = []
    for pkl in pkls:
        print(f"[{pkl.name}] ... ", end="", flush=True)
        r = eval_one_primary(pkl)
        if r["status"] == "OK":
            print(f"OK  edge={r['test_auc_edge']:+.4f}  maker={r['best_maker_bps']:+.2f}bps  n={r['best_maker_n']}")
        else:
            print(r["status"])
        results.append(r)

    df = pd.DataFrame(results)
    df_ok = df[df["status"] == "OK"].copy()

    # Save full
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    # Print leaderboard
    if len(df_ok) == 0:
        print("\nNo successful primaries!")
        return

    print(f"\n{'='*150}")
    print("LEADERBOARD — sorted by per-trade maker bps (descending)")
    print(f"{'='*150}")
    df_sorted = df_ok.sort_values("best_maker_bps", ascending=False)
    cols = ["pick_id", "trial", "group", "bar", "pt_sl",
            "n_meta_train", "n_meta_test", "baseline_prec",
            "test_auc_pr", "test_auc_edge",
            "best_maker_thr", "best_maker_n", "best_maker_prec", "best_maker_lift",
            "best_maker_bps", "best_maker_total", "rec_share"]
    print(df_sorted[cols].to_string(index=False))

    # Summary highlights
    print(f"\n{'='*80}")
    print("HIGHLIGHTS")
    print(f"{'='*80}")
    n_positive_bps = int((df_ok["best_maker_bps"] > 0).sum())
    n_break_even   = int((df_ok["best_maker_bps"] > -2).sum())
    print(f"  Profitable (net maker > 0 bps/trade): {n_positive_bps}/{len(df_ok)}")
    print(f"  Near break-even (> -2 bps/trade)     : {n_break_even}/{len(df_ok)}")
    if n_positive_bps > 0:
        winners = df_sorted[df_sorted["best_maker_bps"] > 0]
        print(f"\n  WINNERS ({len(winners)}):")
        for _, row in winners.iterrows():
            print(f"    pick #{row['pick_id']:02d} (t{row['trial']}, {row['group']}, bar={row['bar']}, pt_sl={row['pt_sl']}) "
                  f"→ thr={row['best_maker_thr']:.2f}, +{row['best_maker_bps']:.2f}bps/trade × {row['best_maker_n']} trades")

    print(f"\n  Saved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
