from __future__ import annotations

import argparse
import asyncio
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.main import (
    compute_bar_features,
    cusum_filter,
    TEST_RATIO
)
from src.pre_process.trippler_barrier import getDailyVol, label_bars

SYMBOL = "SOLUSDT"

# Trade economics — defaults; override via CLI flags
DEFAULT_MARGIN_USD   = 10.0
DEFAULT_LEVERAGE     = 100.0
DEFAULT_TAKER_FEE    = 0.0005   # 0.05 % per side (Binance perp futures taker)
DEFAULT_FUNDING_RATE = 0.0001   # 0.01 % per 8h (rough Binance avg; longs pay if positive)
FUNDING_PERIOD_H     = 8


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", type=Path,
                   default=ROOT / "models" / "trial_000731_bar3_SOLUSDT.pkl",
                   help="Path to the .pkl model file")
    p.add_argument("--trials-csv", type=Path, default=ROOT / "trials.csv")
    p.add_argument("--margin",   type=float, default=DEFAULT_MARGIN_USD,
                   help=f"Margin per trade in USD (default {DEFAULT_MARGIN_USD})")
    p.add_argument("--leverage", type=float, default=DEFAULT_LEVERAGE,
                   help=f"Leverage multiplier (default {DEFAULT_LEVERAGE})")
    p.add_argument("--taker-fee", type=float, default=DEFAULT_TAKER_FEE,
                   help=f"Taker fee per side, fraction (default {DEFAULT_TAKER_FEE})")
    p.add_argument("--funding-rate", type=float, default=DEFAULT_FUNDING_RATE,
                   help=f"Funding rate per 8h, fraction (default {DEFAULT_FUNDING_RATE})")
    p.add_argument("--skip-2026", action="store_true",
                   help="Only run the 2021-2025 held-out backtest")
    p.add_argument("--save-trades", type=Path, default=None,
                   help="Optional CSV path to dump per-trade ledger for both windows")
    return p.parse_args()


# ── Configuration loading ──────────────────────────────────────────────────────

def load_cfg(model_path: Path, trials_csv: Path) -> dict:
    """Read trial hyperparameters from trials.csv based on the trial number in the filename."""
    m = re.search(r"trial_(\d+)_", model_path.name)
    if not m:
        raise ValueError(f"Could not parse trial number from {model_path.name}")
    trial_no = int(m.group(1))
    df = pd.read_csv(trials_csv).set_index("number")
    if trial_no not in df.index:
        raise KeyError(f"trial {trial_no} not found in {trials_csv}")
    row = df.loc[trial_no]
    pt_str, sl_str = str(row["pt_sl"]).split("_")

    def _b(v) -> bool:
        return v if isinstance(v, bool) else str(v).strip().lower() in ("true", "1", "t", "yes")

    return {
        "trial":     trial_no,
        "bar_size":  int(row["bar_size"]),
        "pt":        float(pt_str),
        "sl":        float(sl_str),
        "span":      int(row["span"]),
        "max_hold":  int(row["max_hold"]),
        "use_cusum": _b(row["use_cusum"]),
    }


# ── Dataset construction (shared for both windows) ────────────────────────────

def build_dataset(bars: pd.DataFrame,
                  cfg: dict,
                  feature_cols: list[str],
                  time_mask: np.ndarray | None = None) -> pd.DataFrame:
    """features on FULL bars → CUSUM → triple-barrier labels → binary.

    Mirrors src/main.py: CUSUM is applied unconditionally (the `use_cusum`
    parameter is recorded in trials but never gates the filter in main.py).
    """
    feats_full = compute_bar_features(bars)
    close_ts = pd.Series(bars["close"].values, index=pd.DatetimeIndex(bars["close_time"]))
    vol_full = getDailyVol(close_ts, span=cfg["span"])
    vol_full = vol_full[~vol_full.index.duplicated(keep="last")]

    if time_mask is None:
        time_mask = np.ones(len(bars), dtype=bool)

    # CUSUM is applied unconditionally to mirror src/main.py exactly. main.py's
    # `use_cusum` parameter is recorded in trials but never actually gates the
    # filter; the search space locks it to True.
    events = cusum_filter(close_ts, h=CUSUM_H)
    cusum_mask = bars["close_time"].isin(events).values
    mask = time_mask & cusum_mask

    bars_sel = bars[mask].reset_index(drop=True)
    feats_sel = feats_full[mask].reset_index(drop=True)
    vol_sel = vol_full.reindex(pd.DatetimeIndex(bars_sel["close_time"])).ffill()

    labels_df = label_bars(bars_sel, vol_sel,
                           pt=cfg["pt"], sl=cfg["sl"], max_hold=cfg["max_hold"])
    labelable = len(labels_df)

    df = feats_sel.iloc[:labelable].copy()
    df["t1_time"] = labels_df["t1_time"].values
    df["label"] = labels_df["label"].values
    df["ret"] = labels_df["return"].values
    df = df[df["label"] != 0].copy()
    df["label"] = (df["label"] == 1).astype(int)
    df = df.dropna(subset=feature_cols).reset_index(drop=True)
    return df


def get_or_build_holdout(cfg: dict, feature_cols: list[str]) -> pd.DataFrame:
    cache = ROOT / f"data/processed/holdout_test_trial{cfg['trial']}_SOLUSDT.parquet"
    if cache.exists():
        print(f"[cache] loaded {cache.name}")
        return pd.read_parquet(cache)

    bars_path = ROOT / f"data/processed/dollar_bars_{cfg['bar_size']}_{SYMBOL}.parquet"
    bars = pd.read_parquet(bars_path).reset_index(drop=True)
    n_test = int(len(bars) * TEST_RATIO)
    test_start = len(bars) - n_test
    print(f"[build] held-out: {n_test:,} of {len(bars):,} bars (last {TEST_RATIO*100:.0f}%)")

    mask = np.zeros(len(bars), dtype=bool)
    mask[test_start:] = True
    df = build_dataset(bars, cfg, feature_cols, time_mask=mask)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    print(f"[built] {cache.name}: {len(df):,} rows")
    return df


def get_or_build_2026q1(cfg: dict, feature_cols: list[str]) -> pd.DataFrame:
    cache = ROOT / f"data/processed/holdout_2026Q1_trial{cfg['trial']}_SOLUSDT.parquet"
    if cache.exists():
        print(f"[cache] loaded {cache.name}")
        return pd.read_parquet(cache)

    download_2026_q1_if_missing()
    ext_bars = get_or_build_extended_bars(cfg)
    Q1_START = pd.Timestamp("2026-01-01")
    Q1_END   = pd.Timestamp("2026-04-01")
    mask = ((ext_bars["close_time"] >= Q1_START) & (ext_bars["close_time"] < Q1_END)).values
    print(f"[build] 2026-Q1: {int(mask.sum()):,} bars in window")

    df = build_dataset(ext_bars, cfg, feature_cols, time_mask=mask)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    print(f"[built] {cache.name}: {len(df):,} rows")
    return df


def download_2026_q1_if_missing() -> None:
    raw_dir = ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    missing = [m for m in (1, 2, 3)
               if not (raw_dir / f"{SYMBOL}_2026_{m:02d}.parquet").exists()]
    if not missing:
        print("[cache] 2026 Q1 raw files already present")
        return

    import aiohttp
    from src.pre_process.collector import download_binance_agg_trades, CONCURRENCY

    async def _go():
        connector = aiohttp.TCPConnector(limit=CONCURRENCY)
        async with aiohttp.ClientSession(connector=connector) as session:
            for m in missing:
                await download_binance_agg_trades(session, SYMBOL, 2026, m)

    print(f"[download] fetching 2026 months: {missing}")
    asyncio.run(_go())


def get_or_build_extended_bars(cfg: dict) -> pd.DataFrame:
    cache = ROOT / f"data/processed/dollar_bars_{cfg['bar_size']}_extended_SOLUSDT.parquet"
    if cache.exists():
        print(f"[cache] loaded {cache.name}")
        return pd.read_parquet(cache)

    from src.pre_process.build_dollar_bars import build_dollar_bars_streaming
    raw_files = sorted((ROOT / "data" / "raw").glob(f"{SYMBOL}_*.parquet"))
    threshold = cfg["bar_size"] * 1_000_000
    print(f"[build] dollar bars: {len(raw_files)} raw files at ${threshold/1e6:.1f}M threshold")
    bars = build_dollar_bars_streaming([str(p) for p in raw_files], threshold)
    cache.parent.mkdir(parents=True, exist_ok=True)
    bars.to_parquet(cache)
    print(f"[built] {cache.name}: {len(bars):,} bars")
    return bars


# ── Backtest core ──────────────────────────────────────────────────────────────

def backtest(df: pd.DataFrame,
             y_pred: np.ndarray,
             *,
             label: str,
             notional: float,
             margin_per_trade: float,
             taker_fee: float,
             funding_rate: float) -> dict:
    """Long-only: enter when y_pred == 1, hold from `close_time` to `t1_time`.

    Returns aggregate stats plus a per-trade `ledger` DataFrame.
    """
    take = (y_pred == 1)
    n = int(take.sum())

    trades = df.loc[take].copy().reset_index(drop=True)

    held_h = (trades["t1_time"] - trades["close_time"]).dt.total_seconds() / 3600.0
    held_h = held_h.clip(lower=0)

    simple_ret = np.exp(trades["ret"].to_numpy()) - 1.0
    gross_pnl  = notional * simple_ret
    fees       = np.full(n, notional * taker_fee * 2.0)
    funding    = notional * funding_rate * (held_h.to_numpy() / FUNDING_PERIOD_H)
    net_pnl    = gross_pnl - fees - funding

    ledger = pd.DataFrame({
        "window": label,
        "entry_time": trades["close_time"].values,
        "exit_time": trades["t1_time"].values,
        "entry_close": trades["close"].values if "close" in trades.columns else np.nan,
        "held_hours": held_h.values,
        "log_return":   trades["ret"].values,
        "simple_return": simple_ret,
        "gross_pnl":    gross_pnl,
        "fee":          fees,
        "funding":      funding,
        "net_pnl":      net_pnl,
    })
    ledger["equity"] = ledger["net_pnl"].cumsum()

    wins   = net_pnl[net_pnl > 0]
    losses = net_pnl[net_pnl < 0]
    n_wins = int((net_pnl > 0).sum())
    win_rate = n_wins / n
    avg_win  = float(wins.mean())   if len(wins)   else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0
    pf = float(wins.sum() / -losses.sum()) if (len(losses) and losses.sum() < 0) else float("inf")

    ret_unit = pd.Series(net_pnl) / notional
    duration_years = (
        trades["close_time"].max() - trades["close_time"].min()
    ).total_seconds() / (365.25 * 86400)
    trades_per_year = n / duration_years if duration_years > 0 else 0.0

    sd = ret_unit.std(ddof=1)
    sharpe = float(ret_unit.mean() / sd * np.sqrt(trades_per_year)) if sd > 0 else 0.0

    downside = ret_unit[ret_unit < 0]
    sd_down = downside.std(ddof=1) if len(downside) > 1 else 0.0
    sortino = float(ret_unit.mean() / sd_down * np.sqrt(trades_per_year)) if sd_down > 0 else 0.0

    equity = ledger["equity"]
    running_max = equity.cummax()
    drawdown = equity - running_max
    max_dd = float(drawdown.min())
    if running_max.max() > 0:
        peak = running_max.where(running_max > 0, np.nan)
        max_dd_pct = float((drawdown / peak).min())
    else:
        max_dd_pct = float("nan")

    return {
        "label":              label,
        "n_trades":           n,
        "n_wins":             n_wins,
        "win_rate":           win_rate,
        "total_gross_pnl":    float(gross_pnl.sum()),
        "total_fees":         float(fees.sum()),
        "total_funding":      float(funding.sum()),
        "total_net_pnl":      float(net_pnl.sum()),
        "return_on_margin":   float(net_pnl.sum() / (margin_per_trade * n)),
        "avg_win":            avg_win,
        "avg_loss":           avg_loss,
        "profit_factor":      pf,
        "max_drawdown":       max_dd,
        "max_drawdown_pct":   max_dd_pct,
        "sharpe":             sharpe,
        "sortino":            sortino,
        "trades_per_year":    float(trades_per_year),
        "duration_years":     float(duration_years),
        "first_trade":        trades["close_time"].min(),
        "last_trade":         trades["close_time"].max(),
        "avg_held_hours":     float(held_h.mean()),
        "ledger":             ledger,
    }


def print_report(stats: dict) -> None:
    print("=" * 78)
    print(f"WINDOW: {stats['label']}")
    print("-" * 78)
    if stats["n_trades"] == 0:
        print("  no trades — model never predicted long.")
        return
    print(f"  trades:           {stats['n_trades']:,}  ({stats['first_trade']}  →  {stats['last_trade']})")
    print(f"  duration:         {stats['duration_years']:.2f} yr  ({stats['trades_per_year']:.0f} trades/yr)")
    print(f"  avg held:         {stats['avg_held_hours']:.2f} h")
    print(f"  win rate:         {stats['win_rate']*100:.2f} %  ({stats['n_wins']:,} wins of {stats['n_trades']:,})")
    print()
    print(f"  gross PnL:        ${stats['total_gross_pnl']:>14,.2f}")
    print(f"  fees:             ${-stats['total_fees']:>14,.2f}")
    print(f"  funding:          ${-stats['total_funding']:>14,.2f}")
    print(f"  NET PnL:          ${stats['total_net_pnl']:>14,.2f}")
    print(f"  return on margin: {stats['return_on_margin']*100:>14,.2f} %  (PnL / total margin deployed)")
    print()
    print(f"  avg win:          ${stats['avg_win']:>14,.4f}")
    print(f"  avg loss:         ${stats['avg_loss']:>14,.4f}")
    print(f"  profit factor:    {stats['profit_factor']:>14,.4f}")
    print(f"  max drawdown:     ${stats['max_drawdown']:>14,.2f}  ({stats['max_drawdown_pct']*100:.2f} %)")
    print(f"  Sharpe (ann.):    {stats['sharpe']:>14,.4f}")
    print(f"  Sortino (ann.):   {stats['sortino']:>14,.4f}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_cli()
    notional = args.margin * args.leverage

    print(f"Model: {args.model.name}")
    print(f"Margin: ${args.margin:.2f} × {args.leverage:.0f}x  ⇒  notional ${notional:.2f}/trade")
    print(f"Fees: taker {args.taker_fee*100:.4f}% per side  |  funding {args.funding_rate*100:.4f}% per {FUNDING_PERIOD_H}h")
    print()

    cfg = load_cfg(args.model, args.trials_csv)
    print(f"Trial {cfg['trial']}: bar={cfg['bar_size']}M  pt/sl={cfg['pt']}/{cfg['sl']}"
          f"  span={cfg['span']}  max_hold={cfg['max_hold']}  cusum={cfg['use_cusum']}")
    print()

    model = joblib.load(args.model)
    feature_cols = list(model.feature_names_in_)

    # ── 1) 2021-2025 held-out 20% ──
    df_h = get_or_build_holdout(cfg, feature_cols)
    y_pred_h = model.predict(df_h[feature_cols])
    stats_h = backtest(
        df_h, y_pred_h,
        label="2021-2025 held-out 20% (never seen)",
        notional=notional,
        margin_per_trade=args.margin,
        taker_fee=args.taker_fee,
        funding_rate=args.funding_rate,
    )
    print_report(stats_h)
    print()

    stats_q = None
    if not args.skip_2026:
        # ── 2) 2026-Q1 forward ──
        df_q = get_or_build_2026q1(cfg, feature_cols)
        y_pred_q = model.predict(df_q[feature_cols])
        stats_q = backtest(
            df_q, y_pred_q,
            label="2026-Q1 forward test",
            notional=notional,
            margin_per_trade=args.margin,
            taker_fee=args.taker_fee,
            funding_rate=args.funding_rate,
        )
        print_report(stats_q)
        print()

    if args.save_trades is not None:
        frames = [stats_h["ledger"]]
        if stats_q is not None and not stats_q["ledger"].empty:
            frames.append(stats_q["ledger"])
        ledger_all = pd.concat(frames, ignore_index=True)
        args.save_trades.parent.mkdir(parents=True, exist_ok=True)
        ledger_all.to_csv(args.save_trades, index=False)
        print(f"Saved per-trade ledger → {args.save_trades}")


if __name__ == "__main__":
    main()
