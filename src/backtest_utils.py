"""
Backtest utilities shared by `run_top_trials.py` and the threshold-tuning
notebook.

Conventions:
  - Long-only book. y_pred == 1 means "open a long", everything else means
    "stay flat".
  - **Non-overlapping**: once a trade opens at row i (close_time[i]), every
    later signal is ignored until time has reached t1_time[i] (the bar at
    which the triple barrier resolves).
  - Trade outcome is the realized triple-barrier log return that the labeler
    already computed; we do not re-simulate the exit here.
  - Costs are a per-trade round-trip = 2 * (fee + slippage), applied in
    wealth space and scaled by the bet size.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

FEE_BPS_PER_SIDE_DEFAULT: float = 4.0
SLIPPAGE_BPS_PER_SIDE_DEFAULT: float = 1.0


def _empty_stats(rt_cost: float, n: int) -> dict:
    return {
        "n_signals_evaluated": int(n),
        "n_trades": 0,
        "round_trip_cost_bps": rt_cost * 10_000.0,
        "avg_bet_size": 0.0,
        "gross_pnl_net": 0.0,
        "net_pnl_after_fees": 0.0,
        "win_rate_net": float("nan"),
        "profit_factor_net": float("nan"),
        "avg_trade_log_net": float("nan"),
        "sharpe_trade_annualized": float("nan"),
        "max_drawdown_log_net": 0.0,
        "total_gross_log": 0.0,
        "total_net_log": 0.0,
        "trades_per_year": 0.0,
    }


def realistic_backtest(
    y_pred: np.ndarray | pd.Series,
    triple_barrier_log_ret: np.ndarray | pd.Series,
    close_times: np.ndarray | pd.Series,
    t1_times: np.ndarray | pd.Series,
    *,
    bet_size: np.ndarray | pd.Series | None = None,
    fee_bps_per_side: float = FEE_BPS_PER_SIDE_DEFAULT,
    slippage_bps_per_side: float = SLIPPAGE_BPS_PER_SIDE_DEFAULT,
) -> dict:
    """
    Non-overlapping triple-barrier backtest with fees + optional bet sizing.

    Wealth update per accepted trade:
        mult_i = 1 + size_i * ((exp(gross_log_i) - 1) - rt_cost)
    Total net PnL = prod(mult_i) - 1.
    """
    rt_cost = 2.0 * (fee_bps_per_side + slippage_bps_per_side) / 10_000.0

    y_pred = np.asarray(y_pred).astype(int).ravel()
    ret = np.asarray(triple_barrier_log_ret, dtype=float).ravel()
    ct = pd.to_datetime(np.asarray(close_times)).to_numpy()
    t1 = pd.to_datetime(np.asarray(t1_times)).to_numpy()

    n = len(y_pred)
    if not (len(ret) == len(ct) == len(t1) == n):
        raise ValueError("y_pred, ret, close_times, t1_times must be same length")

    if bet_size is None:
        size = np.ones(n, dtype=float)
    else:
        size = np.clip(np.asarray(bet_size, dtype=float).ravel(), 0.0, 1.0)
        if len(size) != n:
            raise ValueError("bet_size length mismatch")

    take = np.zeros(n, dtype=bool)
    active_until = np.datetime64("NaT")
    for i in range(n):
        if y_pred[i] != 1 or size[i] <= 0.0:
            continue
        if np.isnat(active_until) or ct[i] >= active_until:
            take[i] = True
            active_until = t1[i]

    n_trades = int(take.sum())
    if n_trades == 0:
        return _empty_stats(rt_cost, n)

    gross_log = ret[take]
    s = size[take]
    finite = np.isfinite(gross_log)
    gross_log = gross_log[finite]
    s = s[finite]
    if len(gross_log) == 0:
        return _empty_stats(rt_cost, n)

    simple = np.exp(gross_log) - 1.0
    mult_net = np.maximum(1.0 + s * (simple - rt_cost), 1e-9)
    mult_gross = np.maximum(1.0 + s * simple, 1e-9)
    net_log = np.log(mult_net)
    gross_only = np.log(mult_gross)

    gross_pnl = float(np.exp(np.nansum(gross_only)) - 1.0)
    net_pnl = float(np.exp(np.nansum(net_log)) - 1.0)

    wr = float((net_log > 0).mean())
    pos = float(net_log[net_log > 0].sum())
    neg = float(-net_log[net_log < 0].sum())
    pf = (pos / neg) if neg > 0 else (float("inf") if pos > 0 else float("nan"))
    avg = float(np.nanmean(net_log))

    if len(ct) >= 2:
        span_sec = (pd.Timestamp(ct[-1]) - pd.Timestamp(ct[0])).total_seconds()
        years = max(span_sec / (365.25 * 24 * 3600), 1e-9)
        tpy = float(n_trades / years)
    else:
        tpy = 0.0

    std = float(np.nanstd(net_log, ddof=1)) if len(net_log) > 1 else 0.0
    sharpe = (avg / std) * np.sqrt(tpy) if std > 0 else float("nan")

    equity = np.nancumsum(net_log)
    peak = np.maximum.accumulate(equity)
    mdd = float((peak - equity).max())

    return {
        "n_signals_evaluated": int(n),
        "n_trades": n_trades,
        "round_trip_cost_bps": rt_cost * 10_000.0,
        "avg_bet_size": float(np.mean(s)),
        "gross_pnl_net": gross_pnl,
        "net_pnl_after_fees": net_pnl,
        "win_rate_net": wr,
        "profit_factor_net": pf,
        "avg_trade_log_net": avg,
        "sharpe_trade_annualized": float(sharpe) if std > 0 else float("nan"),
        "max_drawdown_log_net": mdd,
        "total_gross_log": float(np.nansum(gross_only)),
        "total_net_log": float(np.nansum(net_log)),
        "trades_per_year": tpy,
    }


def afml_bet_size(
    proba_long: np.ndarray | pd.Series,
    n_classes: int = 2,
) -> np.ndarray:
    """
    AFML Ch.10 bet sizing, mapped to a long-only book.

    Standardised score:  m = (p - 1/K) / sqrt(p * (1 - p))
    Bet:                 b = 2 * Phi(m) - 1     # in [-1, 1]
    Long-only:           clip to [0, 1]         # negative scores -> sit out
    """
    p = np.clip(np.asarray(proba_long, dtype=float), 1e-6, 1.0 - 1e-6)
    m = (p - 1.0 / n_classes) / np.sqrt(p * (1.0 - p))
    bet = 2.0 * norm.cdf(m) - 1.0
    return np.clip(bet, 0.0, 1.0)


def threshold_sweep(
    proba_long: np.ndarray | pd.Series,
    y_true: np.ndarray | pd.Series,
    triple_barrier_log_ret: np.ndarray | pd.Series,
    close_times: np.ndarray | pd.Series,
    t1_times: np.ndarray | pd.Series,
    *,
    thresholds: np.ndarray | None = None,
    bet_size: np.ndarray | pd.Series | None = None,
    fee_bps_per_side: float = FEE_BPS_PER_SIDE_DEFAULT,
    slippage_bps_per_side: float = SLIPPAGE_BPS_PER_SIDE_DEFAULT,
) -> pd.DataFrame:
    """
    Sweep a probability threshold, run the non-overlapping backtest at each
    cut, and return a DataFrame keyed by threshold. Useful for the notebook.
    """
    if thresholds is None:
        thresholds = np.linspace(0.30, 0.70, 41)
    p = np.asarray(proba_long, dtype=float)
    y_t = np.asarray(y_true).astype(int)
    rows = []
    for th in thresholds:
        y_p = (p >= th).astype(int)
        bt = realistic_backtest(
            y_pred=y_p,
            triple_barrier_log_ret=triple_barrier_log_ret,
            close_times=close_times,
            t1_times=t1_times,
            bet_size=bet_size,
            fee_bps_per_side=fee_bps_per_side,
            slippage_bps_per_side=slippage_bps_per_side,
        )
        n_sig = int((y_p == 1).sum())
        if n_sig > 0:
            tp = int(((y_p == 1) & (y_t == 1)).sum())
            prec = tp / n_sig
        else:
            prec = float("nan")
        rows.append({
            "threshold": float(th),
            "n_signals": n_sig,
            "precision_long_at_signal": prec,
            **bt,
        })
    return pd.DataFrame(rows)
