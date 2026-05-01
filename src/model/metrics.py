import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

BARS_YEAR = 38_000


def strategy_log_returns(
    y_pred: np.ndarray,
    bar_ret: np.ndarray,
) -> pd.Series:
    return pd.Series(y_pred.astype(float) * bar_ret, name="strategy_ret")


def sharpe_ratio(strat_ret: pd.Series, periods_per_year: int = BARS_YEAR) -> float:
    mu  = strat_ret.mean()
    sig = strat_ret.std(ddof=1)
    if sig == 0:
        return np.nan
    return (mu / sig) * np.sqrt(periods_per_year)


def max_drawdown(strat_ret: pd.Series) -> float:
    cum   = strat_ret.cumsum()
    peak  = cum.cummax()
    dd    = peak - cum
    return float(dd.max())


def profit_factor(strat_ret: pd.Series) -> float:
    pos_ret   = strat_ret[strat_ret > 0].sum()
    neg_ret   = strat_ret[strat_ret < 0].abs().sum()
    return float(pos_ret / neg_ret) if neg_ret > 0 else np.inf


def win_rate(strat_ret: pd.Series) -> float:
    active = strat_ret[strat_ret != 0]
    if len(active) == 0:
        return np.nan
    return float((active > 0).mean())


def full_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bar_ret: np.ndarray,
    periods_per_year: int = BARS_YEAR,
) -> dict:
    strat_ret = strategy_log_returns(y_pred, bar_ret)

    sr   = sharpe_ratio(strat_ret, periods_per_year)
    mdd  = max_drawdown(strat_ret)
    pf   = profit_factor(strat_ret)
    wr   = win_rate(strat_ret)
    total_return = float(strat_ret.sum())
    active_pct   = float((y_pred != 0).mean())

    print("Classification:")
    print(classification_report(
        y_true, y_pred,
        labels=[-1, 0, 1],
        target_names=["short (-1)", "hold (0)", "long (+1)"],
        zero_division=0,
    ))

    print("Strategy:")
    print(f"Sharpe ratio (annualised) : {sr:>10.4f}")
    print(f"Maximum drawdown: {mdd:>10.4f} ({mdd*100:.1f}%)")
    print(f"Profit factor: {pf:>10.4f}")
    print(f"Win rate (active bars): {wr:>10.4f} ({wr*100:.1f}%)")
    print(f"Total log-return: {total_return:>10.4f} ({total_return*100:.2f}%)")
    print(f"Active bars: {active_pct:>10.4f} ({active_pct*100:.1f}% of all bars)")

    return {
        "sharpe":        sr,
        "max_drawdown":  mdd,
        "profit_factor": pf,
        "win_rate":      wr,
        "total_return":  total_return,
        "active_pct":    active_pct,
    }


def plot_equity_curve(strat_ret: pd.Series, title: str = "Strategy Equity Curve") -> None:
    cum = strat_ret.cumsum()
    peak = cum.cummax()

    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    ax1.plot(cum.values, linewidth=0.8, label="Cumulative log-return")
    ax1.plot(peak.values, linewidth=0.5, linestyle="--", color="grey", alpha=0.6)
    ax1.fill_between(range(len(cum)), cum.values, peak.values,
                     alpha=0.25, color="red", label="Drawdown")
    ax1.set_ylabel("Cumulative log-return")
    ax1.set_title(title)
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    dd = (peak - cum)
    ax2.fill_between(range(len(dd)), dd.values, alpha=0.6, color="red")
    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Bar index")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("data/processed/equity_curve_SOLUSDT.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    oos = pd.read_parquet("data/processed/oos_preds_SOLUSDT.parquet")
    print(f"OOS predictions:")

    metrics = full_report(
        y_true  = oos["y_true"].to_numpy(),
        y_pred  = oos["y_pred"].to_numpy(),
        bar_ret = oos["log_return"].to_numpy(),
    )

    strat_ret = strategy_log_returns(oos["y_pred"].to_numpy(), oos["log_return"].to_numpy())
    plot_equity_curve(strat_ret)
