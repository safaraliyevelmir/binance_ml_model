import glob
import os

import numba
import numpy as np
import pandas as pd

SYMBOL   = "SOLUSDT"

BARS_PATH = f"data/processed/dollar_bars_25_{SYMBOL}.parquet"
RAW_DIR  = "data/raw"
OUT_PATH = f"data/processed/labels_{SYMBOL}.parquet"

PT_SL = [1.2, 1]
SPAN = 100
MAX_HOLD = 400

def getDailyVol(close: pd.Series, span: int = 100):
    past_pos = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    past_pos = past_pos[past_pos > 0]
    curr_pos = np.arange(close.shape[0] - past_pos.shape[0], close.shape[0])
    df0_new = pd.Series(
        close.iloc[curr_pos].values / close.iloc[past_pos - 1].values - 1,
        index=close.index[curr_pos],
    )
    df0_new = df0_new.ewm(span=span).std()
    return df0_new


@numba.njit(cache=True)
def process_ticks_nb(
    tick_prices:    np.ndarray,
    tick_times:     np.ndarray,
    entry_times:    np.ndarray,
    vertical_times: np.ndarray,
    uppers:         np.ndarray,
    lowers:         np.ndarray,
    entry_prices:   np.ndarray,
    labels:         np.ndarray,
    log_returns:    np.ndarray,
    resolved:       np.ndarray,
    first_bar:      int,
    resolved_times: np.ndarray,
) -> int:
    n_bars = len(entry_times)

    for ti in range(len(tick_prices)):
        t = tick_times[ti]
        p = tick_prices[ti]

        bi = first_bar
        while bi < n_bars and entry_times[bi] <= t:
            if not resolved[bi]:
                if t >= vertical_times[bi]:
                    r = np.log(p / entry_prices[bi])
                    log_returns[bi] = r
                    labels[bi] = 0
                    resolved[bi] = True
                    resolved_times[bi] = t
                elif p >= uppers[bi]:
                    log_returns[bi] = np.log(p / entry_prices[bi])
                    labels[bi] = 1
                    resolved[bi] = True
                    resolved_times[bi] = t
                elif p <= lowers[bi]:
                    log_returns[bi] = np.log(p / entry_prices[bi])
                    labels[bi] = -1
                    resolved[bi] = True
                    resolved_times[bi] = t
            bi += 1

        # advance past resolved prefix
        while first_bar < n_bars and resolved[first_bar]:
            first_bar += 1

    return first_bar


def label_bars(
    bars: pd.DataFrame,
    vol: pd.Series,
    pt: float,
    sl: float,
    max_hold: int,
) -> pd.DataFrame:
    n = len(bars)
    labelable = n - max_hold

    close_times  = bars["close_time"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    closes = bars["close"].to_numpy(dtype=np.float64)
    vols_np = vol.to_numpy(dtype=np.float64)

    entry_times = close_times[:labelable]
    vertical_times = close_times[max_hold:]
    entry_prices = closes[:labelable]
    vols_arr = vols_np[:labelable]
    uppers = entry_prices * np.exp(pt * vols_arr)
    lowers = entry_prices * np.exp(-sl * vols_arr)

    labels = np.zeros(labelable, dtype=np.int8)
    log_returns = np.full(labelable, np.nan)
    resolved = np.zeros(labelable, dtype=np.bool_)
    resolved_times = np.zeros(labelable, dtype=np.int64)
    tick_files = sorted(glob.glob(f"{RAW_DIR}/{SYMBOL}_*.parquet"))

    first_bar = 0
    for f in tick_files:
        if first_bar >= labelable:
            break
        print(f"{os.path.basename(f)}  —  {labelable - first_bar:,} bars still open")
        ticks = (
            pd.read_parquet(f, columns=["timestamp", "price"])
            .sort_values("timestamp")
        )
        first_bar = process_ticks_nb(
            ticks["price"].to_numpy(dtype=np.float64),
            ticks["timestamp"].to_numpy(dtype="datetime64[ns]").astype(np.int64),
            entry_times, vertical_times,
            uppers, lowers, entry_prices,
            labels, log_returns, resolved,
            first_bar,resolved_times
        )
        del ticks

    unresolved = int((~resolved).sum())
    if unresolved:
        print(f"Warning: {unresolved} bars unresolved — tick data may not cover full period")

    out = pd.DataFrame({
        "open_time": bars["open_time"].values[:labelable],
        "close_time": bars["close_time"].values[:labelable],
        "close": entry_prices,
        "vol": vols_arr,
        "ofi": bars["ofi"].values[:labelable],
        "label": labels,
        "return": log_returns,
        "t1_time": pd.to_datetime(resolved_times, unit="ns"),
    })
    return out


if __name__ == "__main__":
    print(f"Loading dollar bars from {BARS_PATH}…")
    bars = pd.read_parquet(BARS_PATH)
    print(f"  {len(bars):,} bars")

    print(f"Computing daily volatility (span={SPAN})…")
    close_ts = bars["close"].copy()
    close_ts.index = pd.to_datetime(bars.close_time)
    vol = getDailyVol(close_ts, span=SPAN)

    pt, sl = PT_SL
    print(f"Labeling with PT={pt}×σ  SL={sl}×σ  MAX_HOLD={MAX_HOLD} bars…")
    labels_df = label_bars(bars, vol, pt=pt, sl=sl, max_hold=MAX_HOLD)

    labels_df.to_parquet(OUT_PATH)
    dist = labels_df["label"].value_counts().sort_index().to_dict()
    print(f"\nSaved {len(labels_df):,} labelled bars → {OUT_PATH}")
    print(f"Label distribution: {dist}")

