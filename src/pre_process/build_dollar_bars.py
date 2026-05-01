# pip install numba
import glob
import os

import numba
import numpy as np
import pandas as pd

SYMBOL = "SOLUSDT"
THRESHOLD = 25_000_000  # $25M per bar
RAW_DIR = "data/raw"
OUT_PATH = f"data/processed/dollar_bars_25_{SYMBOL}.parquet"


@numba.njit(cache=True)
def process_chunk_nb(
    prices: np.ndarray,
    quantities: np.ndarray,
    timestamps: np.ndarray,
    is_buy: np.ndarray,
    threshold: float,
    # state_f: [open_p, high_p, low_p, cum_vol, cum_dollar, buy_vol]
    state_f: np.ndarray,
    # state_i: [open_t, trades]
    state_i: np.ndarray,
) -> tuple:
    n = len(prices)
    out_open_t  = np.empty(n, dtype=np.int64)
    out_close_t = np.empty(n, dtype=np.int64)
    out_open    = np.empty(n, dtype=np.float64)
    out_high    = np.empty(n, dtype=np.float64)
    out_low     = np.empty(n, dtype=np.float64)
    out_close   = np.empty(n, dtype=np.float64)
    out_vol     = np.empty(n, dtype=np.float64)
    out_dvol    = np.empty(n, dtype=np.float64)
    out_trades  = np.empty(n, dtype=np.int64)
    out_buy_vol = np.empty(n, dtype=np.float64)
    out_sell_vol= np.empty(n, dtype=np.float64)
    out_ofi     = np.empty(n, dtype=np.float64)

    count = 0

    for i in range(n):
        p = prices[i]
        q = quantities[i]

        if state_f[4] == 0.0:  # new bar
            state_f[0] = p     # open_p
            state_f[1] = p     # high_p
            state_f[2] = p     # low_p
            state_i[0] = timestamps[i]

        if p > state_f[1]:
            state_f[1] = p
        if p < state_f[2]:
            state_f[2] = p

        state_f[3] += q        # cum_vol
        state_f[4] += p * q    # cum_dollar
        state_i[1] += 1        # trades
        if is_buy[i]:
            state_f[5] += q    # buy_vol

        if state_f[4] >= threshold:
            sell_vol = state_f[3] - state_f[5]
            out_open_t[count]   = state_i[0]
            out_close_t[count]  = timestamps[i]
            out_open[count]     = state_f[0]
            out_high[count]     = state_f[1]
            out_low[count]      = state_f[2]
            out_close[count]    = p
            out_vol[count]      = state_f[3]
            out_dvol[count]     = state_f[4]
            out_trades[count]   = state_i[1]
            out_buy_vol[count]  = state_f[5]
            out_sell_vol[count] = sell_vol
            out_ofi[count]      = (state_f[5] - sell_vol) / state_f[3]
            count += 1

            state_f[3] = 0.0
            state_f[4] = 0.0
            state_f[5] = 0.0
            state_i[1] = 0

    return (count, out_open_t, out_close_t, out_open, out_high, out_low, out_close,
            out_vol, out_dvol, out_trades, out_buy_vol, out_sell_vol, out_ofi)


def chunk_to_df(result: tuple) -> pd.DataFrame:
    count, open_t, close_t, opens, highs, lows, closes, vols, dvols, trades, buy_vols, sell_vols, ofis = result
    if count == 0:
        return pd.DataFrame()
    return pd.DataFrame({
        "open_time":    pd.to_datetime(open_t[:count]),
        "close_time":   pd.to_datetime(close_t[:count]),
        "open":         opens[:count],
        "high":         highs[:count],
        "low":          lows[:count],
        "close":        closes[:count],
        "volume":       vols[:count],
        "dollar_volume":dvols[:count],
        "trades":       trades[:count],
        "buy_volume":   buy_vols[:count],
        "sell_volume":  sell_vols[:count],
        "ofi":          ofis[:count],
    })


def build_dollar_bars_streaming(files: list[str], threshold: float) -> pd.DataFrame:
    state_f = np.zeros(6, dtype=np.float64)  # [open_p, high_p, low_p, cum_vol, cum_dollar, buy_vol]
    state_i = np.zeros(2, dtype=np.int64)    # [open_t, trades]

    all_frames = []

    for f in files:
        df = pd.read_parquet(f).sort_values("timestamp")
        result = process_chunk_nb(
            df["price"].to_numpy(dtype=np.float64),
            df["quantity"].to_numpy(dtype=np.float64),
            df["timestamp"].to_numpy(dtype="datetime64[ns]").astype(np.int64),
            (~df["is_buyer_maker"].astype(bool)).to_numpy(),
            threshold,
            state_f,
            state_i,
        )
        del df
        frame = chunk_to_df(result)
        if not frame.empty:
            all_frames.append(frame)

    return pd.concat(all_frames, ignore_index=True)


if __name__ == "__main__":
    files = sorted(glob.glob(f"{RAW_DIR}/{SYMBOL}_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {RAW_DIR}/ for {SYMBOL}.")
    print(f"Found {len(files)} monthly files.")
    print(f"Threshold: ${THRESHOLD/1e6:.0f}M per bar")

    print("Building dollar bars (streaming + Numba)...")
    bars = build_dollar_bars_streaming(files, THRESHOLD)

    bars.to_parquet(OUT_PATH)
    print(f"\n{len(bars):,} bars saved to {OUT_PATH}")
    print(bars.head(10).to_string())
