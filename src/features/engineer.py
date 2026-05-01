import numba
import numpy as np
import pandas as pd
import pandas_ta as ta
from numpy.lib.stride_tricks import sliding_window_view
from statsmodels.tsa.stattools import adfuller

HALFLIFE_DAYS = 30
DURATION_WINDOW = 100
FRAC_D_STEP = 0.05
FRAC_THRES = 1e-5
FRAC_PVAL_MAX = 0.05
FRAC_CORR_MIN = 0.80


def ffd_weights(d: float, thres: float = FRAC_THRES) -> np.ndarray:
    w, k = [1.0], 1
    while True:
        w_next = -w[-1] * (d - k + 1) / k
        if abs(w_next) < thres:
            break
        w.append(w_next)
        k += 1
    return np.array(w[::-1])


def frac_diff_ffd(
    series: pd.Series,
    d: float,
    thres: float = FRAC_THRES,
) -> pd.Series:
    w = ffd_weights(d, thres)
    width = len(w)
    arr = series.ffill().to_numpy(dtype=np.float64)
    windows = sliding_window_view(arr, width)
    out = np.full(len(arr), np.nan)
    out[width - 1 :] = windows @ w
    return pd.Series(out, index=series.index, name="frac_diff")


def find_min_d(
    close: pd.Series,
    d_step: float = FRAC_D_STEP,
    thres: float = FRAC_THRES,
    pval_max: float = FRAC_PVAL_MAX,
    corr_min: float = FRAC_CORR_MIN,
) -> tuple[float, pd.Series]:
    log_price = pd.Series(np.log(close.to_numpy(dtype=np.float64)), index=close.index)
    best_d, best_fd = 1.0, frac_diff_ffd(log_price, 1.0, thres)

    for d_raw in np.arange(0.0, 1.0 + d_step / 2, d_step):
        d = round(float(d_raw), 4)
        fd = frac_diff_ffd(log_price, d, thres)
        fd_clean = fd.dropna()
        if len(fd_clean) < 30:
            continue
        try:
            pval = adfuller(fd_clean.to_numpy(), maxlag=1, regression="c", autolag=None)[1]
        except Exception:
            continue
        corr = float(fd_clean.corr(log_price.reindex(fd_clean.index)))
        if pval <= pval_max and corr >= corr_min:
            best_d, best_fd = d, fd
            break

    return best_d, best_fd


@numba.njit(cache=True)
def ewm_std_time_nb(
    returns: np.ndarray,
    dt_ns: np.ndarray,
    halflife_ns: float,
) -> np.ndarray:
    n = len(returns)
    out = np.full(n, np.nan)
    lam = np.log(2.0) / halflife_ns
    mean = 0.0
    var_ = 0.0
    init = False
    for i in range(n):
        r = returns[i]
        if np.isnan(r):
            continue
        if not init:
            mean = r
            init = True
            out[i] = np.nan
            continue
        dt = dt_ns[i]
        alpha = 1.0 - np.exp(-lam * dt)
        delta = r - mean
        mean = mean + alpha * delta
        var_ = (1.0 - alpha) * (var_ + alpha * delta * delta)
        out[i] = np.sqrt(var_)
    return out


def dynamic_sigma(
    bars: pd.DataFrame,
    halflife_days: float = HALFLIFE_DAYS,
) -> pd.Series:
    log_ret = np.log(
        bars["close"].to_numpy(dtype=np.float64)
        / np.concatenate([[np.nan], bars["close"].to_numpy(dtype=np.float64)[:-1]])
    )
    close_times_ns = bars["close_time"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    dt_ns = np.concatenate([[0], np.diff(close_times_ns)]).astype(np.float64)

    halflife_ns = float(halflife_days * 24 * 3600 * 1e9)
    sigma_vals = ewm_std_time_nb(log_ret, dt_ns, halflife_ns)
    return pd.Series(sigma_vals, index=bars.index, name="sigma")


def relative_duration(
    bars: pd.DataFrame,
    window: int = DURATION_WINDOW,
) -> pd.Series:
    dur = (bars["close_time"] - bars["open_time"]).dt.total_seconds()
    rel = dur / dur.rolling(window, min_periods=1).mean()
    return rel.rename("rel_duration")



def dollar_volume_imbalance(bars: pd.DataFrame) -> pd.Series:
    return bars["ofi"].rename("dvi")

def log_returns(bars: pd.DataFrame) -> pd.Series:
    return np.log(bars["close"] / bars["close"].shift(1)).rename("log_return")

def bollinger_bands(
    bars: pd.DataFrame,
    period: int = 20,
    n_std: float = 2.0,
) -> tuple[pd.Series, pd.Series]:
    bb = ta.bbands(bars["close"], length=period, std=n_std)
    upper = bb[[c for c in bb.columns if c.startswith("BBU_")][0]]
    lower = bb[[c for c in bb.columns if c.startswith("BBL_")][0]]
    mid = bb[[c for c in bb.columns if c.startswith("BBM_")][0]]
    denom = (upper - lower).replace(0.0, np.nan)
    pct_b = ((bars["close"] - lower) / denom).rename("bb_pct_b")
    width = ((upper - lower) / mid.replace(0.0, np.nan)).rename("bb_width")
    return pct_b, width


def rsi_feature(bars: pd.DataFrame, period: int = 14) -> pd.Series:
    return ta.rsi(bars["close"], length=period).rename("rsi")


def macd_features(
    bars: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    m = ta.macd(bars["close"], fast=fast, slow=slow, signal=signal)
    norm = bars["close"].replace(0.0, np.nan)
    line = (m[f"MACD_{fast}_{slow}_{signal}"] / norm).rename("macd")
    sig = (m[f"MACDs_{fast}_{slow}_{signal}"] / norm).rename("macd_signal")
    hist = (m[f"MACDh_{fast}_{slow}_{signal}"] / norm).rename("macd_hist")
    return line, sig, hist


def adx_features(
    bars: pd.DataFrame,
    period: int = 14,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    adx = ta.adx(bars["high"], bars["low"], bars["close"], length=period)
    adx_val = adx[f"ADX_{period}"].rename("adx")
    plus_di = adx[f"DMP_{period}"].rename("adx_pdi")
    minus_di = adx[f"DMN_{period}"].rename("adx_mdi")
    return adx_val, plus_di, minus_di


def stochastic_rsi(
    bars: pd.DataFrame,
    rsi_period: int = 14,
    stoch_period: int = 14,
    k_smooth: int = 3,
    d_smooth: int = 3,
) -> tuple[pd.Series, pd.Series]:
    stochrsi = ta.stochrsi(
        bars["close"],
        length=rsi_period,
        rsi_length=stoch_period,
        k=k_smooth,
        d=d_smooth,
    )
    k = stochrsi[f"STOCHRSIk_{rsi_period}_{stoch_period}_{k_smooth}_{d_smooth}"].rename("stoch_rsi_k")
    d = stochrsi[f"STOCHRSId_{rsi_period}_{stoch_period}_{k_smooth}_{d_smooth}"].rename("stoch_rsi_d")
    return k, d



def build_features(
    bars_path: str,
    labels_path: str,
    out_path: str,
) -> pd.DataFrame:
    bars = pd.read_parquet(bars_path)
    labels = pd.read_parquet(labels_path)
    print(f"Bars: {len(bars)} Label: {len(labels):,}")

    print("log-returns")
    lr = log_returns(bars)

    print(f"dynamic sigma halflife={HALFLIFE_DAYS}")
    sigma = dynamic_sigma(bars)

    print(f"relative bar duration window={DURATION_WINDOW}")
    rel_dur = relative_duration(bars)

    print("DVI (from ofi)")
    dvi = dollar_volume_imbalance(bars)

    print(f"FracDiff d_step={FRAC_D_STEP}, corr_min={FRAC_CORR_MIN}")
    best_d, fd = find_min_d(bars["close"])
    print(f"min d = {best_d}")

    print("Bollinger Bands (period=20)")
    bb_pct_b, bb_width = bollinger_bands(bars)

    print("RSI (period=14)")
    rsi_val = rsi_feature(bars)

    print("MACD (12/26/9)")
    macd_line, macd_sig, macd_hist = macd_features(bars)

    print("ADX (period=14)")
    adx_val, adx_pdi, adx_mdi = adx_features(bars)

    print("Stochastic RSI (14/14/3/3)")
    stoch_k, stoch_d = stochastic_rsi(bars)

    feats = pd.DataFrame({
        "open_time": bars["open_time"],
        "close_time": bars["close_time"],
        "t1_time": labels["t1_time"],
        "close": bars["close"],
        # original features
        "log_return": lr,
        "sigma": sigma,
        "rel_duration": rel_dur,
        "dvi": dvi,
        "frac_diff": fd,
        # technical indicators
        "bb_pct_b": bb_pct_b,
        "bb_width": bb_width,
        "rsi": rsi_val,
        "macd": macd_line,
        "macd_signal": macd_sig,
        "macd_hist": macd_hist,
        "adx": adx_val,
        "adx_pdi": adx_pdi,
        "adx_mdi": adx_mdi,
        "stoch_rsi_k": stoch_k,
        "stoch_rsi_d": stoch_d,
    })

    feats = feats.loc[labels.index].copy()
    feats["label"] = labels["label"].values
    feats["ret"] = labels["return"].values

    feats.to_parquet(out_path)
    print(f"Saved {len(feats)} rows → {out_path}")
    return feats


if __name__ == "__main__":
    SYMBOL = "SOLUSDT"
    for dollar_thresholds in [5,10,15,20,25,50]:
        bar_path = f"data/processed/dollar_bars_{dollar_thresholds}_{SYMBOL}.parquet"
        label_path = f"data/processed/labels_{dollar_thresholds}_{SYMBOL}.parquet"
        out_path = f"data/processed/features_{dollar_thresholds}_{SYMBOL}.parquet"

        df = build_features(bar_path, label_path, out_path)
        dist = df["label"].value_counts().sort_index()
        print(dist)