import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def compute_rsi(prices, window=14):
    delta = prices.diff()
    gain = delta.clip(lower=0).ewm(com=window - 1, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).ewm(com=window - 1, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).rename("RSI")


def compute_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({
        "MACD": macd_line,
        "MACD_Signal": signal_line,
        "MACD_Hist": macd_line - signal_line,
    })


def build_features_for_ticker(prices, returns, vol_windows=[5, 21], ma_windows=[10, 50],
                               momentum_windows=[5, 21], lag_periods=[1, 2, 3, 5],
                               forward_return_window=5):
    df = pd.DataFrame(index=prices.index)
    df["return"] = returns

    for w in vol_windows:
        df[f"vol_{w}d"] = returns.rolling(w).std() * np.sqrt(252)

    mas = {}
    for w in ma_windows:
        mas[w] = prices.rolling(w).mean()
        df[f"ma_{w}d"] = prices / mas[w] - 1

    if len(ma_windows) >= 2:
        short_w, long_w = sorted(ma_windows)[:2]
        df["ma_crossover"] = mas[short_w] / mas[long_w] - 1

    for w in momentum_windows:
        df[f"momentum_{w}d"] = returns.rolling(w).sum()

    df["rsi_14"] = compute_rsi(prices, window=14)
    df["rsi_norm"] = df["rsi_14"] / 100.0

    macd_df = compute_macd(prices)
    df["macd_hist"] = macd_df["MACD_Hist"]

    for lag in lag_periods:
        df[f"return_lag_{lag}"] = returns.shift(lag)

    if len(vol_windows) >= 2:
        short_v, long_v = sorted(vol_windows)[:2]
        df["vol_ratio"] = df[f"vol_{short_v}d"] / (df[f"vol_{long_v}d"] + 1e-10)

    df["y_forward_return"] = returns.shift(-forward_return_window).rolling(forward_return_window).sum()

    return df


def build_all_features(prices, returns, forward_return_window=5):
    feature_dict = {}
    for ticker in prices.columns:
        logger.info(f"Building features for {ticker}")
        df = build_features_for_ticker(
            prices=prices[ticker],
            returns=returns[ticker],
            forward_return_window=forward_return_window,
        )
        feature_dict[ticker] = df.dropna()
    return feature_dict