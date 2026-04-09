import yfinance as yf
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def fetch_price_data(tickers, start_date, end_date, price_col="Adj Close"):
    logger.info(f"Fetching data for {tickers} from {start_date} to {end_date}")
    raw = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw[price_col]
    else:
        prices = raw[[price_col]].rename(columns={price_col: tickers[0]})

    prices = prices.dropna(how="all").ffill().dropna(axis=1)
    logger.info(f"Retrieved {len(prices)} trading days for {list(prices.columns)}")
    return prices


def compute_returns(prices, method="log"):
    if method == "log":
        return np.log(prices / prices.shift(1)).dropna()
    return prices.pct_change().dropna()


def get_benchmark_data(benchmark_ticker, start_date, end_date):
    prices = fetch_price_data([benchmark_ticker], start_date, end_date)
    returns = compute_returns(prices).iloc[:, 0].rename(benchmark_ticker)
    return prices, returns


def validate_data(prices, min_obs=252):
    if len(prices) < min_obs:
        logger.warning(f"Only {len(prices)} observations. Minimum recommended: {min_obs}")
        return False
    return True