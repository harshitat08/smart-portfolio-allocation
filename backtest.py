import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

ANNUALIZATION_FACTOR = 252
FREQ_MAP = {"M": "ME", "W": "W", "Q": "QE"}


def compute_performance_metrics(portfolio_values, rf=0.045, label="Portfolio"):
    daily_returns = portfolio_values.pct_change().dropna()
    n_days = len(portfolio_values)
    n_years = n_days / ANNUALIZATION_FACTOR
    total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
    cagr = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1
    ann_vol = daily_returns.std() * np.sqrt(ANNUALIZATION_FACTOR)
    daily_rf = (1 + rf) ** (1 / ANNUALIZATION_FACTOR) - 1
    excess = daily_returns - daily_rf
    sharpe = (excess.mean() / (daily_returns.std() + 1e-12)) * np.sqrt(ANNUALIZATION_FACTOR)
    downside = daily_returns[daily_returns < 0]
    sortino = (excess.mean() / (downside.std() + 1e-12)) * np.sqrt(ANNUALIZATION_FACTOR)
    cumulative = (1 + daily_returns).cumprod()
    drawdowns = (cumulative - cumulative.cummax()) / cumulative.cummax()
    max_drawdown = drawdowns.min()

    return {
        "label": label,
        "total_return": total_return,
        "cagr": cagr,
        "ann_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_drawdown,
        "calmar_ratio": cagr / (abs(max_drawdown) + 1e-12),
        "win_rate": (daily_returns > 0).mean(),
    }


def compute_drawdown_series(portfolio_values):
    daily_rets = portfolio_values.pct_change().fillna(0)
    cumulative = (1 + daily_rets).cumprod()
    return (cumulative - cumulative.cummax()) / cumulative.cummax()


def equal_weight_portfolio(returns, initial_value=100_000.0, transaction_cost=0.001,
                            rebalance_frequency="M"):
    rebalance_frequency = FREQ_MAP.get(rebalance_frequency, rebalance_frequency)
    simple_returns = np.exp(returns) - 1
    n = simple_returns.shape[1]
    equal_w = np.full(n, 1.0 / n)
    nav = initial_value
    nav_series = {}
    current_weights = equal_w.copy()
    rebal_dates = simple_returns.resample(rebalance_frequency).last().index
    prev_rebal = simple_returns.index[0]

    for rebal_date in rebal_dates:
        actual_rb = simple_returns.index[simple_returns.index.get_indexer([rebal_date], method="ffill")[0]]
        period = simple_returns.loc[prev_rebal:actual_rb]
        for date, row in period.iterrows():
            nav *= (1 + float(current_weights @ row.values))
            nav_series[date] = nav
        turnover = np.abs(equal_w - current_weights).sum()
        nav *= (1 - turnover * transaction_cost)
        current_weights = equal_w.copy()
        prev_rebal = actual_rb

    return pd.Series(nav_series).sort_index()


def default_optimize_fn(returns_window, ml_preds=None, max_weight=0.40,
                        ml_blend=0.4, objective="sharpe"):
    from optimizer import compute_expected_returns, compute_covariance, optimize_portfolio
    mu = compute_expected_returns(returns_window, ml_preds, ml_blend=ml_blend)
    cov = compute_covariance(returns_window, method="ledoit_wolf")
    return optimize_portfolio(mu, cov, objective=objective, max_weight=max_weight)


class Backtester:
    def __init__(self, prices, returns, optimize_fn, rebalance_frequency="M",
                 lookback_window=252, initial_value=100_000.0, transaction_cost=0.001,
                 max_weight=0.40, use_ml=True, model_type="random_forest", ml_blend=0.4):
        self.rebalance_frequency = FREQ_MAP.get(rebalance_frequency, rebalance_frequency)
        self.prices = prices
        self.returns = returns
        self.optimize_fn = optimize_fn
        self.lookback_window = lookback_window
        self.initial_value = initial_value
        self.transaction_cost = transaction_cost
        self.max_weight = max_weight
        self.use_ml = use_ml
        self.model_type = model_type
        self.ml_blend = ml_blend
        self.tickers = list(returns.columns)

    def run(self):
        from features import build_all_features
        from model import train_all_models, get_predicted_returns

        simple_returns = np.exp(self.returns) - 1
        rebal_dates = (simple_returns.iloc[self.lookback_window:]
                       .resample(self.rebalance_frequency).last().index)

        nav = self.initial_value
        current_weights = np.full(len(self.tickers), 1.0 / len(self.tickers))
        nav_series = {}
        weights_history = {}
        turnover_history = {}
        prev_rebal = simple_returns.index[self.lookback_window]

        for rebal_date in rebal_dates:
            actual_rb = simple_returns.index[
                simple_returns.index.get_indexer([rebal_date], method="ffill")[0]
            ]
            period_returns = simple_returns.loc[prev_rebal:actual_rb]
            if len(period_returns) < 2:
                continue

            for date, row in period_returns.iterrows():
                nav *= (1 + float(current_weights @ row.values))
                nav_series[date] = nav

            window_idx = simple_returns.index.get_loc(actual_rb)
            start_idx = max(0, window_idx - self.lookback_window)
            returns_window = self.returns.iloc[start_idx:window_idx]

            if len(returns_window) < 60:
                prev_rebal = actual_rb
                continue

            ml_preds = None
            if self.use_ml:
                try:
                    prices_window = self.prices.iloc[start_idx:window_idx]
                    features = build_all_features(prices_window, returns_window, forward_return_window=5)
                    valid = {t: f for t, f in features.items() if len(f) >= 80}
                    if valid:
                        model_res = train_all_models(valid, model_type=self.model_type)
                        ml_preds = get_predicted_returns(model_res)
                except Exception as e:
                    logger.warning(f"ML step failed at {actual_rb}: {e}")

            new_weights = self.optimize_fn(
                returns_window=returns_window, ml_preds=ml_preds,
                max_weight=self.max_weight, ml_blend=self.ml_blend,
            )
            new_weights = new_weights.reindex(self.tickers).fillna(0).values
            turnover = np.sum(np.abs(new_weights - current_weights))
            nav *= (1 - turnover * self.transaction_cost)
            turnover_history[actual_rb] = turnover
            weights_history[actual_rb] = dict(zip(self.tickers, new_weights))
            current_weights = new_weights
            prev_rebal = actual_rb

        portfolio_values = pd.Series(nav_series).sort_index()
        weights_df = pd.DataFrame(weights_history).T
        metrics = compute_performance_metrics(portfolio_values, label="Smart Portfolio")
        metrics["avg_turnover"] = np.mean(list(turnover_history.values())) if turnover_history else 0

        return {
            "portfolio_values": portfolio_values,
            "weights_history": weights_df,
            "turnover_history": pd.Series(turnover_history),
            "metrics": metrics,
        }