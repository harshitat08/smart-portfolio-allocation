import numpy as np
import pandas as pd
from scipy.optimize import minimize, Bounds
import logging

logger = logging.getLogger(__name__)

ANNUALIZATION_FACTOR = 252
RISK_FREE_RATE = 0.045


def portfolio_return(weights, mu):
    return float(weights @ mu)


def portfolio_volatility(weights, cov):
    return float(np.sqrt(weights @ cov @ weights))


def sharpe_ratio(weights, mu, cov, rf=RISK_FREE_RATE):
    return (portfolio_return(weights, mu) - rf) / (portfolio_volatility(weights, cov) + 1e-12)


def compute_covariance(returns, method="ledoit_wolf"):
    if method == "ledoit_wolf":
        try:
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf()
            lw.fit(returns.values)
            return pd.DataFrame(
                lw.covariance_ * ANNUALIZATION_FACTOR,
                index=returns.columns, columns=returns.columns,
            )
        except Exception as e:
            logger.warning(f"Ledoit-Wolf failed ({e}), falling back to sample.")
    return returns.cov() * ANNUALIZATION_FACTOR


def compute_expected_returns(returns, ml_predictions=None, ml_blend=0.5):
    hist_mu = returns.mean() * ANNUALIZATION_FACTOR
    if ml_predictions is None or ml_blend == 0:
        return hist_mu
    ml_mu = ml_predictions * (ANNUALIZATION_FACTOR / 5)
    ml_mu = ml_mu.reindex(hist_mu.index).fillna(hist_mu)
    return (1 - ml_blend) * hist_mu + ml_blend * ml_mu


def optimize_portfolio(mu, cov, objective="sharpe", max_weight=0.40, rf=RISK_FREE_RATE):
    n = len(mu)
    mu_arr = mu.values
    cov_arr = cov.values
    w0 = np.full(n, 1.0 / n)

    if objective == "sharpe":
        obj_fn = lambda w: -sharpe_ratio(w, mu_arr, cov_arr, rf)
    elif objective == "minvol":
        obj_fn = lambda w: portfolio_volatility(w, cov_arr)
    elif objective == "maxret":
        obj_fn = lambda w: -portfolio_return(w, mu_arr)
    else:
        raise ValueError(f"Unknown objective: '{objective}'")

    result = minimize(
        obj_fn, x0=w0, method="SLSQP",
        bounds=Bounds(lb=0.0, ub=max_weight),
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        options={"ftol": 1e-9, "maxiter": 1000},
    )

    if not result.success:
        logger.warning(f"Optimiser warning: {result.message}. Falling back to equal weight.")
        return pd.Series(w0, index=mu.index)

    weights = pd.Series(np.clip(result.x, 0, max_weight), index=mu.index)
    return weights / weights.sum()


def compute_efficient_frontier(mu, cov, n_points=50, max_weight=0.40):
    n = len(mu)
    mu_arr = mu.values
    cov_arr = cov.values
    frontier = []

    for target in np.linspace(mu_arr.min() * 0.9, mu_arr.max() * 1.1, n_points):
        result = minimize(
            lambda w: portfolio_volatility(w, cov_arr),
            x0=np.full(n, 1 / n), method="SLSQP",
            bounds=Bounds(lb=0.0, ub=max_weight),
            constraints=[
                {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
                {"type": "eq", "fun": lambda w, t=target: portfolio_return(w, mu_arr) - t},
            ],
            options={"ftol": 1e-9, "maxiter": 500},
        )
        if result.success:
            vol = portfolio_volatility(result.x, cov_arr)
            frontier.append({"expected_return": target, "volatility": vol,
                             "sharpe": (target - RISK_FREE_RATE) / (vol + 1e-12)})

    return pd.DataFrame(frontier)


def portfolio_stats(weights, mu, cov):
    w = weights.values
    return {
        "expected_return": portfolio_return(w, mu.values),
        "volatility": portfolio_volatility(w, cov.values),
        "sharpe_ratio": sharpe_ratio(w, mu.values, cov.values),
    }