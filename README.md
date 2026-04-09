# Smart Portfolio Allocation System

An ML-enhanced quantitative portfolio system that combines machine learning return prediction with Markowitz mean-variance optimization to intelligently allocate weights across a basket of equities.

## Results (2020–2026 Backtest)

| Metric | Smart Portfolio | Equal Weight | SPY |
|---|---|---|---|
| Total Return | 146.8% | 318.7% | 123.7% |
| CAGR | 18.81% | 25.79% | 13.78% |
| Sharpe Ratio | 0.684 | 0.848 | 0.518 |
| Max Drawdown | -37.5% | -35.2% | -33.7% |
| Sortino Ratio | 0.969 | 1.096 | 0.639 |

Smart Portfolio outperformed SPY by ~5% CAGR with a Sharpe improvement of 32% (0.684 vs 0.518).

## How It Works

1. **Data** — Fetches adjusted closing prices via yfinance
2. **Feature Engineering** — Computes RSI, MACD, rolling volatility, momentum, moving average crossovers, and lag returns for each asset
3. **ML Prediction** — Trains a per-asset Random Forest model to predict 5-day forward returns using walk-forward validation (no lookahead bias)
4. **Return Estimation** — Blends ML predictions (40%) with historical mean returns (60%) to produce expected returns
5. **Covariance Estimation** — Uses Ledoit-Wolf shrinkage for a more stable covariance matrix
6. **Optimization** — Maximizes Sharpe ratio via scipy SLSQP, constrained to long-only with a 35% per-asset cap
7. **Backtesting** — Monthly rebalancing with 10bps transaction costs, walk-forward to prevent lookahead bias
8. **Benchmarking** — Compares against equal-weight portfolio and SPY

## Project Structure
├── data.py          # Data fetching via yfinance
├── features.py      # Technical indicator engineering
├── model.py         # ML return prediction (Random Forest)
├── optimizer.py     # Markowitz MVO via scipy
├── backtest.py      # Walk-forward simulation
├── main.py          # CLI entry point
└── requirements.txt

## Usage

```bash
pip install -r requirements.txt

# Default run
python main.py

# Custom
python main.py --risk aggressive --rebalance W
python main.py --tickers AAPL MSFT NVDA TSLA --start 2021-01-01
python main.py --no-ml --objective minvol
```

## CLI Options

| Flag | Options | Default |
|---|---|---|
| `--risk` | aggressive / moderate / conservative | moderate |
| `--model` | random_forest / ridge / lasso / gradient_boosting | random_forest |
| `--rebalance` | W / M / Q | M |
| `--objective` | sharpe / minvol / maxret | sharpe |
| `--no-ml` | — | False |

## Tech Stack

Python · yfinance · scikit-learn · scipy · pandas · numpy · matplotlib

Push it:
git add README.md
git commit -m "Add README"
git push
