import argparse
import json
import os
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("MacOSX")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker

from data import fetch_price_data, compute_returns, validate_data
from features import build_all_features
from model import train_all_models, get_predicted_returns
from optimizer import (compute_covariance, compute_expected_returns,
                       optimize_portfolio, compute_efficient_frontier, portfolio_stats)
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from backtest import (Backtester, equal_weight_portfolio,
                      compute_performance_metrics, compute_drawdown_series, default_optimize_fn)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

STYLE = {
    "bg": "#0D1117", "panel": "#161B22", "border": "#30363D",
    "text": "#E6EDF3", "muted": "#8B949E", "accent1": "#58A6FF",
    "accent2": "#3FB950", "accent3": "#FF7B72", "accent4": "#D2A8FF",
    "accent5": "#FFA657", "grid": "#21262D",
}
PALETTE = [STYLE["accent1"], STYLE["accent2"], STYLE["accent3"],
           STYLE["accent4"], STYLE["accent5"], "#79C0FF", "#56D364"]

RISK_PROFILES = {
    "aggressive":   {"max_weight": 0.40, "objective": "sharpe",  "ml_blend": 0.55, "transaction_cost": 0.001},
    "moderate":     {"max_weight": 0.35, "objective": "sharpe",  "ml_blend": 0.40, "transaction_cost": 0.001},
    "conservative": {"max_weight": 0.30, "objective": "minvol",  "ml_blend": 0.25, "transaction_cost": 0.001},
}


def parse_args():
    p = argparse.ArgumentParser(description="Smart Portfolio Allocation System")
    p.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "JPM", "BRK-B", "UNH"])
    p.add_argument("--start",  default="2020-01-01")
    p.add_argument("--end",    default=datetime.today().strftime("%Y-%m-%d"))
    p.add_argument("--risk",   choices=["aggressive", "moderate", "conservative"], default="moderate")
    p.add_argument("--model",  choices=["random_forest", "ridge", "lasso", "gradient_boosting"], default="random_forest")
    p.add_argument("--objective", choices=["sharpe", "minvol", "maxret"], default=None)
    p.add_argument("--rebalance", choices=["W", "M", "Q"], default="M")
    p.add_argument("--initial-value", type=float, default=100_000.0)
    p.add_argument("--benchmark", default="SPY")
    p.add_argument("--no-ml", action="store_true")
    p.add_argument("--output-dir", default="results")
    return p.parse_args()


def set_dark_theme():
    plt.rcParams.update({
        "figure.facecolor": STYLE["bg"], "axes.facecolor": STYLE["panel"],
        "axes.edgecolor": STYLE["border"], "axes.labelcolor": STYLE["text"],
        "xtick.color": STYLE["muted"], "ytick.color": STYLE["muted"],
        "text.color": STYLE["text"], "grid.color": STYLE["grid"],
        "grid.linestyle": "--", "grid.alpha": 0.6,
        "legend.facecolor": STYLE["panel"], "legend.edgecolor": STYLE["border"],
        "font.family": "monospace", "font.size": 9,
    })


def plot_dashboard(smart_values, bench_values, equal_values, weights, weights_history,
                   frontier, smart_metrics, bench_metrics, equal_metrics, output_path, tickers):
    set_dark_theme()
    fig = plt.figure(figsize=(20, 14), facecolor=STYLE["bg"])
    fig.suptitle("SMART PORTFOLIO ALLOCATION SYSTEM", fontsize=16, fontweight="bold",
                 color=STYLE["text"], y=0.97, fontfamily="monospace")
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.38,
                           top=0.93, bottom=0.06, left=0.07, right=0.97)
    ax_nav = fig.add_subplot(gs[0, :2])
    ax_dd  = fig.add_subplot(gs[1, :2])
    ax_bar = fig.add_subplot(gs[0, 2])
    ax_wh  = fig.add_subplot(gs[1, 2])
    ax_ef  = fig.add_subplot(gs[2, :2])
    ax_mt  = fig.add_subplot(gs[2, 2])

    def sax(ax, title):
        ax.set_facecolor(STYLE["panel"])
        for sp in ax.spines.values():
            sp.set_edgecolor(STYLE["border"])
        ax.set_title(title, color=STYLE["muted"], fontsize=9, pad=8, loc="left")
        ax.grid(True, alpha=0.4)

    base = smart_values.iloc[0]
    ax_nav.plot(smart_values.index, smart_values / base * 100, color=STYLE["accent1"], lw=1.8, label="Smart Portfolio", zorder=3)
    ax_nav.plot(equal_values.index, equal_values / base * 100, color=STYLE["accent2"], lw=1.4, ls="--", label="Equal Weight", alpha=0.8)
    if bench_values is not None and len(bench_values) > 5:
        ax_nav.plot(bench_values.index, bench_values / bench_values.iloc[0] * 100, color=STYLE["accent3"], lw=1.2, ls=":", label="Benchmark", alpha=0.8)
    ax_nav.fill_between(smart_values.index, smart_values / base * 100, 100, alpha=0.08, color=STYLE["accent1"])
    ax_nav.axhline(100, color=STYLE["border"], lw=0.8)
    ax_nav.legend(fontsize=8, loc="upper left")
    ax_nav.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
    sax(ax_nav, "PORTFOLIO GROWTH  (rebased to 100)")

    dd = compute_drawdown_series(smart_values) * 100
    ax_dd.fill_between(dd.index, dd, 0, alpha=0.45, color=STYLE["accent3"])
    ax_dd.plot(dd.index, dd, color=STYLE["accent3"], lw=1.2)
    ax_dd.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    sax(ax_dd, "DRAWDOWN  (%)")

    sw = weights.sort_values(ascending=True)
    bars = ax_bar.barh(sw.index, sw.values * 100, color=PALETTE[:len(sw)], edgecolor=STYLE["bg"], lw=0.5)
    for b, v in zip(bars, sw.values):
        ax_bar.text(v * 100 + 0.3, b.get_y() + b.get_height() / 2,
                    f"{v*100:.1f}%", va="center", fontsize=8, color=STYLE["text"])
    ax_bar.set_xlim(0, sw.values.max() * 130)
    ax_bar.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    sax(ax_bar, "CURRENT ALLOCATION")

    if len(weights_history) > 1:
        wh = weights_history.fillna(0)
        ax_wh.stackplot(wh.index, [wh[c] * 100 for c in wh.columns],
                        labels=wh.columns, colors=PALETTE[:len(wh.columns)], alpha=0.85)
        ax_wh.set_ylim(0, 105)
        ax_wh.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
        ax_wh.legend(fontsize=6, loc="upper left", ncol=2)
    sax(ax_wh, "WEIGHTS OVER TIME")

    if frontier is not None and len(frontier) > 2:
        sc = ax_ef.scatter(frontier["volatility"] * 100, frontier["expected_return"] * 100,
                           c=frontier["sharpe"], cmap="plasma", s=18, alpha=0.8)
        cb = fig.colorbar(sc, ax=ax_ef, pad=0.02)
        cb.set_label("Sharpe", color=STYLE["muted"], fontsize=8)
        cb.ax.yaxis.set_tick_params(color=STYLE["muted"])
        plt.setp(cb.ax.yaxis.get_ticklabels(), color=STYLE["muted"])
    ax_ef.set_xlabel("Annualised Volatility (%)")
    ax_ef.set_ylabel("Annualised Return (%)")
    ax_ef.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    ax_ef.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    sax(ax_ef, "EFFICIENT FRONTIER  (colour = Sharpe)")

    ax_mt.set_facecolor(STYLE["panel"])
    for sp in ax_mt.spines.values():
        sp.set_edgecolor(STYLE["border"])
    ax_mt.set_title("PERFORMANCE METRICS", color=STYLE["muted"], fontsize=9, pad=8, loc="left")
    ax_mt.axis("off")

    rows = [
        ("Metric", "Smart", "EqWt", "Bench"),
        ("Total Return",  f"{smart_metrics['total_return']*100:.1f}%",  f"{equal_metrics['total_return']*100:.1f}%",  f"{bench_metrics.get('total_return',0)*100:.1f}%" if bench_metrics else "—"),
        ("CAGR",          f"{smart_metrics['cagr']*100:.1f}%",          f"{equal_metrics['cagr']*100:.1f}%",          f"{bench_metrics.get('cagr',0)*100:.1f}%" if bench_metrics else "—"),
        ("Ann. Vol",      f"{smart_metrics['ann_volatility']*100:.1f}%", f"{equal_metrics['ann_volatility']*100:.1f}%",f"{bench_metrics.get('ann_volatility',0)*100:.1f}%" if bench_metrics else "—"),
        ("Sharpe",        f"{smart_metrics['sharpe_ratio']:.2f}",        f"{equal_metrics['sharpe_ratio']:.2f}",        f"{bench_metrics.get('sharpe_ratio',0):.2f}" if bench_metrics else "—"),
        ("Max DD",        f"{smart_metrics['max_drawdown']*100:.1f}%",   f"{equal_metrics['max_drawdown']*100:.1f}%",   f"{bench_metrics.get('max_drawdown',0)*100:.1f}%" if bench_metrics else "—"),
        ("Sortino",       f"{smart_metrics['sortino_ratio']:.2f}",       f"{equal_metrics['sortino_ratio']:.2f}",       f"{bench_metrics.get('sortino_ratio',0):.2f}" if bench_metrics else "—"),
        ("Calmar",        f"{smart_metrics['calmar_ratio']:.2f}",        f"{equal_metrics['calmar_ratio']:.2f}",        f"{bench_metrics.get('calmar_ratio',0):.2f}" if bench_metrics else "—"),
        ("Win Rate",      f"{smart_metrics['win_rate']*100:.1f}%",       f"{equal_metrics['win_rate']*100:.1f}%",       f"{bench_metrics.get('win_rate',0)*100:.1f}%" if bench_metrics else "—"),
    ]
    for i, row in enumerate(rows):
        y_pos = 1.0 - i * (1.0 / len(rows)) - 0.02
        col_colors = [STYLE["muted"], STYLE["accent1"], STYLE["accent2"],
                      STYLE["accent3"] if bench_metrics else STYLE["muted"]]
        for j, (val, col) in enumerate(zip(row, col_colors)):
            ax_mt.text(j * 0.27, y_pos, val, transform=ax_mt.transAxes,
                       fontsize=8, color=col, fontweight="bold" if i == 0 else "normal", va="top")

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=STYLE["bg"])
    plt.close()
    logger.info(f"Dashboard saved → {output_path}")


def main():
    args = parse_args()
    profile = RISK_PROFILES[args.risk]
    if args.objective:
        profile["objective"] = args.objective
    use_ml = not args.no_ml
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"Tickers: {args.tickers}")
    logger.info(f"Period: {args.start} → {args.end}")
    logger.info(f"Risk: {args.risk}  |  Objective: {profile['objective']}  |  ML: {use_ml}")
    logger.info("=" * 60)

    prices = fetch_price_data(args.tickers, args.start, args.end)
    validate_data(prices)
    returns = compute_returns(prices)
    tickers = list(returns.columns)

    feature_dict = build_all_features(prices, returns, forward_return_window=5) if use_ml else {}

    ml_predictions = None
    if use_ml and feature_dict:
        model_results = train_all_models(feature_dict, model_type=args.model)
        ml_predictions = get_predicted_returns(model_results)
        logger.info(f"ML Predicted returns:\n{ml_predictions.round(5)}")

    mu = compute_expected_returns(returns, ml_predictions, ml_blend=profile["ml_blend"])
    cov = compute_covariance(returns, method="ledoit_wolf")
    opt_weights = optimize_portfolio(mu, cov, objective=profile["objective"], max_weight=profile["max_weight"])
    logger.info(f"Optimal weights:\n{opt_weights.round(4)}")

    frontier = compute_efficient_frontier(mu, cov, n_points=60, max_weight=profile["max_weight"])

    def opt_fn(returns_window, ml_preds, max_weight, ml_blend):
        return default_optimize_fn(returns_window, ml_preds, max_weight, ml_blend, objective=profile["objective"])

    tester = Backtester(
        prices=prices, returns=returns, optimize_fn=opt_fn,
        rebalance_frequency=args.rebalance, lookback_window=252,
        initial_value=args.initial_value, transaction_cost=profile["transaction_cost"],
        max_weight=profile["max_weight"], use_ml=use_ml,
        model_type=args.model, ml_blend=profile["ml_blend"],
    )
    backtest_results = tester.run()
    smart_values    = backtest_results["portfolio_values"]
    weights_history = backtest_results["weights_history"]
    smart_metrics   = backtest_results["metrics"]

    equal_values  = equal_weight_portfolio(returns, initial_value=args.initial_value,
                                           transaction_cost=profile["transaction_cost"],
                                           rebalance_frequency=args.rebalance)
    equal_metrics = compute_performance_metrics(equal_values, label="Equal Weight")

    bench_values, bench_metrics = None, {}
    if args.benchmark:
        try:
            bench_px = fetch_price_data([args.benchmark], args.start, args.end)
            bench_r  = compute_returns(bench_px)
            bench_nav = {}
            nav = args.initial_value
            for date, r in bench_r.iloc[:, 0].items():
                nav *= np.exp(r)
                bench_nav[date] = nav
            bench_values  = pd.Series(bench_nav)
            bench_metrics = compute_performance_metrics(bench_values, label=args.benchmark)
        except Exception as e:
            logger.warning(f"Benchmark fetch failed: {e}")

    print("\n" + "=" * 65)
    print("  PERFORMANCE SUMMARY")
    print("=" * 65)
    print(f"{'Metric':<22} {'Smart Portfolio':>15} {'Equal Weight':>15} {args.benchmark:>10}")
    print("-" * 65)
    for label_str, key, fmt in [
        ("Total Return",   "total_return",   "{:.2%}"),
        ("CAGR",           "cagr",           "{:.2%}"),
        ("Ann. Volatility","ann_volatility",  "{:.2%}"),
        ("Sharpe Ratio",   "sharpe_ratio",   "{:.3f}"),
        ("Sortino Ratio",  "sortino_ratio",  "{:.3f}"),
        ("Max Drawdown",   "max_drawdown",   "{:.2%}"),
        ("Calmar Ratio",   "calmar_ratio",   "{:.3f}"),
        ("Win Rate",       "win_rate",       "{:.2%}"),
    ]:
        print(f"  {label_str:<20} {fmt.format(smart_metrics.get(key,0)):>15} "
              f"{fmt.format(equal_metrics.get(key,0)):>15} "
              f"{fmt.format(bench_metrics.get(key,0)) if bench_metrics else '—':>10}")
    print("=" * 65)

    print("\n  FINAL PORTFOLIO ALLOCATION")
    print("-" * 40)
    for ticker, w in opt_weights.sort_values(ascending=False).items():
        print(f"  {ticker:<8} {'█' * int(w * 50):<25} {w*100:.1f}%")
    print("=" * 65)

    plot_dashboard(
        smart_values=smart_values, bench_values=bench_values, equal_values=equal_values,
        weights=opt_weights, weights_history=weights_history, frontier=frontier,
        smart_metrics=smart_metrics, bench_metrics=bench_metrics, equal_metrics=equal_metrics,
        output_path=os.path.join(args.output_dir, "dashboard.png"), tickers=tickers,
    )

    export = {
        "run_timestamp": datetime.now().isoformat(),
        "config": vars(args),
        "tickers": tickers,
        "allocation": opt_weights.round(6).to_dict(),
        "smart_portfolio_metrics": {k: float(v) if isinstance(v, (float, np.floating)) else v
                                    for k, v in smart_metrics.items()},
        "equal_weight_metrics": {k: float(v) if isinstance(v, (float, np.floating)) else v
                                  for k, v in equal_metrics.items()},
        "benchmark_metrics": {k: float(v) if isinstance(v, (float, np.floating)) else v
                               for k, v in bench_metrics.items()},
    }
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(export, f, indent=2)

    if not weights_history.empty:
        weights_history.to_csv(os.path.join(args.output_dir, "weights_history.csv"))

    pd.DataFrame({"Smart Portfolio": smart_values, "Equal Weight": equal_values}).to_csv(
        os.path.join(args.output_dir, "portfolio_values.csv"))

    logger.info("\n✅  Done. Results in: " + args.output_dir)


if __name__ == "__main__":
    main()