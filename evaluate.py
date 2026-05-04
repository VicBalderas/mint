"""
evaluate.py — Compute performance metrics and generate plots from backtest output.

Calls backtest.run() internally — no need to run backtest.py separately.
Saves plots to data/plots/<TICKER>/.

Metrics: total return, annualized return, Sharpe ratio, max drawdown, win rate.
Plots:
    1. equity_curve.png     — model vs buy-and-hold portfolio value over time
    2. drawdown.png         — rolling drawdown over time
    3. monthly_returns.png  — monthly return heatmap

Usage:
    python evaluate.py --ticker QQQ --model logreg
"""

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

import backtest


PLOTS_DIR = os.path.join(os.path.dirname(__file__), "data", "plots")

plt.rcParams.update({
    "figure.facecolor": "#0f0f0f",
    "axes.facecolor":   "#1a1a1a",
    "axes.edgecolor":   "#444444",
    "axes.labelcolor":  "#cccccc",
    "axes.titlecolor":  "#ffffff",
    "xtick.color":      "#888888",
    "ytick.color":      "#888888",
    "text.color":       "#cccccc",
    "grid.color":       "#2a2a2a",
    "grid.linewidth":   0.8,
    "font.family":      "monospace",
    "font.size":        10,
})

UP_COLOR   = "#00c896"
DOWN_COLOR = "#ff4f4f"
NEUTRAL    = "#5b8cff"
BH_COLOR   = "#ffaa00"

TRADING_DAYS = 252


def get_plot_dir(ticker: str) -> str:
    plot_dir = os.path.join(PLOTS_DIR, ticker.upper())
    os.makedirs(plot_dir, exist_ok=True)
    return plot_dir


def save_plot(path: str) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[evaluate] Saved → {path}")


# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(results: pd.DataFrame, initial_cash: float) -> dict:
    pv     = results["portfolio_value"]
    bh     = results["buy_and_hold"]
    n_days = len(results)
    n_years = n_days / TRADING_DAYS

    # Returns
    total_return    = (pv.iloc[-1] - initial_cash) / initial_cash
    bh_total_return = (bh.iloc[-1] - initial_cash) / initial_cash
    ann_return      = (1 + total_return) ** (1 / n_years) - 1
    bh_ann_return   = (1 + bh_total_return) ** (1 / n_years) - 1

    # Daily returns for Sharpe
    daily_returns = pv.pct_change().dropna()
    sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(TRADING_DAYS)
              if daily_returns.std() > 0 else 0.0)

    # Max drawdown
    rolling_max  = pv.cummax()
    drawdown     = (pv - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    # Trade stats
    trades    = results[results["action"] == "sell"].copy()
    n_trades  = len(results[results["action"] == "buy"])

    # Win rate — compare exec_price at buy vs exec_price at sell (actual trade P&L)
    buy_prices  = results[results["action"] == "buy"]["exec_price"].values
    sell_prices = results[results["action"] == "sell"]["exec_price"].values
    n_pairs     = min(len(buy_prices), len(sell_prices))
    win_rate    = (np.sum(sell_prices[:n_pairs] > buy_prices[:n_pairs]) / n_pairs
                   if n_pairs > 0 else 0.0)

    return {
        "total_return":    total_return,
        "bh_total_return": bh_total_return,
        "ann_return":      ann_return,
        "bh_ann_return":   bh_ann_return,
        "sharpe":          sharpe,
        "max_drawdown":    max_drawdown,
        "n_trades":        n_trades,
        "win_rate":        win_rate,
        "n_days":          n_days,
    }


def print_metrics(metrics: dict, ticker: str, model_name: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  Evaluation — {ticker.upper()} — {model_name.upper()}")
    print(f"{'─' * 60}")
    print(f"  {'Metric':<25} {'Model':>10}  {'Buy & Hold':>10}")
    print(f"  {'─'*25} {'─'*10}  {'─'*10}")
    print(f"  {'Total Return':<25} {metrics['total_return']:>+9.2%}  {metrics['bh_total_return']:>+9.2%}")
    print(f"  {'Annualized Return':<25} {metrics['ann_return']:>+9.2%}  {metrics['bh_ann_return']:>+9.2%}")
    print(f"  {'Sharpe Ratio':<25} {metrics['sharpe']:>10.3f}  {'—':>10}")
    print(f"  {'Max Drawdown':<25} {metrics['max_drawdown']:>+9.2%}  {'—':>10}")
    print(f"  {'Trades':<25} {metrics['n_trades']:>10}  {'—':>10}")
    print(f"  {'Win Rate':<25} {metrics['win_rate']:>9.1%}  {'—':>10}")
    print(f"  {'Days in test':<25} {metrics['n_days']:>10}  {'—':>10}")
    print(f"{'─' * 60}\n")


# ─── Plot 1: Equity Curve ─────────────────────────────────────────────────────

def plot_equity_curve(results: pd.DataFrame, ticker: str, model_name: str, plot_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle(f"{ticker} — Equity Curve (Test Period)", fontsize=14)

    ax.plot(results.index, results["portfolio_value"], color=NEUTRAL,
            linewidth=1.5, label=f"Model ({model_name})")
    ax.plot(results.index, results["buy_and_hold"], color=BH_COLOR,
            linewidth=1.5, linestyle="--", label="Buy & Hold", alpha=0.8)

    # Mark buy/sell points
    buys  = results[results["action"] == "buy"]
    sells = results[results["action"] == "sell"]
    ax.scatter(buys.index,  buys["portfolio_value"],  marker="^", color=UP_COLOR,
               s=60, zorder=5, label="Buy")
    ax.scatter(sells.index, sells["portfolio_value"], marker="v", color=DOWN_COLOR,
               s=60, zorder=5, label="Sell")

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.set_ylabel("Portfolio Value")
    ax.set_xlabel("Date")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.4)

    save_plot(os.path.join(plot_dir, "8_equity_curve.png"))


# ─── Plot 2: Drawdown ─────────────────────────────────────────────────────────

def plot_drawdown(results: pd.DataFrame, ticker: str, plot_dir: str) -> None:
    rolling_max = results["portfolio_value"].cummax()
    drawdown    = (results["portfolio_value"] - rolling_max) / rolling_max * 100

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle(f"{ticker} — Drawdown (Test Period)", fontsize=14)

    ax.fill_between(results.index, drawdown, 0, color=DOWN_COLOR, alpha=0.5)
    ax.plot(results.index, drawdown, color=DOWN_COLOR, linewidth=1)
    ax.axhline(0, color="#ffffff", linewidth=0.6, alpha=0.4)

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax.set_ylabel("Drawdown %")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.4)

    save_plot(os.path.join(plot_dir, "9_drawdown.png"))


# ─── Plot 3: Monthly Returns Heatmap ──────────────────────────────────────────

def plot_monthly_returns(results: pd.DataFrame, ticker: str, plot_dir: str) -> None:
    # Resample to monthly, compute return per month
    monthly = results["portfolio_value"].resample("ME").last().pct_change().dropna() * 100
    monthly.index = monthly.index.to_period("M")

    years  = sorted(set(monthly.index.year))
    months = list(range(1, 13))
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    grid = pd.DataFrame(index=years, columns=months, dtype=float)
    for period, val in monthly.items():
        grid.loc[period.year, period.month] = val

    fig, ax = plt.subplots(figsize=(14, max(3, len(years) * 0.7 + 1)))
    fig.suptitle(f"{ticker} — Monthly Returns % (Model)", fontsize=14)

    vmax = max(abs(grid.values[~np.isnan(grid.values.astype(float))].max()),
               abs(grid.values[~np.isnan(grid.values.astype(float))].min()), 5)

    im = ax.imshow(grid.values.astype(float), cmap="RdYlGn", aspect="auto",
                   vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, format="%.1f%%", shrink=0.8)

    ax.set_xticks(range(12))
    ax.set_xticklabels(month_labels)
    ax.set_yticks(range(len(years)))
    ax.set_yticklabels(years)

    # Annotate cells
    for i, year in enumerate(years):
        for j, month in enumerate(months):
            val = grid.loc[year, month]
            if pd.isna(val):
                continue
            ax.text(j, i, f"{float(val):.1f}%", ha="center", va="center",
                    fontsize=8, color="#000000" if abs(float(val)) < vmax * 0.6 else "#ffffff")

    save_plot(os.path.join(plot_dir, "10_monthly_returns.png"))


# ─── Pipeline ─────────────────────────────────────────────────────────────────

def run(ticker: str, model_name: str) -> dict:
    print(f"\n{'─' * 60}")
    print(f"  MINT Evaluate — {ticker.upper()} — {model_name.upper()}")
    print(f"{'─' * 60}\n")

    results  = backtest.run(ticker, model_name)
    metrics  = compute_metrics(results, backtest.INITIAL_CASH)
    plot_dir = get_plot_dir(ticker)

    print_metrics(metrics, ticker, model_name)

    plot_equity_curve(results, ticker, model_name, plot_dir)
    plot_drawdown(results, ticker, plot_dir)
    plot_monthly_returns(results, ticker, plot_dir)

    print(f"[evaluate] Done. Plots saved to {plot_dir}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, required=True)
    parser.add_argument("--model",  type=str, required=True, choices=["logreg", "randomforest"])
    args = parser.parse_args()

    run(ticker=args.ticker, model_name=args.model)