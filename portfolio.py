"""
portfolio.py — Multi-asset ML portfolio simulation.

At each rebalance day (every REBALANCE_FREQ trading days):
    - Collect signals from each ticker's trained model
    - If one or more tickers signal Buy → split capital evenly among them
    - If none signal Buy → move fully to cash

Transaction costs of 0.1% applied on each trade.
Compared against a blended equal-weight buy-and-hold baseline.
Model selection per ticker is set via default_model in config.py.

Usage:
    python portfolio.py
    python portfolio.py --tickers QQQ NVDA TSLA
    python portfolio.py --tickers QQQ NVDA TSLA --initial-cash 25000
"""

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

import backtest
from config import CONFIGS


PLOTS_DIR        = os.path.join(os.path.dirname(__file__), "data", "plots")
TRADING_DAYS     = 252
INITIAL_CASH     = 10_000.0
TRANSACTION_COST = 0.001  # 0.1% per trade
REBALANCE_FREQ   = 5      # rebalance every 5 trading days (~weekly)

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

NEUTRAL       = "#5b8cff"
BH_COLOR      = "#ffaa00"
DOWN_COLOR    = "#ff4f4f"
TICKER_COLORS = ["#5b8cff", "#00c896", "#ff4f4f", "#ffaa00", "#cc77ff"]


# ─── Signals ──────────────────────────────────────────────────────────────────

def collect_signals(tickers: list[str]) -> dict[str, pd.DataFrame]:
    results = {}
    for ticker in tickers:
        model = CONFIGS[ticker]["default_model"]
        results[ticker] = backtest.run(ticker, model)
    return results


def prepare_data(ticker_results: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    signals = pd.DataFrame({t: r["signal"] for t, r in ticker_results.items()})
    closes  = pd.DataFrame({t: r["close"]  for t, r in ticker_results.items()})

    # Note: no shift needed — backtest.py already executes signals
    # at next day's open, so lookahead bias is already prevented.
    common = signals.dropna().index.intersection(closes.dropna().index)
    return signals.loc[common], closes.loc[common]


# ─── Simulation ───────────────────────────────────────────────────────────────

def simulate(ticker_results: dict[str, pd.DataFrame], initial_cash: float) -> pd.DataFrame:
    signals, prices = prepare_data(ticker_results)

    cash     = initial_cash
    holdings = {t: 0.0 for t in ticker_results}
    rows     = []

    for i, date in enumerate(signals.index):
        if i % REBALANCE_FREQ == 0:
            active      = [t for t in signals.columns if signals.loc[date, t] == 1]
            total_value = cash + sum(holdings[t] * prices.loc[date, t] for t in holdings)

            if active:
                target_weight = 1.0 / len(active)
                for t in holdings:
                    price         = prices.loc[date, t]
                    current_value = holdings[t] * price
                    target_value  = total_value * (target_weight if t in active else 0.0)
                    diff          = target_value - current_value
                    if abs(diff) > 0:
                        cash        -= diff + abs(diff) * TRANSACTION_COST
                        holdings[t] += diff / price
            else:
                # No signals — liquidate to cash
                for t in holdings:
                    if holdings[t] > 0:
                        value       = holdings[t] * prices.loc[date, t]
                        cash       += value - value * TRANSACTION_COST
                        holdings[t] = 0.0

        portfolio_value = cash + sum(holdings[t] * prices.loc[date, t] for t in holdings)

        rows.append({
            "date":        date,
            "portfolio_value": portfolio_value,
            "cash":        cash,
            "n_positions": sum(1 for t in holdings if holdings[t] > 0),
        })

    result = pd.DataFrame(rows).set_index("date")

    # Blended buy-and-hold baseline — equal weight, no rebalancing
    weight = 1.0 / len(prices.columns)
    result["buy_and_hold"] = sum(
        weight * (initial_cash / prices.iloc[0][t]) * prices[t]
        for t in prices.columns
    )

    return result


# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(results: pd.DataFrame, initial_cash: float) -> dict:
    pv      = results["portfolio_value"]
    bh      = results["buy_and_hold"]
    n_years = len(results) / TRADING_DAYS

    total_return    = (pv.iloc[-1] - initial_cash) / initial_cash
    bh_total_return = (bh.iloc[-1] - initial_cash) / initial_cash
    ann_return      = (1 + total_return)    ** (1 / n_years) - 1
    bh_ann_return   = (1 + bh_total_return) ** (1 / n_years) - 1

    daily        = pv.pct_change().dropna()
    sharpe       = (daily.mean() / daily.std() * np.sqrt(TRADING_DAYS)) if daily.std() > 0 else 0.0
    max_drawdown = (pv / pv.cummax() - 1).min()
    pct_invested = (results["n_positions"] > 0).sum() / len(results)

    return {
        "total_return":    total_return,
        "bh_total_return": bh_total_return,
        "ann_return":      ann_return,
        "bh_ann_return":   bh_ann_return,
        "sharpe":          sharpe,
        "max_drawdown":    max_drawdown,
        "pct_invested":    pct_invested,
        "n_days":          len(results),
    }


def print_metrics(metrics: dict, tickers: list[str]) -> None:
    print(f"\n{'═' * 60}")
    print(f"  MINT Portfolio — {' + '.join(tickers)}")
    print(f"{'═' * 60}")
    print(f"  {'Metric':<25} {'Portfolio':>10}  {'B&H Blend':>10}")
    print(f"  {'─'*25} {'─'*10}  {'─'*10}")
    print(f"  {'Total Return':<25} {metrics['total_return']:>+9.2%}  {metrics['bh_total_return']:>+9.2%}")
    print(f"  {'Annualized Return':<25} {metrics['ann_return']:>+9.2%}  {metrics['bh_ann_return']:>+9.2%}")
    print(f"  {'Sharpe Ratio':<25} {metrics['sharpe']:>10.3f}  {'—':>10}")
    print(f"  {'Max Drawdown':<25} {metrics['max_drawdown']:>+9.2%}  {'—':>10}")
    print(f"  {'Days Invested':<25} {metrics['pct_invested']:>9.1%}  {'—':>10}")
    print(f"  {'Trading Days':<25} {metrics['n_days']:>10}  {'—':>10}")
    print(f"{'═' * 60}\n")


# ─── Plot ─────────────────────────────────────────────────────────────────────

def plot_equity_curve(
    results: pd.DataFrame,
    ticker_results: dict[str, pd.DataFrame],
    tickers: list[str],
    initial_cash: float,
) -> None:
    plot_dir = os.path.join(PLOTS_DIR, "portfolio")
    os.makedirs(plot_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(f"MINT Portfolio — {' + '.join(tickers)}", fontsize=14)

    # ── Top: combined portfolio vs blended B&H ────────────────────────────
    axes[0].plot(results.index, results["portfolio_value"],
                 color=NEUTRAL, linewidth=2, label="ML Portfolio", zorder=3)
    axes[0].plot(results.index, results["buy_and_hold"],
                 color=BH_COLOR, linewidth=1.5, linestyle="--", label="Blended B&H", alpha=0.8)

    cash_mask = results["n_positions"] == 0
    if cash_mask.any():
        axes[0].fill_between(results.index,
                             results["portfolio_value"].min(),
                             results["portfolio_value"].max(),
                             where=cash_mask, alpha=0.08,
                             color=DOWN_COLOR, label="In cash")

    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    axes[0].set_ylabel("Portfolio Value")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.4)

    # ── Bottom: individual ticker model curves (normalised) ───────────────
    common = results.index
    for ticker, color in zip(tickers, TICKER_COLORS):
        aligned = ticker_results[ticker]["portfolio_value"].reindex(common).ffill()
        scale   = initial_cash / aligned.iloc[0]
        axes[1].plot(common, aligned * scale, color=color,
                     linewidth=1.2, label=ticker, alpha=0.85)

    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    axes[1].set_ylabel("Individual Model Value (normalised)")
    axes[1].set_xlabel("Date")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.4)

    plt.tight_layout()
    path = os.path.join(plot_dir, "portfolio_equity_curve.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[portfolio] Saved → {path}")


# ─── Pipeline ─────────────────────────────────────────────────────────────────

def run(tickers: list[str], initial_cash: float) -> None:
    print(f"\n{'═' * 60}")
    print(f"  MINT Portfolio Simulation")
    print(f"  Tickers : {' | '.join(tickers)}")
    print(f"  Capital : ${initial_cash:,.2f}")
    print(f"{'═' * 60}")

    ticker_results = collect_signals(tickers)
    results        = simulate(ticker_results, initial_cash)
    metrics        = compute_metrics(results, initial_cash)

    print_metrics(metrics, tickers)
    plot_equity_curve(results, ticker_results, tickers, initial_cash)

    print("[portfolio] Done.")


if __name__ == "__main__":
    default_tickers = [t for t in CONFIGS if CONFIGS[t].get("default_model") is not None]

    parser = argparse.ArgumentParser(
        description="MINT — Multi-asset ML portfolio simulation."
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=default_tickers,
        help="Tickers to include (default: all with default_model set)"
    )
    parser.add_argument(
        "--initial-cash",
        type=float,
        default=INITIAL_CASH,
        help="Starting capital (default: 10000)"
    )
    args = parser.parse_args()

    run(tickers=[t.upper() for t in args.tickers], initial_cash=args.initial_cash)