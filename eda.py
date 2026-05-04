"""
eda.py — Exploratory Data Analysis. Run after preprocess.py, before train.py.
Loads the TRAIN split only — avoids leaking val/test info into feature selection.
Saves plots to data/plots/<TICKER>/.

Plots:
    1. label_distribution   — class balance + rolling regime view
    2. feature_distributions — histograms by class
    3. correlation_heatmap  — feature-feature correlation matrix
    4. feature_label_corr   — each feature's correlation with label
    5. class_separation     — boxplots by class
    6. returns_over_time    — cumulative return + split boundaries
    7. feature_linearity    — feature vs forward return (binned)

Usage:
    python eda.py --ticker QQQ
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import get_config


SPLITS_DIR = os.path.join(os.path.dirname(__file__), "data", "splits")
PLOTS_DIR  = os.path.join(os.path.dirname(__file__), "data", "plots")

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


def load_train(ticker: str) -> pd.DataFrame:
    # EDA runs on train split only — prevents val/test info from influencing feature selection
    path = os.path.join(SPLITS_DIR, f"{ticker.upper()}_train.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"[eda] Train split not found: {path}\n      Run preprocess.py first.")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    print(f"[eda] Loaded train split — {len(df)} rows  |  {df.index[0].date()} → {df.index[-1].date()}")
    return df


def get_plot_dir(ticker: str) -> str:
    plot_dir = os.path.join(PLOTS_DIR, ticker.upper())
    os.makedirs(plot_dir, exist_ok=True)
    return plot_dir


def grid_axes(n_features: int, n_cols: int = 3):
    n_rows = int(np.ceil(n_features / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3.5))
    return fig, axes, n_rows


def hide_unused(axes, n_features: int, n_rows: int, n_cols: int = 3) -> None:
    for j in range(n_features, n_rows * n_cols):
        axes[j // n_cols][j % n_cols].set_visible(False)


def save_plot(path: str) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[eda] Saved → {path}")


# ─── Plot 1: Label Distribution ───────────────────────────────────────────────

def plot_label_distribution(df: pd.DataFrame, ticker: str, plot_dir: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle(f"{ticker} — Label Distribution (Train Split)", fontsize=14)

    counts = df["label"].value_counts().sort_index()
    bars = axes[0].bar(["Down / Flat (0)", "Up (1)"], counts.values,
                       color=[DOWN_COLOR, UP_COLOR], width=0.5, edgecolor="#000000")
    for bar, count in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                     f"{count} ({count / len(df) * 100:.1f}%)", ha="center", va="bottom", fontsize=11)
    axes[0].set_title("Overall Class Balance")
    axes[0].set_ylabel("Count")
    axes[0].grid(axis="y")
    axes[0].set_ylim(0, counts.max() * 1.15)

    rolling_up = df["label"].rolling(60).mean() * 100
    axes[1].plot(df.index, rolling_up, color=UP_COLOR, linewidth=1.5, label="60-day rolling up%")
    axes[1].axhline(50, color="#ffffff", linewidth=0.8, linestyle="--", alpha=0.4, label="50% baseline")
    axes[1].fill_between(df.index, rolling_up, 50, where=(rolling_up >= 50), alpha=0.15, color=UP_COLOR)
    axes[1].fill_between(df.index, rolling_up, 50, where=(rolling_up < 50),  alpha=0.15, color=DOWN_COLOR)
    axes[1].set_title("Rolling 60-Day Up% (Regime View)")
    axes[1].set_ylabel("% Up Days")
    axes[1].set_xlabel("Date")
    axes[1].legend()
    axes[1].grid(True)

    save_plot(os.path.join(plot_dir, "1_label_distribution.png"))


# ─── Plot 2: Feature Distributions ───────────────────────────────────────────

def plot_feature_distributions(df: pd.DataFrame, ticker: str, plot_dir: str, feature_cols: list[str]) -> None:
    fig, axes, n_rows = grid_axes(len(feature_cols))
    fig.suptitle(f"{ticker} — Feature Distributions by Class (Train)", fontsize=14)

    up, down = df[df["label"] == 1], df[df["label"] == 0]

    for i, col in enumerate(feature_cols):
        ax = axes[i // 3][i % 3]
        lo, hi = df[col].quantile(0.01), df[col].quantile(0.99)
        ax.hist(down[col].clip(lo, hi), bins=40, alpha=0.6, color=DOWN_COLOR,
                label="Down/Flat", density=True, edgecolor="none")
        ax.hist(up[col].clip(lo, hi),   bins=40, alpha=0.6, color=UP_COLOR,
                label="Up", density=True, edgecolor="none")
        ax.set_title(col, fontsize=10)
        ax.grid(True, alpha=0.4)
        if i == 0:
            ax.legend(fontsize=8)

    hide_unused(axes, len(feature_cols), n_rows)
    save_plot(os.path.join(plot_dir, "2_feature_distributions.png"))


# ─── Plot 3: Correlation Heatmap ──────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame, ticker: str, plot_dir: str, feature_cols: list[str]) -> None:
    corr = df[feature_cols + ["label"]].corr()

    fig, ax = plt.subplots(figsize=(13, 11))
    fig.suptitle(f"{ticker} — Feature Correlation Heatmap (Train)", fontsize=14)

    # Mask upper triangle — lower triangle only avoids duplicate pairs
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, linewidths=0.5, linecolor="#111111",
                ax=ax, annot_kws={"size": 8}, cbar_kws={"shrink": 0.8})
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

    high_corr = [
        f"{corr.columns[i]}↔{corr.columns[j]} ({corr.iloc[i,j]:.2f})"
        for i in range(len(corr.columns))
        for j in range(i + 1, len(corr.columns))
        if abs(corr.iloc[i, j]) > 0.85 and corr.columns[j] != "label"
    ]
    if high_corr:
        fig.text(0.5, -0.02, "High correlation (>0.85): " + ", ".join(high_corr),
                 ha="center", fontsize=8, color="#ffaa00", wrap=True)

    save_plot(os.path.join(plot_dir, "3_correlation_heatmap.png"))


# ─── Plot 4: Feature-Label Correlation ───────────────────────────────────────

def plot_feature_label_correlation(df: pd.DataFrame, ticker: str, plot_dir: str, feature_cols: list[str]) -> None:
    corrs = df[feature_cols].corrwith(df["label"]).sort_values()

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.suptitle(f"{ticker} — Feature Correlation with Label (Train)", fontsize=14)

    bars = ax.barh(corrs.index, corrs.values,
                   color=[UP_COLOR if v >= 0 else DOWN_COLOR for v in corrs.values],
                   edgecolor="#111111", height=0.6)
    for bar, val in zip(bars, corrs.values):
        x = bar.get_width()
        ax.text(x + (0.002 if x >= 0 else -0.002), bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left" if x >= 0 else "right", fontsize=9)

    ax.axvline(0, color="#ffffff", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Pearson Correlation with Label")
    ax.set_title("Positive = tends to predict Up  |  Negative = tends to predict Down\n"
                 "Near zero = weak linear signal (doesn't mean useless for non-linear models)",
                 fontsize=9, pad=8)
    ax.grid(axis="x", alpha=0.4)
    save_plot(os.path.join(plot_dir, "4_feature_label_correlation.png"))


# ─── Plot 5: Class Separation ─────────────────────────────────────────────────

def plot_class_separation(df: pd.DataFrame, ticker: str, plot_dir: str, feature_cols: list[str]) -> None:
    fig, axes, n_rows = grid_axes(len(feature_cols))
    fig.suptitle(f"{ticker} — Class Separation per Feature (Train)", fontsize=14)

    for i, col in enumerate(feature_cols):
        ax = axes[i // 3][i % 3]
        lo, hi = df[col].quantile(0.02), df[col].quantile(0.98)
        clipped = df[[col, "label"]].copy()
        clipped[col] = clipped[col].clip(lo, hi)

        bp = ax.boxplot(
            [clipped[clipped["label"] == 0][col].values, clipped[clipped["label"] == 1][col].values],
            patch_artist=True, tick_labels=["Down/Flat", "Up"],
            medianprops=dict(color="#ffffff", linewidth=1.5),
            whiskerprops=dict(color="#888888"), capprops=dict(color="#888888"),
            flierprops=dict(marker=".", color="#555555", markersize=2),
        )
        bp["boxes"][0].set_facecolor(DOWN_COLOR + "55")
        bp["boxes"][1].set_facecolor(UP_COLOR + "55")
        ax.set_title(col, fontsize=10)
        ax.grid(axis="y", alpha=0.4)

    hide_unused(axes, len(feature_cols), n_rows)
    save_plot(os.path.join(plot_dir, "5_class_separation.png"))


# ─── Plot 6: Returns Over Time ────────────────────────────────────────────────

def plot_returns_over_time(df: pd.DataFrame, ticker: str, plot_dir: str, cfg: dict) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    fig.suptitle(f"{ticker} — Price & Returns (Train Split)", fontsize=14)

    cum_returns = (1 + df["returns"]).cumprod()
    axes[0].plot(df.index, cum_returns, color=NEUTRAL, linewidth=1.2)
    axes[0].fill_between(df.index, 1, cum_returns, where=(cum_returns >= 1), alpha=0.15, color=UP_COLOR)
    axes[0].fill_between(df.index, 1, cum_returns, where=(cum_returns < 1),  alpha=0.15, color=DOWN_COLOR)
    axes[0].axhline(1, color="#ffffff", linewidth=0.6, linestyle="--", alpha=0.4)
    axes[0].set_title("Cumulative Return")
    axes[0].set_ylabel("Cumulative Return (1 = start)")
    axes[0].grid(True, alpha=0.4)

    axes[1].bar(df.index, df["returns"],
                color=[UP_COLOR if r >= 0 else DOWN_COLOR for r in df["returns"]],
                alpha=0.7, width=1)
    axes[1].axhline(0, color="#ffffff", linewidth=0.5, alpha=0.4)
    axes[1].set_title("Daily Returns")
    axes[1].set_ylabel("Return")
    axes[1].set_xlabel("Date")
    axes[1].grid(True, alpha=0.4)

    save_plot(os.path.join(plot_dir, "6_returns_over_time.png"))


# ─── Plot 7: Feature Linearity ────────────────────────────────────────────────

def plot_feature_linearity(df: pd.DataFrame, ticker: str, plot_dir: str, feature_cols: list[str]) -> None:
    df = df.copy()
    # Use label's implicit forward return as proxy — close is available in splits
    df["fwd_return"] = df["close"].shift(-1) / df["close"] - 1 if "close" in df.columns else df["returns"].shift(-1)
    df = df.dropna(subset=["fwd_return"])

    fig, axes, n_rows = grid_axes(len(feature_cols))
    fig.suptitle(f"{ticker} — Feature vs Forward Return (binned mean, Train)\n"
                 "Straight line → linear model fits well  |  Curved → tree model may help", fontsize=12)

    for i, col in enumerate(feature_cols):
        ax = axes[i // 3][i % 3]
        lo, hi = df[col].quantile(0.02), df[col].quantile(0.98)
        subset = df[[col, "fwd_return"]].copy()
        subset[col] = subset[col].clip(lo, hi)

        try:
            subset["bin"] = pd.qcut(subset[col], q=20, duplicates="drop")
            binned = subset.groupby("bin", observed=True)["fwd_return"].mean()
            x = range(len(binned))
            ax.bar(x, binned.values,
                   color=[UP_COLOR if v >= 0 else DOWN_COLOR for v in binned.values],
                   alpha=0.8, edgecolor="none")
            ax.axhline(0, color="#ffffff", linewidth=0.6, alpha=0.5)
            if len(binned) > 3:
                z = np.polyfit(x, binned.values, 1)
                ax.plot(x, np.poly1d(z)(x), color="#ffaa00", linewidth=1.5, linestyle="--", alpha=0.8)
        except Exception:
            ax.text(0.5, 0.5, "insufficient data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8)

        ax.set_title(col, fontsize=10)
        ax.set_xlabel("Feature quantile bins →")
        ax.set_ylabel("Mean fwd return")
        ax.grid(True, alpha=0.3)

    hide_unused(axes, len(feature_cols), n_rows)
    save_plot(os.path.join(plot_dir, "7_feature_linearity.png"))


# ─── Summary ──────────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame, ticker: str, feature_cols: list[str]) -> None:
    print(f"\n{'─' * 60}")
    print(f"  EDA Summary — {ticker.upper()} (Train Split)")
    print(f"{'─' * 60}")
    print(f"  Rows    : {len(df)}")
    print(f"  Range   : {df.index[0].date()} → {df.index[-1].date()}")
    print(f"  Labels  : {df['label'].mean() * 100:.1f}% up  |  {(1 - df['label'].mean()) * 100:.1f}% down/flat")

    label_corrs = df[feature_cols].corrwith(df["label"])
    abs_corrs   = label_corrs.abs().sort_values(ascending=False)

    print(f"\n  Feature-Label Correlations (Pearson):")
    for feat, val in abs_corrs.items():
        sign = "+" if label_corrs[feat] > 0 else "-"
        print(f"    {feat:<20} {sign}{val:.4f}  {'█' * int(val * 60)}")

    print(f"\n  High feature-feature correlations (>0.7):")
    corr_matrix = df[feature_cols].corr().abs()
    pairs = [(feature_cols[i], feature_cols[j], corr_matrix.iloc[i, j])
             for i in range(len(feature_cols))
             for j in range(i + 1, len(feature_cols))
             if corr_matrix.iloc[i, j] > 0.7]

    if pairs:
        for a, b, val in pairs:
            print(f"    {a:<20} ↔  {b:<20}  {val:.4f}")
    else:
        print("    None above 0.7 — no redundant features detected.")

    print(f"\n  Recommendation:")
    max_corr = abs_corrs.max()
    if max_corr < 0.05:
        print("    All feature-label correlations near zero.")
        print("    → Linear models will struggle. Try tree models or add features.")
    elif max_corr < 0.10:
        print("    Weak linear signal. → Try both linear and tree models.")
    else:
        print("    Moderate linear signal. → Linear models are a reasonable starting point.")
    print(f"{'─' * 60}\n")


# ─── Pipeline ─────────────────────────────────────────────────────────────────

def run(ticker: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  MINT EDA — {ticker.upper()}")
    print(f"{'─' * 60}\n")

    df           = load_train(ticker)
    cfg          = get_config(ticker, "data")
    feature_cols = cfg["features"]
    plot_dir     = get_plot_dir(ticker)

    print(f"[eda] {len(feature_cols)} active features: {feature_cols}")
    print(f"[eda] Saving plots to {plot_dir}\n")

    plot_label_distribution(df, ticker, plot_dir)
    plot_feature_distributions(df, ticker, plot_dir, feature_cols)
    plot_correlation_heatmap(df, ticker, plot_dir, feature_cols)
    plot_feature_label_correlation(df, ticker, plot_dir, feature_cols)
    plot_class_separation(df, ticker, plot_dir, feature_cols)
    plot_returns_over_time(df, ticker, plot_dir, cfg)
    plot_feature_linearity(df, ticker, plot_dir, feature_cols)

    print_summary(df, ticker, feature_cols)
    print(f"[eda] Done. All plots saved to {plot_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, required=True)
    args = parser.parse_args()
    run(ticker=args.ticker)