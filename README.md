# MINT — Machine Intelligence for Trading

An end-to-end ML pipeline for equity signal generation and multi-asset portfolio simulation. Designed to streamline the full workflow from data collection to portfolio simulation.

The entire pipeline is controlled from a single file: `config.py`.

---

## What it does

MINT trains classification models to predict next-day price direction for individual tickers, then simulates how those signals would have performed as a portfolio. Each step is a standalone script that can be run independently or as part of the full pipeline.

```
fetch → preprocess → eda → preprocess → train → evaluate → portfolio
```

It also ships a `predict.py` for generating live signals after market close — plug the output into any execution system you already use.

---

## Background

This project started with a simple question: could an ML-driven system generate signals that grow a portfolio and could the entire workflow be made fast, reproducible, and easy to iterate on?

From the start, the goal was not to beat the market, but to build a system that makes experimenting with ML in finance straightforward and honest. Financial data is noisy, feature–label relationships are weak, and consistently outperforming the market is inherently difficult. Rather than working around those constraints, this project embraces them.

MINT was designed as a clean, configurable pipeline where the full process from raw data to portfolio simulation could be run, inspected, and modified with minimal friction. The emphasis is on clarity, reproducibility, and rapid iteration through a single configuration file.

Several observations shaped the system:

- EDA before training revealed that multiple features were highly correlated, adding noise rather than signal
- Logistic Regression consistently outperformed Random Forest signaling that simpler models handled low-signal data more effectively
- Not all tickers are equally predictable in the sense that highly volatile assets like TSLA produced noisier signals than more stable ones like AAPL
- Portfolio-level behavior provided more insight than single-ticker results by achieving a Sharpe ratio of 1.628 while being invested ~60% of the time reflects a different risk profile, not just a different return

The full development process — including experiments, trade-offs, and design decisions — is documented in [`docs/devlog.md`](docs/devlog.md).  
Model-specific results and notes are available in [`docs/models.md`](docs/models.md).

---

## Pipeline

| Script          | What it does                                                   |
|-----------------|----------------------------------------------------------------|
| `fetch.py`      | Downloads raw OHLCV data via yfinance                          |
| `preprocess.py` | Engineers features, creates labels, splits train/val/test      |
| `eda.py`        | Correlation analysis and feature selection plots               |
| `train.py`      | Trains and saves a model + scaler                              |
| `evaluate.py`   | Backtests and generates performance plots                      |
| `portfolio.py`  | Multi-asset portfolio simulation across all configured tickers |
| `predict.py`    | Generates a live Buy/Sell signal for tomorrow's open           |

---

## Quickstart

```bash
# 1. Fetch data (optional — preprocess.py will fetch automatically if needed)
python fetch.py --ticker QQQ --start 2018-01-01

# 2. Preprocess — engineers all 12 features, creates train/val/test splits
python preprocess.py --ticker QQQ

# 3. EDA — inspect feature correlations, drop redundant ones in config.py
python eda.py --ticker QQQ

# 4. Rerun preprocess with trimmed feature set from config.py
python preprocess.py --ticker QQQ

# 5. Train
python train.py --ticker QQQ --model logreg

# 6. Evaluate — backtest on test split, generates plots
python evaluate.py --ticker QQQ --model logreg

# 7. Portfolio simulation — uses default_model set per ticker in config.py
python portfolio.py
```

---

## Configuration

Everything lives in `config.py`. To add a new ticker, copy and paste the following template and edit:

```python
"QQQ": {  # change to your target ticker

    "data": {
        "label_threshold": 0.003,  # min next-day return to label as "up"
        "train_ratio": 0.70,
        "val_ratio":   0.15,

        # Active features passed to the model.
        # Edit after running eda.py to drop redundant ones.
        "features": [
            "returns",
            "sma_ratio",
            "log_returns",
            "ema_ratio",
            "volatility_20",
            "rsi_14",
            "macd_hist",
            "bb_width",
            "bb_position",
            "atr_pct",
            "volume_ratio",
            "momentum_10",
        ],
    },

    "logreg": {
        "C": 0.1,  # tune as needed
        "max_iter": 1000,
        "class_weight": "balanced",
    },

    "randomforest": {  # tune or remove if not using
        "n_estimators": 200,
        "max_depth": 6,
        "min_samples_split": 20,
        "min_samples_leaf": 10,
        "max_features": "sqrt",
        "class_weight": "balanced",
        "random_state": 42,
    },

    "default_model": "logreg",  # model used by portfolio.py
},
```

Model hyperparameters are set per ticker in `config.py`. No touching the training code.

---

## Feature Universe

All features are engineered from raw OHLCV — no external data required. After running `eda.py`, redundant features (correlation > 0.85) are dropped from each ticker's config.

| Feature         | Description                               |
|-----------------|-------------------------------------------|
| `returns`       | Daily close-to-close return               |
| `log_returns`   | Log return (often redundant with returns) |
| `sma_ratio`     | Close / 20-day SMA                        |
| `ema_ratio`     | 12-day EMA / 26-day EMA                   |
| `volatility_20` | 20-day rolling return std                 |
| `rsi_14`        | 14-day Relative Strength Index            |
| `macd_hist`     | MACD histogram normalised by close        |
| `bb_width`      | Bollinger Band width                      |
| `bb_position`   | Close position within Bollinger Bands     |
| `atr_pct`       | ATR normalised by close                   |
| `volume_ratio`  | Volume / 20-day average volume            |
| `momentum_10`   | 10-day price momentum                     |

---

## Feature Selection

After running `eda.py`, the correlation heatmap and terminal summary identify 
redundant features. Anything above 0.85 correlation with another feature is a 
candidate to drop — keeping both would add noise without adding information.

![QQQ Correlation Heatmap](assets/qqq_correlation_heatmap.png)

```bash
python eda.py --ticker QQQ
```

![QQQ EDA terminal output](assets/qqq_eda_terminal.png)

For QQQ, four features were dropped after EDA:
- `log_returns` — 0.9997 correlation with `returns`
- `momentum_10` — 0.9167 correlation with `sma_ratio`
- `bb_position` — 0.9146 correlation with `rsi_14`
- `atr_pct` &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;— 0.9517 correlation with `volatility_20`

---

## Models

| Model          | Notes                                                                              |
|----------------|------------------------------------------------------------------------------------|
| `logreg`       | Logistic Regression — strong baseline, performs well on low-signal financial data  |
| `randomforest` | Random Forest — useful for non-linear regimes, tune carefully to avoid overfitting |

To add a new model (e.g. LightGBM): add its hyperparameters to `config.py` and a build branch in `train.py`'s `build_model()`. Nothing else changes.

---

## Portfolio Results

Simulation period: **May 2025 → May 2026** — test split only, data the models never saw during training.

| Metric            | ML Portfolio | Blended B&H |
|-------------------|--------------|-------------|
| Total Return      | **+31.07%**  | +37.92%     |
| Annualized Return | **+34.33%**  | +42.01%     |
| Sharpe Ratio      | **1.628**    | —           |
| Max Drawdown      | **-10.37%**  | —           |
| Days Invested     | **60.6%**    | 100%        |

The portfolio was in cash ~40% of the time (dark bands in the chart below), sitting out uncertain periods rather than holding through them. It captured most of the upside with meaningfully less market exposure and a Sharpe ratio of 1.628 — well above the 1.0 threshold considered strong.

![Portfolio Equity Curve](assets/portfolio_equity_curve.png)

The top panel shows the combined ML portfolio vs the blended buy-and-hold baseline. The bottom panel shows each ticker's individual model performance normalized to the same starting capital — useful for seeing which signals are contributing and which are dragging.

---

## Live Signal

Run after market close to get tomorrow's signal:

```bash
python predict.py --ticker QQQ --model logreg
```

![predict output](assets/predict_demo.png)

The `--confidence` flag sets the minimum probability threshold to act on a signal. Below it, the signal prints but is marked as not actionable.

---

## Project Structure

```
mint/
├── .gitignore
├── README.md
├── requirements.txt
├── assets/                   # screenshots and plots for README
├── config.py                 # single source of truth — all settings per ticker
├── backtest.py               # simulation engine (used by evaluate + portfolio)
├── eda.py                    # exploratory analysis + feature selection
├── evaluate.py               # backtest + performance plots
├── fetch.py                  # data download
├── portfolio.py              # multi-asset portfolio simulation
├── predict.py                # live signal generation
├── preprocess.py             # feature engineering + splits
├── train.py                  # model training
├── data/
│   ├── raw/                  # downloaded OHLCV
│   ├── splits/               # train / val / test CSVs
│   ├── processed/            # full processed dataset
│   └── plots/                # all generated plots
├── models/                   # saved model + scaler pairs (.pkl)
└── docs/
    ├── devlog.md             # development log
    └── models.md             # model notes and results
```

---

## Notes

- **No lookahead bias** — signals generated at close execute at the following day's open. The test split is never touched during training or feature selection.
- **Transaction costs** — portfolio simulation applies 0.1% per trade.
- **Labels** — binary classification: 1 if next-day return exceeds `label_threshold`, else 0. Threshold is tuned per ticker based on typical volatility.
- **Scaling** — `StandardScaler` fitted on train split only, applied to val and test.

---

## Dependencies

```bash
pip install -r requirements.txt
```

or manually:

```bash
pip install yfinance pandas numpy scikit-learn pandas-ta matplotlib seaborn joblib
```

Python 3.11+