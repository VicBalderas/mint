# MINT — Machine Intelligence for Trading

An end-to-end ML pipeline that learns from market data and simulates model-driven portfolio decisions under realistic conditions.

> **Status:** Active development — architecture and feature set are still being refined.

---

## Overview

MINT explores whether ML-driven trading decisions can grow a portfolio when evaluated under realistic constraints: no lookahead bias, execution-time-aligned predictions, and time-series-respecting validation.

The focus is on **system design and applied ML** — not financial advice or guaranteed returns.

---

## Pipeline

```
Data → Features → Model → Predictions → Signals → Portfolio Simulation
```

---

## Workflow

### 1. Data Collection
Historical OHLCV data fetched via `yfinance` (Open, High, Low, Close, Volume).

### 2. Feature Engineering
Raw price data is transformed into predictive features:
- Return series (log and simple)
- Moving averages (SMA, EMA)
- Volatility estimates (rolling std, ATR)
- Momentum indicators (RSI, MACD)

### 3. Dataset Splitting
Strict **time-based splits** to prevent data leakage:

| Split      | Purpose                          |
|------------|----------------------------------|
| Train      | Model learning                   |
| Validation | Hyperparameter tuning, iteration |
| Test       | Final unbiased evaluation        |

### 4. Model Training
Candidate models for predicting future price movement or returns:
- Random Forest
- Gradient Boosting (XGBoost / LightGBM)
- LSTM / Transformer (sequence models)

### 5. Evaluation & Iteration
Models are evaluated on validation data. The loop continues until the model demonstrates generalization — not just in-sample fit. Metrics tracked include directional accuracy, Sharpe ratio proxy, and drawdown.

### 6. Trading Signal Generation
Model outputs (probabilities or regression scores) are converted into discrete actions based on configurable thresholds and risk tolerance:
- **Buy** — model confidence exceeds entry threshold
- **Sell** — model signals reversal or exit condition
- **Hold** — confidence below threshold; no action

### 7. Portfolio Backtesting
Simulated trading over historical data using generated signals:
- Trades execute at **next-day open** (no same-bar execution)
- Tracks cash, positions, and total portfolio value over time
- Enforces realistic slippage and transaction cost assumptions

---

## Key Design Principles

**No data leakage.** All splits are time-ordered. Features are computed using only past data at each point in time.

**Execution realism.** Predictions made at close are executed at the following open — matching how live systems would behave.

**Accuracy ≠ profitability.** A model can be directionally correct and still lose money. Evaluation goes beyond classification metrics.

**Generalization over fit.** The system is designed to surface overfitting early and penalize it across the evaluation loop.

---

## Tech Stack

| Layer            | Tools                                   |
|------------------|-----------------------------------------|
| Data             | `yfinance`, `pandas`                    |
| Feature Eng.     | `numpy`, `pandas`, `ta-lib`             |
| Modeling         | `scikit-learn`, `XGBoost`, `PyTorch`    |
| Backtesting      | Custom simulation engine                |
| Visualization    | `matplotlib`, `seaborn`                 |

---

## Roadmap

- [x] Pipeline architecture and design
- [ ] Feature engineering module
- [ ] Model training and evaluation loop
- [ ] Backtesting engine
- [ ] Multi-asset support
- [ ] Position sizing and risk management (stop-loss, Kelly criterion)
- [ ] Automated retraining pipeline
- [ ] Reinforcement learning extension

---

## Disclaimer

MINT is a **research and learning project** — not financial advice. Nothing here constitutes a recommendation to buy or sell any asset. Past simulated performance does not predict future results.
