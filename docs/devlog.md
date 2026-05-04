# MINT — Development Log
**Machine Intelligence for Trading**
A running log of decisions, fixes, and progress across the project.

---

## Project Goal

Build an end-to-end ML pipeline that learns from historical market data and simulates
model-driven portfolio decisions under realistic conditions. Focus is on system design
and applied ML — not guaranteed financial performance.

---

## Status

> **Current phase:** Complete. Multi-asset portfolio simulation live.
> **Tickers:** QQQ, NVDA, AAPL
> **Next step (optional):** Add new tickers, experiment with LightGBM, external features (VIX, sector ETFs).

---

## Session Log

### Session 1 — Project Setup & Data Pipeline

#### Decisions

**Ticker: QQQ (single stock as Phase 2)**
Chose QQQ over SPY and individual stocks for the first model.
- SPY is the most analyzed instrument in the world — hardest to beat
- Individual stocks carry company-specific noise that price features can't capture
- QQQ tracks the Nasdaq 100 — tech-heavy, more volatile than SPY, cleaner than a stock

Phase 2: single stock model. Phase 3: a combined signal system where both models feed
a decision layer — long-term this could become a stacking ensemble or RL policy that
learns to allocate between the two dynamically.

---

**Date range: 2020-01-01 to present (~6.5 years)**
Captures COVID crash + recovery + bull run + 2022 correction + 2023–2026 regime.
Regime diversity matters for generalization. Produces ~1,539 usable rows after
preprocessing warmup.

---

**Problem framing: Binary classification**
Predict direction: up (1) or down/flat (0).
Cleaner to evaluate than regression, maps directly to Buy/Hold signal logic.

---

**Label threshold: > 0.3% forward return = label 1 (up)**
`> 0%` labels noise as signal — flat days near zero are unpredictable.
0.3% filters low-conviction moves without being too aggressive.
Label balance: **43.7% up / 56.3% down** — mild imbalance, manageable with `class_weight='balanced'`.

---

**Train/Val/Test split: 70 / 15 / 15 (chronological, no shuffling)**
70% over 60% because 60% train (~920 rows) was too lean. 70% gives ~1,077 rows.
Time order always preserved — shuffling causes data leakage.

Split result:
- Train : 1,089 rows (2020-02-20 → 2024-06-17)
- Val   :   233 rows (2024-06-18 → 2025-05-22)
- Test  :   234 rows (2025-05-23 → 2026-04-29)

---

**Model plan (scikit-learn first)**
Starting with classical ML before PyTorch. Planned progression:
1. Logistic Regression — establish baseline
2. Random Forest — first real model, use feature importances to refine features
3. LightGBM — workhorse once pipeline is solid

PyTorch (LSTM / Transformer) comes later once the full pipeline is validated.
Starting with simpler models surfaces feature and framing issues faster.

---

**Backtesting: custom engine**
Decided against `backtrader` — too heavy for what's needed here.
A simple custom engine gives full control and is easier to reason about.

---

#### Folder Structure

```
mint/
├── data/
│   ├── raw/          ← yfinance output (QQQ.csv)
│   ├── processed/    ← featured + labeled (QQQ.csv)
│   ├── splits/       ← QQQ_train.csv / QQQ_val.csv / QQQ_test.csv
│   └── plots/        ← EDA plots per ticker (data/plots/QQQ/)
├── docs/
│   ├── devlog.md     ← this file
│   └── models.md     ← saved model reference
├── models/           ← saved model + scaler .pkl files
├── config.py
├── fetch.py
├── preprocess.py
├── eda.py
├── train.py
├── backtest.py
├── evaluate.py
├── portfolio.py
├── predict.py
├── requirements.txt
└── README.md
```

---

#### Scripts Written

**`fetch.py`**
Fetches raw OHLCV data from yfinance. Validates output and saves to `data/raw/<TICKER>.csv`.
CLI + importable `run()` function.

**`preprocess.py`**
Full preprocessing pipeline:
`load → clean → engineer_features → create_labels → drop_nulls → select_features → split → save`
Auto-fetches raw data if not found. Engineers the full feature universe first, then filters
down to only the active features listed in `config.py` before saving splits.
Reruns overwrite existing files — no manual cleanup needed.
CLI + importable `run()` function.

Full feature universe engineered (all scale-invariant — no raw dollar amounts):
- `returns`, `log_returns`
- `sma_ratio` (close / sma_20), `ema_ratio` (ema_12 / ema_26)
- `volatility_20`
- `rsi_14`
- `macd_hist` (normalized by close)
- `bb_width`, `bb_position`
- `atr_pct` (atr / close)
- `volume_ratio` (volume / volume_sma_20)
- `momentum_10`

Active features for QQQ (post-EDA, 8 features):
`returns`, `sma_ratio`, `ema_ratio`, `volatility_20`, `rsi_14`, `macd_hist`, `bb_width`, `volume_ratio`

**`eda.py`**
Exploratory Data Analysis — run after preprocess.py, before train.py.
Reads active feature list from config.py per ticker. Generates 7 plots saved to
`data/plots/<TICKER>/` and prints a terminal summary with feature-label correlations,
redundancy flags, and a model recommendation.

Plots:
1. Label distribution + rolling 60-day up% (regime view)
2. Feature distributions by class (up vs down overlay)
3. Feature-feature correlation heatmap
4. Feature-label correlation bar chart
5. Class separation box-plots per feature
6. Cumulative returns + daily returns
7. Feature vs forward return (linearity check — straight line = linear model fits)

**`train.py`**
Trains a classification model on preprocessed splits. Loads splits, scales features
(fit on train only), builds model from config, evaluates on all three splits, saves
model + scaler as `.pkl` files. Supports multiple model types via `--model` flag.
CLI + importable `run()` function.

Supported models: `logreg`, `randomforest`

**`config.py`**
Single source of truth for all settings per ticker:
- `data` — label threshold, split ratios, active feature list
- `logreg` — Logistic Regression hyperparameters
- `randomforest` — Random Forest hyperparameters
- `default_model` — model used by portfolio.py for this ticker

Edit this file to tune — never touch `train.py` or `preprocess.py` for config changes.

---

#### Bugs Fixed

**Bollinger Bands KeyError (`BBU_20_2.0`)**
`pandas_ta` appends an extra `_2.0` to Bollinger Band column names in this environment.
Expected: `BBU_20_2.0` / Actual: `BBU_20_2.0_2.0`
Fix: updated all BB column references to use the `_2.0_2.0` suffix.

**LogisticRegression `penalty='l2'` deprecation warning**
`penalty` parameter deprecated in sklearn 1.8. Removed explicit `penalty='l2'` —
`lbfgs` uses L2 by default, behavior unchanged.

**Matplotlib boxplot `labels` deprecation warning**
`labels` parameter renamed to `tick_labels` in Matplotlib 3.9.
Fix: updated `plot_class_separation()` in `eda.py`.

---

#### Tech Notes

**PyTorch install — RTX 5080 (CUDA 13.1)**
RTX 5080 (Blackwell, sm_120) requires CUDA 12.8+. PyTorch doesn't publish `cu131`
wheels — use `cu130` (forward-compatible with 13.1).
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```
Plain `pip install torch` installs an incompatible CPU build.

---

#### What I Learned

- **Scale-invariant features matter.** Early versions included raw dollar-denominated
  values (sma_20, bb_upper, atr_14). These cause distribution shift — the model learns
  price levels instead of patterns and breaks on val/test from a different price regime.
  All features are now ratios or normalized values.

- **Accuracy ≠ profitability.** A model can be directionally correct and still lose money
  once you factor in transaction costs, slippage, and signal timing. Evaluation needs to
  go beyond classification metrics — hence the backtesting step.

- **Simpler models first.** Logistic Regression and Random Forest surface data and framing
  issues faster than a neural net would. Reaching for PyTorch before the pipeline is solid
  adds complexity without insight.

- **EDA before training saves time.** Running EDA after preprocessing revealed 4 redundant
  features (correlation > 0.85) that were adding noise. Dropping them improved LogReg
  val accuracy from 54.51% → 56.22%. Doing EDA first would have avoided training on a
  noisy feature set entirely.

- **Config-driven design pays off early.** Separating hyperparameters and feature lists
  into config.py means tuning never requires touching training logic. Adding a new ticker
  is a config change, not a code change.

---

### Session 2 — EDA, Feature Cleanup, Model Selection

#### What We Did

**Added `eda.py`**
Built a full EDA script that generates 7 diagnostic plots per ticker and prints a
terminal summary. Key finding for QQQ: all feature-label correlations are weak (max 0.07
for rsi_14), confirming the problem has low linear signal. This is normal for financial
data on a heavily analyzed instrument like QQQ.

---

**Discovered and removed redundant features**
EDA correlation heatmap revealed 15 high inter-feature correlation pairs (>0.7).
The worst offenders:

| Dropped Feature | Correlated With | Correlation |
|-----------------|-----------------|-------------|
| `log_returns`   | `returns`       | 0.9997      |
| `momentum_10`   | `sma_ratio`     | 0.9152      |
| `bb_position`   | `rsi_14`        | 0.9176      |
| `atr_pct`       | `volatility_20` | 0.9465      |

Removing these improved LogReg val accuracy from 54.51% → 56.22%.

---

**Made feature selection per-ticker via config.py**
`preprocess.py` now engineers the full feature universe every run, then filters down
to only the features listed in `config.py` under `data["features"]`.
`eda.py` reads the same list so plots always reflect the active feature set.

Workflow for a new ticker:
1. Run `fetch.py` — pull raw OHLCV data
2. Run `preprocess.py` — engineers all features, saves splits
3. Run `eda.py` — inspect correlations and redundancies
4. Update `config.py` features list — drop redundant ones
5. Rerun `preprocess.py` — overwrites splits with clean feature set
6. Run `train.py`

To refresh data for an existing ticker: rerun `fetch.py` (overwrites raw CSV),
then rerun `preprocess.py` (overwrites processed + splits). No manual cleanup needed.

---

#### Model Selection Results

**Final results — 8 clean features:**

| Model                   | Train  | Val        | Test   | Val F1 |
|-------------------------|--------|------------|--------|--------|
| Logistic Regression     | 53.54% | **56.22%** | 54.70% | 0.5622 |
| Random Forest (depth 6) | 78.05% | 50.64%     | 53.85% | 0.5081 |
| Random Forest (depth 5) | 73.55% | 48.50%     | 52.56% | 0.4890 |
| Random Forest (depth 4) | 68.32% | 48.93%     | 50.00% | 0.4932 |

**Winner: Logistic Regression**

Random Forest consistently overfits at every depth configuration — 25%+ gap between
train and val. The near-zero feature-label correlations (max 0.07) confirm the signal
structure is approximately linear. A complex non-linear model has nothing meaningful
to find and memorizes training noise instead.

Logistic Regression has almost no train/val gap (53.54% vs 56.22%) — it generalizes
cleanly and actually improves slightly from train to val, which is healthy.

Active QQQ model: `models/QQQ_logreg_model.pkl` + `models/QQQ_logreg_scaler.pkl`

Random Forest models kept for reference — may perform differently on Phase 2
single stock where signal structure is noisier and less linear.

---

### Session 2.5 — Live Signal Generation & Practical Utility

#### Context

After training and evaluating the QQQ model, the question came up: how would someone
actually use this in practice? The backtest shows historical performance but a real
user needs a way to query the model after market close each day and get a concrete
actionable signal for tomorrow.

This led to `predict.py` — a standalone script that bridges the research pipeline
and real-world usage.

---

**Added `predict.py`**
Fetches the most recent 60 days of OHLCV data (enough for rolling indicator warmup),
engineers the full feature universe using the same logic as `preprocess.py`, extracts
the latest row, scales it with the saved scaler, and runs the trained model to produce
a signal with confidence.

The 60-day lookback is intentional — rolling windows like `volatility_20` and `rsi_14`
need warmup rows to be accurate. Only the final row is actually used for prediction.

Output:
- Signal: BUY or SELL/HOLD
- Confidence: model's probability for the predicted direction
- Actionable flag: whether confidence meets the threshold (default 55%)
- Full feature breakdown for the prediction date — useful for transparency and debugging

The `--confidence` flag overrides the default threshold. Below threshold the signal
is printed but marked not actionable — no trade recommended.

---

#### Key Design Decision

`predict.py` is intentionally self-contained. It doesn't depend on any saved split
data — it fetches fresh market data directly. This means it works the same way
whether you're running it the day after training or a year later. The only
dependencies are the saved model and scaler `.pkl` files.

The output is designed to be plugged into any external execution system. MINT
doesn't handle order placement — it stops at the signal. What you do with that
signal is up to the user's own infrastructure.

---

#### What This Confirmed

The pipeline has a clean separation of concerns:
- `preprocess.py` + `train.py` → research and training
- `backtest.py` + `evaluate.py` → historical validation
- `predict.py` → live usage
- `portfolio.py` → multi-asset simulation

Each layer is independently usable. A user could run only `predict.py` daily
once models are trained, without ever touching the rest of the pipeline again.

---

### Session 3 — Multi-Ticker Expansion, Portfolio Simulation, Project Wrap

#### What We Did

**Expanded to NVDA and AAPL**
Ran the full pipeline (fetch → preprocess → eda → train) on two additional tickers.
Each ticker went through its own EDA to identify redundant features independently.

NVDA feature drops (post-EDA):
- `log_returns` — 0.9990 with `returns`
- `bb_position` — high correlation with `sma_ratio` and `rsi_14`
- `bb_width` — high correlation cluster
- `momentum_10` — 0.93+ with `sma_ratio`
- `volatility_20` kept over `atr_pct`

Final NVDA features (7): `returns`, `sma_ratio`, `ema_ratio`, `rsi_14`, `macd_hist`, `atr_pct`, `volume_ratio`

AAPL followed the same process with its own EDA pass and feature trimming.

---

**Model experiments on NVDA — confirmed linear signal structure**
Ran a key experiment: reduced Random Forest `max_depth` from 6 → 1 (decision stump).
Performance matched Logistic Regression almost exactly. Conclusion: forcing the tree
to behave linearly recovers the same performance as a linear model, confirming the
signal structure is approximately linear for these tickers. A deep tree has nothing
non-linear to find — it overfits noise instead.

This is a known pattern in financial ML: simpler models with regularization often
outperform complex ones on noisy tabular data.

---

**Why not LSTM or Transformers?**
Considered but decided against for this project:
- The bottleneck is signal quality, not model capacity
- EDA confirmed weak linear correlations — more compute on the same features
  won't move the needle meaningfully
- LightGBM would be the next reasonable step if pushing further
- LSTM/Transformer complexity is not justified at this stage

---

**Added `portfolio.py` — multi-asset portfolio simulation**
Combines signals from all configured tickers into a single portfolio simulation.

Logic:
- Rebalances every 5 trading days (~weekly)
- On each rebalance: if N tickers signal Buy → split capital evenly among them
- If no signals → move fully to cash
- 0.1% transaction cost applied per trade
- Compared against a blended equal-weight buy-and-hold baseline

No lookahead bias — signals from backtest.py already execute at next day's open.
No additional shift needed.

`default_model` field added to each ticker's config block — `portfolio.py` reads
this to know which model to use per ticker without requiring CLI arguments.

---

#### Portfolio Results — QQQ + NVDA + AAPL (May 2025 → May 2026)

| Metric            | ML Portfolio | Blended B&H |
|-------------------|--------------|-------------|
| Total Return      | +31.07%      | +37.92%     |
| Annualized Return | +34.33%      | +42.01%     |
| Sharpe Ratio      | 1.628        | —           |
| Max Drawdown      | -10.37%      | —           |
| Days Invested     | 60.6%        | 100%        |

The portfolio was in cash ~40% of the time, avoiding uncertain periods.
Sharpe of 1.628 is well above the 1.0 threshold considered strong.
Captured most of the upside with significantly less market exposure.

Earlier experiment with TSLA instead of AAPL produced worse results — TSLA's
high volatility made it harder to model cleanly with technical indicators alone.
AAPL is more institutionally driven and produced cleaner signals.

---

#### What I Learned

- **Not every ticker is equally modelable.** TSLA with the same feature set and
  model produced poor buy signal recall — the model defaulted to predicting down
  most of the time. High volatility doesn't mean more signal, it often means more
  noise. Ticker selection matters as much as model selection.

- **EDA weak linear signal ≠ linear model loses.** Logistic Regression outperformed
  Random Forest even when EDA flagged weak linear correlations. EDA measures individual
  feature-label relationships — logistic regression finds a linear boundary in the
  combined feature space, which can be meaningful even when no single feature is
  strongly predictive.

- **Portfolio diversification smooths results.** Individual ticker backtests showed
  NVDA losing money with the model while QQQ gained. Combined in a portfolio, the
  signals partially cancel each other's bad calls. Uncorrelated errors average out.

- **Sharpe ratio tells a better story than raw return.** +31% vs +37.9% sounds like
  underperformance. But 1.628 Sharpe with -10.37% max drawdown while being invested
  only 60% of the time is a genuinely different risk profile — not just a worse return.

- **The pipeline design paid off.** Adding NVDA and AAPL required zero changes to any
  script. Config entry → run pipeline → done. That's the right abstraction.

---

## Notes & Reminders

- Always use `class_weight='balanced'` — label split is roughly 44% up / 56% down
- Never shuffle splits — time order must be preserved to avoid leakage
- `forward_return` is computed for labeling only and immediately dropped
- Raw data preserved in `data/raw/` so preprocessing can rerun without re-fetching
- All scripts follow the same pattern: importable `run()` + CLI via `__main__`
- Run EDA before training on any new ticker — saves time, informs feature selection
- Feature list per ticker lives in `config.py` under `data["features"]`
- To refresh data: rerun `fetch.py` then `preprocess.py` — both overwrite existing files
- `default_model` must be set in config.py before running `portfolio.py`