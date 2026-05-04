"""
config.py — Single source of truth for all settings per ticker.
preprocess.py, eda.py, and train.py all read from here.

Workflow for a new ticker:
    1. Add ticker block below (copy QQQ as template)
    2. Run preprocess.py → engineers all features
    3. Run eda.py → inspect correlations and redundancies
    4. Update "features" list to drop redundant ones
    5. Rerun preprocess.py → splits saved with selected features
    6. Run train.py

To add a new model type:
    1. Add a sub-key under each ticker (e.g. "lightgbm")
    2. Add the corresponding build function in train.py's build_model()

12 features:
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

"QQQ": {

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
            "C": 0.1,
            "max_iter": 1000,
            "class_weight": "balanced",
        },

        "randomforest": {
            "n_estimators": 200,
            "max_depth": 6,
            "min_samples_split": 20,
            "min_samples_leaf": 10,
            "max_features": "sqrt",
            "class_weight": "balanced",
            "random_state": 42,
        },

        "default_model": "logreg",
},
"""

CONFIGS = {
    "QQQ": {
        "data": {
            "label_threshold": 0.003,  # min next-day return to label as "up"
            "train_ratio": 0.70,
            "val_ratio":   0.15,
            "features": [
                "returns",
                "sma_ratio",
                "ema_ratio",
                "volatility_20",
                "rsi_14",
                "macd_hist",
                "bb_width",
                "volume_ratio",
            ],
        },

        "logreg": {
            "C": 0.1,
            "max_iter": 1000,
            "class_weight": "balanced",
        },

        "default_model": "logreg",
    },

    "NVDA": {
        "data": {
            "label_threshold": 0.005,
            "train_ratio": 0.70,
            "val_ratio":   0.15,
            "features": [
                "returns",
                "sma_ratio",
                "ema_ratio",
                "rsi_14",
                "macd_hist",
                "atr_pct",
                "volume_ratio",
            ],
        },

        "logreg": {
            "C": 0.1,
            "max_iter": 1000,
            "class_weight": "balanced",
        },

        "default_model": "logreg",
    },

    "AAPL": {
        "data": {
            "label_threshold": 0.005,
            "train_ratio": 0.70,
            "val_ratio":   0.15,
            "features": [
                "returns",
                "sma_ratio",
                "ema_ratio",
                "volatility_20",
                "rsi_14",
                "macd_hist",
                "bb_width",
                "atr_pct",
                "volume_ratio",
            ],
        },
        "logreg": {
            "C": 1.0,
            "max_iter": 1000,
            "class_weight": "balanced",
        },
        "default_model": "logreg",
    },

}


def get_config(ticker: str, section: str) -> dict:
    ticker = ticker.upper()
    if ticker not in CONFIGS:
        raise KeyError(f"[config] No config for '{ticker}'. Available: {list(CONFIGS.keys())}")
    if section not in CONFIGS[ticker]:
        raise KeyError(f"[config] No '{section}' section for '{ticker}'. Available: {list(CONFIGS[ticker].keys())}")
    return CONFIGS[ticker][section]