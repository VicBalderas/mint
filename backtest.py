"""
backtest.py — Simulate model-driven trades on the test split.

Loads the test split + trained model, generates signals, then simulates
an all-in / all-out portfolio day by day. Predictions made at close are
executed at the following open — no same-bar execution.

Position sizing: 100% in on buy signal, 100% cash on sell/hold.
Data: test split only — the only data the model has never touched.

Returns a DataFrame with the full trade log and portfolio value over time,
consumed by evaluate.py for metrics and plots.

Usage:
    python backtest.py --ticker QQQ --model logreg
"""

import argparse
import os

import joblib
import pandas as pd
import numpy as np

from config import get_config


SPLITS_DIR = os.path.join(os.path.dirname(__file__), "data", "splits")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

INITIAL_CASH = 10_000.0


def load_test(ticker: str) -> pd.DataFrame:
    path = os.path.join(SPLITS_DIR, f"{ticker.upper()}_test.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"[backtest] Test split not found: {path}\n         Run preprocess.py first.")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    print(f"[backtest] Loaded test split — {len(df)} rows  |  {df.index[0].date()} → {df.index[-1].date()}")
    return df


def load_model(ticker: str, model_name: str):
    model_path  = os.path.join(MODELS_DIR, f"{ticker.upper()}_{model_name}_model.pkl")
    scaler_path = os.path.join(MODELS_DIR, f"{ticker.upper()}_{model_name}_scaler.pkl")

    for path in (model_path, scaler_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"[backtest] Model file not found: {path}\n           Run train.py first.")

    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print(f"[backtest] Loaded {model_name} model + scaler for {ticker.upper()}")
    return model, scaler


def generate_signals(df: pd.DataFrame, model, scaler, ticker: str) -> pd.Series:
    # Model sees only active features — same columns used during training
    feature_cols = get_config(ticker, "data")["features"]
    X = scaler.transform(df[feature_cols].values)
    signals = model.predict(X)
    return pd.Series(signals, index=df.index, name="signal")


def simulate(df: pd.DataFrame, signals: pd.Series) -> pd.DataFrame:
    """
    Simulate day-by-day portfolio. Signal from day N's close executes at day N+1's open.
    All-in on buy (1), all-cash on sell/hold (0). No shorting.
    """
    cash     = INITIAL_CASH
    shares   = 0.0
    in_trade = False
    rows     = []

    for i in range(len(df) - 1):
        today    = df.index[i]
        tomorrow = df.index[i + 1]
        signal   = signals.iloc[i]
        exec_price = df["open"].iloc[i + 1]  # execute at next day's open

        action = "hold"

        if signal == 1 and not in_trade:
            # Buy — go all in at tomorrow's open
            shares   = cash / exec_price
            cash     = 0.0
            in_trade = True
            action   = "buy"

        elif signal == 0 and in_trade:
            # Sell — liquidate at tomorrow's open
            cash     = shares * exec_price
            shares   = 0.0
            in_trade = False
            action   = "sell"

        portfolio_value = cash + shares * df["close"].iloc[i + 1]

        rows.append({
            "date":            tomorrow,
            "signal":          signal,
            "action":          action,
            "exec_price":      exec_price if action in ("buy", "sell") else np.nan,
            "shares":          shares,
            "cash":            cash,
            "close":           df["close"].iloc[i + 1],
            "portfolio_value": portfolio_value,
        })

    result = pd.DataFrame(rows).set_index("date")

    # Compute buy-and-hold baseline — buy at first open, hold through test period
    bh_shares = INITIAL_CASH / df["open"].iloc[0]
    result["buy_and_hold"] = bh_shares * result["close"]

    return result


def print_summary(results: pd.DataFrame) -> None:
    start_val = INITIAL_CASH
    end_val   = results["portfolio_value"].iloc[-1]
    bh_end    = results["buy_and_hold"].iloc[-1]

    model_return = (end_val - start_val) / start_val * 100
    bh_return    = (bh_end  - start_val) / start_val * 100

    n_trades = (results["action"] == "buy").sum()

    print(f"\n{'─' * 60}")
    print(f"  Backtest Summary")
    print(f"{'─' * 60}")
    print(f"  Period       : {results.index[0].date()} → {results.index[-1].date()}")
    print(f"  Start        : ${start_val:,.2f}")
    print(f"  Model end    : ${end_val:,.2f}  ({model_return:+.2f}%)")
    print(f"  B&H end      : ${bh_end:,.2f}  ({bh_return:+.2f}%)")
    print(f"  Trades       : {n_trades}")
    print(f"{'─' * 60}\n")


def run(ticker: str, model_name: str) -> pd.DataFrame:
    print(f"\n{'─' * 60}")
    print(f"  MINT Backtest — {ticker.upper()} — {model_name.upper()}")
    print(f"{'─' * 60}\n")

    df      = load_test(ticker)
    model, scaler = load_model(ticker, model_name)
    signals = generate_signals(df, model, scaler, ticker)
    results = simulate(df, signals)

    print_summary(results)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, required=True)
    parser.add_argument("--model",  type=str, required=True, choices=["logreg", "randomforest"])
    args = parser.parse_args()

    results = run(ticker=args.ticker, model_name=args.model)
    print(results[["action", "portfolio_value", "buy_and_hold"]].tail(10))