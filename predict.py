"""
predict.py
----------
MINT — Machine Intelligence for Trading

Generates a trading signal for the next trading day using the trained model.
Fetches the most recent market data, engineers features, and runs the model
on today's close to produce a Buy/Sell signal for tomorrow's open.

Run this after market close each day to get tomorrow's signal.

Usage:
    python predict.py --ticker QQQ --model logreg
    python predict.py --ticker QQQ --model logreg --confidence 0.60
"""

import argparse
import os
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf

from config import get_config


MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# Lookback window — how many days of history to fetch for feature computation.
# Must be longer than the longest rolling window in feature engineering (20 days).
# 60 days gives comfortable warmup headroom.
LOOKBACK_DAYS = 60

DEFAULT_CONFIDENCE = 0.55   # minimum confidence to act on a signal


# ─── Data ─────────────────────────────────────────────────────────────────────

def fetch_recent(ticker: str, lookback_days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    """
    Fetch the most recent OHLCV data for feature computation.
    Only the final row (today) will be used for prediction — the rest
    is needed to compute rolling indicators without NaNs.

    Parameters
    ----------
    ticker : str
        Ticker symbol (e.g. 'QQQ').
    lookback_days : int
        Number of calendar days to fetch for warmup.

    Returns
    -------
    pd.DataFrame
        Recent OHLCV DataFrame sorted oldest → newest.
    """
    end   = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    print(f"[predict] Fetching recent data for {ticker.upper()} ({start} → {end})...")
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    if df.empty:
        raise RuntimeError(f"[predict] No data returned for {ticker}. Check ticker and connection.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    df.index   = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    print(f"[predict] Fetched {len(df)} rows  |  latest close: {df.index[-1].date()}")
    return df


# ─── Features ─────────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer the full feature universe — same logic as preprocess.py.
    Must stay in sync with preprocess.py at all times.
    Only the final row is used for prediction.
    """
    df["returns"]     = df["close"].pct_change()
    df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

    sma_20 = ta.sma(df["close"], length=20)
    ema_12 = ta.ema(df["close"], length=12)
    ema_26 = ta.ema(df["close"], length=26)
    df["sma_ratio"] = df["close"] / sma_20
    df["ema_ratio"] = ema_12 / ema_26

    df["volatility_20"] = df["returns"].rolling(window=20).std()
    df["rsi_14"]        = ta.rsi(df["close"], length=14)

    macd_data       = ta.macd(df["close"], fast=12, slow=26, signal=9)
    df["macd_hist"] = macd_data["MACDh_12_26_9"] / df["close"]

    bbands            = ta.bbands(df["close"], length=20, std=2)
    bb_upper          = bbands["BBU_20_2.0_2.0"]
    bb_lower          = bbands["BBL_20_2.0_2.0"]
    bb_range          = bb_upper - bb_lower
    df["bb_width"]    = bb_range / bbands["BBM_20_2.0_2.0"]
    df["bb_position"] = (df["close"] - bb_lower) / bb_range

    df["atr_pct"]      = ta.atr(df["high"], df["low"], df["close"], length=14) / df["close"]
    df["volume_ratio"] = df["volume"] / ta.sma(df["volume"], length=20)
    df["momentum_10"]  = df["close"] / df["close"].shift(10) - 1

    return df


def get_latest_features(df: pd.DataFrame, ticker: str) -> tuple[np.ndarray, pd.Timestamp]:
    """
    Extract the most recent row's active features for prediction.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with full feature universe engineered.
    ticker : str
        Ticker symbol — used to look up active features from config.

    Returns
    -------
    tuple of (X, latest_date)
        X           : np.ndarray of shape (1, n_features)
        latest_date : the date of the most recent row (today's close)
    """
    active_features = get_config(ticker, "data")["features"]

    # Drop NaNs from rolling warmup, take the last row
    df = df.dropna(subset=active_features)

    if df.empty:
        raise RuntimeError(
            "[predict] No valid rows after dropping NaNs. "
            "Try increasing LOOKBACK_DAYS."
        )

    latest      = df.iloc[-1]
    latest_date = df.index[-1]
    X           = latest[active_features].values.reshape(1, -1)

    return X, latest_date


# ─── Model ────────────────────────────────────────────────────────────────────

def load_model(ticker: str, model_name: str):
    """
    Load trained model and paired scaler from models/.

    Parameters
    ----------
    ticker : str
        Ticker symbol.
    model_name : str
        Model identifier (e.g. 'logreg').

    Returns
    -------
    tuple of (model, scaler)
    """
    model_path  = os.path.join(MODELS_DIR, f"{ticker.upper()}_{model_name}_model.pkl")
    scaler_path = os.path.join(MODELS_DIR, f"{ticker.upper()}_{model_name}_scaler.pkl")

    for path in (model_path, scaler_path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"[predict] Model file not found: {path}\n"
                f"          Run train.py first."
            )

    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print(f"[predict] Loaded {model_name} model + scaler for {ticker.upper()}")
    return model, scaler


def predict(model, scaler, X: np.ndarray) -> tuple[int, float, float]:
    """
    Generate a signal and confidence from the model.

    Parameters
    ----------
    model : sklearn estimator
        Trained model.
    scaler : StandardScaler
        Fitted scaler — must be the one paired with this model.
    X : np.ndarray
        Feature vector of shape (1, n_features).

    Returns
    -------
    tuple of (signal, up_prob, down_prob)
        signal   : 1 = Buy, 0 = Sell/Hold
        up_prob  : model's confidence in upward move
        down_prob: model's confidence in downward move
    """
    X_scaled  = scaler.transform(X)
    signal    = int(model.predict(X_scaled)[0])
    proba     = model.predict_proba(X_scaled)[0]
    down_prob = float(proba[0])
    up_prob   = float(proba[1])
    return signal, up_prob, down_prob


# ─── Output ───────────────────────────────────────────────────────────────────

def print_signal(
    ticker: str,
    model_name: str,
    signal: int,
    up_prob: float,
    down_prob: float,
    latest_date: pd.Timestamp,
    confidence_threshold: float,
    active_features: list[str],
    X: np.ndarray,
) -> None:
    """Print a formatted signal report to the terminal."""

    action     = "BUY" if signal == 1 else "SELL / HOLD"
    confidence = up_prob if signal == 1 else down_prob
    actionable = confidence >= confidence_threshold

    print(f"\n{'═' * 50}")
    print(f"  MINT Signal — {ticker.upper()} — {latest_date.date()}")
    print(f"{'═' * 50}")
    print(f"  Model      : {model_name.upper()}")
    print(f"  Signal     : {action}")
    print(f"  Confidence : {confidence * 100:.1f}%  "
          f"(threshold: {confidence_threshold * 100:.0f}%)")
    print(f"  Actionable : {'✅ YES — act on this signal' if actionable else '⚠️  NO — below confidence threshold'}")
    print(f"{'─' * 50}")
    print(f"  Up prob    : {up_prob * 100:.1f}%")
    print(f"  Down prob  : {down_prob * 100:.1f}%")
    print(f"{'─' * 50}")

    # Feature values used for this prediction
    print(f"  Features used ({latest_date.date()}):")
    for feat, val in zip(active_features, X[0]):
        print(f"    {feat:<20} {val:.6f}")

    print(f"{'─' * 50}")
    if signal == 1 and actionable:
        print(f"  → Enter LONG position at {ticker.upper()} open tomorrow")
    elif signal == 0 and actionable:
        print(f"  → Exit or stay in CASH — no position tomorrow")
    else:
        print(f"  → Signal below confidence threshold — no action recommended")
    print(f"{'═' * 50}\n")


# ─── Pipeline ─────────────────────────────────────────────────────────────────

def run(
    ticker: str,
    model_name: str,
    confidence_threshold: float = DEFAULT_CONFIDENCE,
) -> dict:
    """
    Full prediction pipeline:
    fetch recent data → engineer features → load model → predict → print signal.

    Parameters
    ----------
    ticker : str
        Ticker symbol (e.g. 'QQQ').
    model_name : str
        Model identifier (e.g. 'logreg').
    confidence_threshold : float
        Minimum confidence to mark signal as actionable. Default: 0.55.

    Returns
    -------
    dict
        Signal result with keys: signal, up_prob, down_prob, date, actionable.
    """
    print(f"\n{'─' * 50}")
    print(f"  MINT Predict — {ticker.upper()} — {model_name.upper()}")
    print(f"{'─' * 50}\n")

    # Fetch and engineer
    df = fetch_recent(ticker)
    df = engineer_features(df)

    # Get latest feature row
    active_features = get_config(ticker, "data")["features"]
    X, latest_date  = get_latest_features(df, ticker)

    # Load model and predict
    model, scaler            = load_model(ticker, model_name)
    signal, up_prob, down_prob = predict(model, scaler, X)

    # Print signal report
    print_signal(
        ticker, model_name, signal, up_prob, down_prob,
        latest_date, confidence_threshold, active_features, X
    )

    confidence = up_prob if signal == 1 else down_prob
    return {
        "ticker":     ticker.upper(),
        "model":      model_name,
        "date":       latest_date.date(),
        "signal":     signal,
        "up_prob":    up_prob,
        "down_prob":  down_prob,
        "confidence": confidence,
        "actionable": confidence >= confidence_threshold,
    }


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MINT — Generate a trading signal for the next trading day."
    )
    parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Ticker symbol (e.g. QQQ)."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["logreg", "randomforest"],
        help="Model to use for prediction."
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=DEFAULT_CONFIDENCE,
        help=f"Minimum confidence threshold to act on signal (default: {DEFAULT_CONFIDENCE})."
    )
    args = parser.parse_args()

    run(
        ticker=args.ticker,
        model_name=args.model,
        confidence_threshold=args.confidence,
    )