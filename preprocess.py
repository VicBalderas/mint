"""
preprocess.py — Clean, engineer features, label, and split raw OHLCV data.

Engineers ALL possible features first, then keeps only the ones listed
in config.py under data["features"] for the given ticker.

To change active features: edit config.py, rerun preprocess.py.

Saves to data/processed/ and data/splits/.

Usage:
    python preprocess.py --ticker QQQ --start 2020-01-01
    python preprocess.py --ticker QQQ
"""

import argparse
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pandas_ta as ta

import fetch as fetcher
from config import get_config


RAW_DATA_DIR  = os.path.join(os.path.dirname(__file__), "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "data", "processed")
SPLITS_DIR    = os.path.join(os.path.dirname(__file__), "data", "splits")

DEFAULT_END   = datetime.today().strftime("%Y-%m-%d")
DEFAULT_START = (datetime.today() - timedelta(days=365 * 7)).strftime("%Y-%m-%d")

OHLCV_COLS = ["open", "high", "low", "close", "volume"]


def load(ticker: str, start: str, end: str) -> pd.DataFrame:
    filepath = os.path.join(RAW_DATA_DIR, f"{ticker.upper()}.csv")

    if not os.path.exists(filepath):
        print(f"[preprocess] Raw data not found for {ticker}. Fetching...")
        df = fetcher.run(ticker=ticker, start=start, end=end)
        if df is None:
            raise RuntimeError(f"[preprocess] Failed to fetch data for {ticker}.")
        return df

    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    print(f"[preprocess] Loaded {len(df)} rows  |  {df.index[0].date()} → {df.index[-1].date()}")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df[~df.index.duplicated(keep="first")].dropna(how="all").sort_index()
    for col in OHLCV_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    print(f"[preprocess] Clean: {before} → {len(df)} rows")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the full feature universe — all scale-invariant, no raw dollar values.
    select_features() filters this down to the active set from config.py.
    """
    print("[preprocess] Engineering full feature universe...")

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

    print(f"[preprocess] Full universe: {len(df.columns)} columns")
    return df


def create_labels(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    threshold = get_config(ticker, "data")["label_threshold"]
    print(f"[preprocess] Creating labels (threshold: {threshold * 100:.1f}%)...")

    df["forward_return"] = df["close"].shift(-1) / df["close"] - 1
    df["label"]          = (df["forward_return"] > threshold).astype(int)
    df.drop(columns=["forward_return"], inplace=True)

    up_pct = df["label"].mean() * 100
    print(f"[preprocess] Labels: {up_pct:.1f}% up  |  {100 - up_pct:.1f}% down/flat")
    return df


def select_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    # Keeps OHLCV (for reference) + active features + label. Drops the rest.
    active = get_config(ticker, "data")["features"]
    all_engineered = [c for c in df.columns if c not in OHLCV_COLS + ["label"]]

    missing = [f for f in active if f not in all_engineered]
    if missing:
        raise ValueError(f"[preprocess] Features in config missing from engineered set: {missing}\n"
                         f"             Available: {all_engineered}")

    dropped = [c for c in all_engineered if c not in active]
    df = df[OHLCV_COLS + active + ["label"]]
    print(f"[preprocess] Keeping {len(active)} features, dropping {len(dropped)}: {dropped}")
    return df


def split(df: pd.DataFrame, ticker: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg       = get_config(ticker, "data")
    n         = len(df)
    train_end = int(n * cfg["train_ratio"])
    val_end   = int(n * (cfg["train_ratio"] + cfg["val_ratio"]))

    train, val, test = df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]
    print(f"[preprocess] Split →"
          f"\n  Train : {len(train):>4} rows  ({train.index[0].date()} → {train.index[-1].date()})"
          f"\n  Val   : {len(val):>4} rows  ({val.index[0].date()} → {val.index[-1].date()})"
          f"\n  Test  : {len(test):>4} rows  ({test.index[0].date()} → {test.index[-1].date()})")
    return train, val, test


def save(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, ticker: str) -> None:
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(SPLITS_DIR, exist_ok=True)

    pd.concat([train, val, test]).to_csv(os.path.join(PROCESSED_DIR, f"{ticker.upper()}.csv"))
    print(f"[preprocess] Saved processed → {PROCESSED_DIR}/{ticker.upper()}.csv")

    for name, df in [("train", train), ("val", val), ("test", test)]:
        path = os.path.join(SPLITS_DIR, f"{ticker.upper()}_{name}.csv")
        df.to_csv(path)
        print(f"[preprocess] Saved {name} → {path}")


def run(ticker: str, start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print(f"\n{'─' * 60}")
    print(f"  MINT Preprocessing — {ticker.upper()}")
    print(f"{'─' * 60}\n")

    df = load(ticker, start, end)
    df = clean(df)
    df = engineer_features(df)
    df = create_labels(df, ticker)
    before = len(df)
    df = df.dropna()
    print(f"[preprocess] Drop NaNs: {before} → {len(df)} rows")
    df = select_features(df, ticker)

    train, val, test = split(df, ticker)
    save(train, val, test, ticker)

    active = get_config(ticker, "data")["features"]
    print(f"\n[preprocess] Done.  Rows: {len(df)}  |  Features: {len(active)}  |  "
          f"Labels: {df['label'].mean() * 100:.1f}% up\n")
    return train, val, test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, required=True)
    parser.add_argument("--start",  type=str, default=DEFAULT_START)
    parser.add_argument("--end",    type=str, default=DEFAULT_END)
    args = parser.parse_args()

    train, val, test = run(args.ticker, args.start, args.end)
    print(train.tail(3))