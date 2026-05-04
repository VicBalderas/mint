"""
fetch.py — Fetch raw OHLCV data from yfinance and save to data/raw/.

Usage:
    python fetch.py --ticker AAPL --start 2018-01-01 --end 2024-01-01
    python fetch.py --ticker AAPL
"""

import argparse
import os
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf


RAW_DATA_DIR  = os.path.join(os.path.dirname(__file__), "data", "raw")
DEFAULT_END   = datetime.today().strftime("%Y-%m-%d")
DEFAULT_START = (datetime.today() - timedelta(days=365 * 5)).strftime("%Y-%m-%d")


def fetch(ticker: str, start: str, end: str) -> pd.DataFrame:
    print(f"[fetch] Fetching {ticker} from {start} to {end}...")

    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    if df.empty:
        print(f"[fetch] WARNING: No data returned for {ticker}.")
        return df

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [col.lower() for col in df.columns]
    df.index   = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    print(f"[fetch] {len(df)} rows  |  {df.index[0].date()} → {df.index[-1].date()}")
    return df


def validate(df: pd.DataFrame, ticker: str) -> bool:
    missing = {"open", "high", "low", "close", "volume"} - set(df.columns)
    if missing:
        print(f"[fetch] ERROR: Missing columns for {ticker}: {missing}")
        return False

    null_pct = df.isnull().mean().mean() * 100
    if null_pct > 5:
        print(f"[fetch] WARNING: {null_pct:.1f}% null values for {ticker}.")

    return True


def save(df: pd.DataFrame, ticker: str) -> str:
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    filepath = os.path.join(RAW_DATA_DIR, f"{ticker.upper()}.csv")
    df.to_csv(filepath)
    print(f"[fetch] Saved → {filepath}")
    return filepath


def run(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    df = fetch(ticker, start, end)
    if df.empty or not validate(df, ticker):
        return None
    save(df, ticker)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, required=True)
    parser.add_argument("--start",  type=str, default=DEFAULT_START)
    parser.add_argument("--end",    type=str, default=DEFAULT_END)
    args = parser.parse_args()

    result = run(args.ticker, args.start, args.end)
    if result is not None:
        print(f"[fetch] Done. Shape: {result.shape}")
        print(result.tail(3))
    else:
        print("[fetch] Fetch failed.")