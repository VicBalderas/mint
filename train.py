"""
train.py — Train a classification model on preprocessed split data.

Hyperparameters are read from config.py — don't edit them here.

Supported models:
    logreg        — Logistic Regression (baseline)
    randomforest  — Random Forest

Usage:
    python train.py --ticker QQQ --model logreg
    python train.py --ticker QQQ --model randomforest
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

from config import get_config


SPLITS_DIR = os.path.join(os.path.dirname(__file__), "data", "splits")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# Columns excluded from model input — raw prices + target
NON_FEATURE_COLS = {"open", "high", "low", "close", "volume", "label"}


def load_splits(ticker: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    splits = {}
    for name in ("train", "val", "test"):
        path = os.path.join(SPLITS_DIR, f"{ticker.upper()}_{name}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"[train] Split not found: {path}\n        Run preprocess.py first.")
        splits[name] = pd.read_csv(path, index_col=0, parse_dates=True)
        print(f"[train] Loaded {name:>5} — {len(splits[name])} rows")
    return splits["train"], splits["val"], splits["test"]


def get_xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    return df[feature_cols].values, df["label"].values, feature_cols


def build_model(model_name: str, ticker: str):
    params = get_config(ticker, model_name)

    if model_name == "logreg":
        print(f"[train] Logistic Regression  |  C={params['C']}  |  class_weight={params['class_weight']}")
        return LogisticRegression(
            C=params["C"],
            max_iter=params["max_iter"],
            class_weight=params["class_weight"],
            solver="lbfgs",
            random_state=42,
        )

    if model_name == "randomforest":
        print(f"[train] Random Forest  |  n_estimators={params['n_estimators']}  |  max_depth={params['max_depth']}")
        return RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            class_weight=params["class_weight"],
            random_state=params["random_state"],
            n_jobs=-1,
        )

    raise ValueError(f"[train] Unknown model: '{model_name}'. Supported: logreg, randomforest.")


def evaluate(model, X: np.ndarray, y: np.ndarray, split_name: str) -> dict:
    preds = model.predict(X)
    acc   = accuracy_score(y, preds)
    f1    = f1_score(y, preds, average="weighted")

    print(f"\n{'─' * 50}  {split_name}")
    print(f"  Accuracy : {acc * 100:.2f}%  |  F1: {f1:.4f}")
    print(classification_report(y, preds, target_names=["Down/Flat", "Up"], zero_division=0))
    print(f"  Confusion Matrix:\n  {confusion_matrix(y, preds)}\n")

    return {"accuracy": acc, "f1": f1}


def print_importances(model, feature_cols: list[str], top_n: int = 10) -> None:
    if not hasattr(model, "feature_importances_"):
        return
    ranked = sorted(zip(feature_cols, model.feature_importances_), key=lambda x: x[1], reverse=True)
    print(f"\n{'─' * 50}  Feature Importances (top {top_n})")
    for name, score in ranked[:top_n]:
        print(f"  {name:<20} {score:.4f}  {'█' * int(score * 200)}")


def run(ticker: str, model_name: str) -> dict:
    print(f"\n{'─' * 60}")
    print(f"  MINT Training — {ticker.upper()} — {model_name.upper()}")
    print(f"{'─' * 60}\n")

    train_df, val_df, test_df = load_splits(ticker)

    X_train, y_train, feature_cols = get_xy(train_df)
    X_val,   y_val,   _            = get_xy(val_df)
    X_test,  y_test,  _            = get_xy(test_df)

    print(f"\n[train] {len(feature_cols)} features: {feature_cols}\n")

    scaler  = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    model = build_model(model_name, ticker)
    print(f"[train] Training on {len(y_train)} samples...")
    model.fit(X_train, y_train)

    train_m = evaluate(model, X_train, y_train, "Train")
    val_m   = evaluate(model, X_val,   y_val,   "Val")
    test_m  = evaluate(model, X_test,  y_test,  "Test")

    print_importances(model, feature_cols)

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model,  os.path.join(MODELS_DIR, f"{ticker.upper()}_{model_name}_model.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, f"{ticker.upper()}_{model_name}_scaler.pkl"))
    print(f"\n[train] Saved model + scaler to {MODELS_DIR}")

    print(f"\n{'─' * 60}")
    print(f"  Train: {train_m['accuracy']*100:.2f}%  |  Val: {val_m['accuracy']*100:.2f}%  |  Test: {test_m['accuracy']*100:.2f}%")
    print(f"  Val F1: {val_m['f1']:.4f}")
    print(f"{'─' * 60}\n")

    return val_m


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, required=True)
    parser.add_argument("--model",  type=str, required=True, choices=["logreg", "randomforest"])
    args = parser.parse_args()
    run(ticker=args.ticker, model_name=args.model)