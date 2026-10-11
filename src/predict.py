"""Step 4 — Use the saved model to score a new transaction.

The API (api/main.py) calls these functions. Because the saved file is the
whole pipeline (preprocessing + model), a raw transaction goes in and a fraud
probability comes out — the exact same cleaning/scaling as during training.
"""

import json

import joblib
import pandas as pd

from src import config
from src.features import add_features


def load_model(path=config.MODEL_PATH):
    if not path.exists():
        raise FileNotFoundError(f"No trained model at {path}. Run: python -m src.train")
    return joblib.load(path)


def load_metrics(path=config.METRICS_PATH):
    return json.loads(path.read_text()) if path.exists() else {}


def risk_level(probability: float) -> str:
    if probability >= 0.7:
        return "high"
    if probability >= config.DECISION_THRESHOLD:
        return "medium"
    return "low"


def predict_transaction(model, transaction: dict, threshold: float = config.DECISION_THRESHOLD) -> dict:
    df = add_features(pd.DataFrame([transaction]))
    df = df[config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES]  # same columns, same order as training
    probability = float(model.predict_proba(df)[:, 1][0])
    return {
        "fraud_probability": round(probability, 4),
        "is_fraud": probability >= threshold,
        "risk_level": risk_level(probability),
    }
