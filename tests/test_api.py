"""Check the FastAPI endpoints using FastAPI's TestClient (no real server needed)."""

import pytest
from fastapi.testclient import TestClient

from api.main import app
from src import config
from src.train import run_training

SUSPICIOUS = {
    "amount": 18500.0, "hour": 2, "merchant_category": "electronics", "is_international": 1,
    "card_present": 0, "distance_from_home_km": 250.0, "txn_count_last_24h": 7,
    "account_age_days": 20, "avg_amount_30d": 2500.0,
}
NORMAL = {
    "amount": 450.0, "hour": 13, "merchant_category": "grocery", "is_international": 0,
    "card_present": 1, "distance_from_home_km": 3.0, "txn_count_last_24h": 1,
    "account_age_days": 1500, "avg_amount_30d": 600.0,
}


@pytest.fixture(scope="module")
def client():
    if not config.MODEL_PATH.exists():
        run_training()
    with TestClient(app) as c:  # "with" runs the startup code that loads the model
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_model_info(client):
    body = client.get("/model-info").json()
    assert body["model"] in body["all_models_compared"]
    assert 0 <= body["test_metrics"]["roc_auc"] <= 1
    assert "amount_to_avg_ratio" in body["top_features"]  # my engineered feature matters


def test_predict_suspicious_vs_normal(client):
    risky = client.post("/predict", json=SUSPICIOUS).json()
    safe = client.post("/predict", json=NORMAL).json()
    assert risky["fraud_probability"] > safe["fraud_probability"]
    assert risky["is_fraud"] is True
    assert safe["is_fraud"] is False


def test_predict_rejects_bad_input(client):
    bad = {**NORMAL, "hour": 30}
    assert client.post("/predict", json=bad).status_code == 422


def test_batch_predict(client):
    response = client.post("/predict/batch", json={"transactions": [SUSPICIOUS, NORMAL]})
    body = response.json()
    assert response.status_code == 200
    assert len(body["predictions"]) == 2
    assert body["flagged_count"] == 1
