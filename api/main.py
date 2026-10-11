"""Step 5 — Serve the model as a REST API with FastAPI.

Run locally:  uvicorn api.main:app --reload
Then open:    http://127.0.0.1:8000/docs   (interactive Swagger page to try it)

Endpoints:
  GET  /health         -> is the service up and is the model loaded?
  GET  /model-info     -> which model is serving and its test-set scores
  POST /predict        -> score one transaction
  POST /predict/batch  -> score up to 1000 transactions at once
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from api.schemas import BatchRequest, BatchResponse, Prediction, Transaction
from src.predict import load_metrics, load_model, predict_transaction

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("fraud-api")

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model ONCE when the server starts, not on every request (much faster)
    try:
        state["model"] = load_model()
        state["metrics"] = load_metrics()
        logger.info("Model loaded: %s", state["metrics"].get("best_model"))
    except FileNotFoundError as err:
        logger.error(err)
        state["model"] = None
    yield
    state.clear()


app = FastAPI(
    title="Fraud Detection API",
    description="Scores card transactions for fraud risk using a trained ML pipeline.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log every request with its status code and how long it took."""
    start = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - start) * 1000
    logger.info("%s %s -> %s (%.1f ms)", request.method, request.url.path, response.status_code, ms)
    return response


def get_model():
    if state.get("model") is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run: python -m src.train")
    return state["model"]


@app.get("/")
def root():
    return {"message": "Fraud Detection API is running. Open /docs to try it."}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": state.get("model") is not None}


@app.get("/model-info")
def model_info():
    get_model()
    metrics = state["metrics"]
    best = metrics["best_model"]
    return {
        "model": best,
        "trained_at": metrics["trained_at"],
        "decision_threshold": metrics["decision_threshold"],
        "test_metrics": metrics["results"][best],
        "all_models_compared": {name: m["roc_auc"] for name, m in metrics["results"].items()},
    }


@app.post("/predict", response_model=Prediction)
def predict(transaction: Transaction):
    return predict_transaction(get_model(), transaction.model_dump())


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(request: BatchRequest):
    model = get_model()
    predictions = [predict_transaction(model, t.model_dump()) for t in request.transactions]
    return {"predictions": predictions, "flagged_count": sum(p["is_fraud"] for p in predictions)}
