"""Request and response shapes for the API.

Pydantic checks every incoming request automatically: if someone sends a
negative amount or hour=30, FastAPI rejects it with a clear 422 error before it
ever reaches the model.
"""

from typing import Literal

from pydantic import BaseModel, Field


class Transaction(BaseModel):
    amount: float = Field(..., gt=0, description="Transaction amount in ₹", examples=[18500.0])
    hour: int = Field(..., ge=0, le=23, description="Hour of day (0-23)", examples=[2])
    merchant_category: Literal["grocery", "restaurant", "fuel", "online", "electronics", "travel"] = Field(
        ..., examples=["electronics"]
    )
    is_international: int = Field(..., ge=0, le=1, examples=[1])
    card_present: int = Field(..., ge=0, le=1, description="1 = card swiped/tapped, 0 = online/phone", examples=[0])
    distance_from_home_km: float = Field(..., ge=0, examples=[250.0])
    txn_count_last_24h: int = Field(..., ge=0, examples=[7])
    account_age_days: int = Field(..., ge=0, examples=[20])
    avg_amount_30d: float = Field(..., ge=0, description="Customer's average spend over 30 days", examples=[2500.0])


class Prediction(BaseModel):
    fraud_probability: float
    is_fraud: bool
    risk_level: Literal["low", "medium", "high"]


class BatchRequest(BaseModel):
    transactions: list[Transaction] = Field(..., min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    predictions: list[Prediction]
    flagged_count: int
