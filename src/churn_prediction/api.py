"""
FastAPI for churn prediction.
"""
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .model import ChurnModel

app = FastAPI(
    title="Churn Prediction API",
    version="0.1.0",
)

MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "production"
model: Optional[ChurnModel] = None


@app.on_event("startup")
async def load_model():
    global model
    if MODEL_PATH.exists():
        model = ChurnModel.load(MODEL_PATH)
    else:
        print(f"Warning: No model found at {MODEL_PATH}")


class UserFeatures(BaseModel):
    n_sessions: int = Field(..., ge=0)
    n_songs: int = Field(..., ge=0)
    n_thumbs_up: int = Field(0, ge=0)
    n_thumbs_down: int = Field(0, ge=0)
    n_add_playlist: int = Field(0, ge=0)
    n_add_friend: int = Field(0, ge=0)
    n_errors: int = Field(0, ge=0)
    n_help: int = Field(0, ge=0)
    n_downgrade: int = Field(0, ge=0)
    n_adverts: int = Field(0, ge=0)
    days_active: int = Field(..., ge=1)
    total_listen_time: float = Field(..., ge=0)
    songs_per_session: float = Field(..., ge=0)
    thumbs_ratio: float = Field(0.5, ge=0, le=1)
    is_paid: int = Field(..., ge=0, le=1)
    is_male: int = Field(..., ge=0, le=1)


class PredictionResponse(BaseModel):
    churn_probability: float = Field(..., ge=0, le=1)
    risk_level: str


@app.get("/")
async def root():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(features: UserFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    df = pd.DataFrame([features.model_dump()])
    proba = model.predict(df)[0]

    if proba < 0.3:
        risk = "low"
    elif proba < 0.6:
        risk = "medium"
    else:
        risk = "high"

    return PredictionResponse(churn_probability=round(proba, 4), risk_level=risk)


@app.post("/predict/batch")
async def predict_batch(features_list: list[UserFeatures]):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    df = pd.DataFrame([f.model_dump() for f in features_list])
    probas = model.predict(df)

    results = []
    for p in probas:
        risk = "low" if p < 0.3 else ("medium" if p < 0.6 else "high")
        results.append({"churn_probability": round(float(p), 4), "risk_level": risk})

    return results
