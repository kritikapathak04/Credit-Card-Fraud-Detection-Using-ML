"""FastAPI service for the credit card fraud model.

Run from the project root (where models/best_model.joblib lives):

    pip install -r requirements-api.txt
    uvicorn api:app --reload --port 8000

Interactive docs: http://localhost:8000/docs

Assumption: models/best_model.joblib holds the fitted pipeline and the tuned
threshold (0.416). Dict, tuple and bare-estimator layouts are all handled. If
loading fails, check what train.py passes to joblib.dump().
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/best_model.joblib"))
ALLOW_ORIGINS = [o.strip() for o in os.getenv("ALLOW_ORIGINS", "*").split(",") if o.strip()]
MAX_BATCH = 10_000
HIGH_RISK = 0.70  # probability at or above this is "High" (the tuned threshold marks "Medium")

RAW_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
STATE: dict = {}


# ---------- model loading ----------
def _unpack(obj):
    """Return (pipeline, threshold) from whatever train.py saved."""
    if isinstance(obj, dict):
        pipe = next((obj[k] for k in ("pipeline", "model", "estimator", "clf") if k in obj), None)
        if pipe is None:
            raise ValueError(f"Unrecognised keys in model file: {list(obj)}")
        thr = next((float(obj[k]) for k in ("threshold", "best_threshold", "decision_threshold") if k in obj), 0.5)
        return pipe, thr
    if isinstance(obj, (tuple, list)) and len(obj) == 2:
        a, b = obj
        return (b, float(a)) if hasattr(b, "predict") else (a, float(b))
    return obj, float(getattr(obj, "threshold", 0.5))


@asynccontextmanager
async def lifespan(_: FastAPI):
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found: {MODEL_PATH.resolve()}. Run `python train.py` first.")
    STATE["pipe"], STATE["threshold"] = _unpack(joblib.load(MODEL_PATH))
    yield
    STATE.clear()


app = FastAPI(title="Credit Card Fraud Detection API", version="1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=ALLOW_ORIGINS, allow_methods=["*"], allow_headers=["*"])


# ---------- scoring ----------
def _frame(rows: List[Dict[str, float]]):
    df = pd.DataFrame(rows)
    filled = [c for c in RAW_COLUMNS if c not in df.columns]
    for c in filled:
        df[c] = 0.0
    df = df[RAW_COLUMNS].astype(float).fillna(0.0)
    df["Hour"] = np.floor(df["Time"] / 3600) % 24
    df["Log_Amount"] = np.log1p(df["Amount"].clip(lower=0))
    return df, filled


def _proba(pipe, X) -> np.ndarray:
    if hasattr(pipe, "predict_proba"):
        return pipe.predict_proba(X)[:, 1]
    if hasattr(pipe, "decision_function"):  # fall back to a squashed score
        return 1 / (1 + np.exp(-pipe.decision_function(X)))
    raise TypeError("Model has neither predict_proba nor decision_function")


def _score(rows):
    pipe = STATE["pipe"]
    df, filled = _frame(rows)
    expected = getattr(pipe, "feature_names_in_", None)
    candidates = [df[list(expected)]] if expected is not None else [df, df[RAW_COLUMNS]]
    last = None
    for X in candidates:
        try:
            return _proba(pipe, X), filled
        except Exception as exc:  # try the next column layout
            last = exc
    raise HTTPException(500, f"Model rejected the input: {last}")


def _risk(p: float) -> str:
    return "High" if p >= max(HIGH_RISK, STATE["threshold"]) else "Medium" if p >= STATE["threshold"] else "Low"


# ---------- schemas ----------
class PredictRequest(BaseModel):
    features: Dict[str, float] = Field(
        ..., description="Any of Time, V1..V28, Amount. Missing ones are filled with 0."
    )


class BatchRequest(BaseModel):
    rows: List[Dict[str, float]] = Field(..., max_length=MAX_BATCH)


class Prediction(BaseModel):
    fraud_probability: float
    is_fraud: bool
    risk_level: str
    threshold: float
    missing_features_filled: List[str]


# ---------- routes ----------
@app.get("/health")
def health():
    pipe = STATE["pipe"]
    exp = getattr(pipe, "feature_names_in_", None)
    return {
        "status": "ok",
        "model": type(pipe).__name__,
        "threshold": STATE["threshold"],
        "expected_columns": list(exp) if exp is not None else None,
    }


@app.post("/predict", response_model=Prediction)
def predict(req: PredictRequest):
    p, filled = _score([req.features])
    p = float(p[0])
    return Prediction(
        fraud_probability=p,
        is_fraud=p >= STATE["threshold"],
        risk_level=_risk(p),
        threshold=STATE["threshold"],
        missing_features_filled=filled,
    )


@app.post("/predict/batch")
def predict_batch(req: BatchRequest):
    if not req.rows:
        raise HTTPException(422, "rows must not be empty")
    probs, filled = _score(req.rows)
    thr = STATE["threshold"]
    preds = [
        {"fraud_probability": float(p), "is_fraud": bool(p >= thr), "risk_level": _risk(float(p))}
        for p in probs
    ]
    return {
        "count": len(preds),
        "flagged": sum(x["is_fraud"] for x in preds),
        "threshold": thr,
        "missing_features_filled": filled,
        "predictions": preds,
    }
