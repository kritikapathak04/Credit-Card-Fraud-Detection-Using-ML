"""Inference and prediction utilities for single transactions and batches."""
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import joblib
import numpy as np
import pandas as pd

from src.config import ARTIFACT_PATH, HIGH_RISK_THRESHOLD, RAW_FEATURES
from src.preprocessing import engineer_features


def load_artifact(path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load the trained model artifact bundle from disk."""
    p = Path(path) if path else ARTIFACT_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"Trained model artifact not found at {p.resolve()}. Please run train.py first."
        )
    artifact = joblib.load(p)
    if not isinstance(artifact, dict):
        # Wrap bare pipeline if loaded
        artifact = {
            "pipeline": artifact,
            "model": artifact,
            "threshold": getattr(artifact, "threshold", 0.5),
            "model_name": type(artifact).__name__,
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "data_source": "Default",
        }
    return artifact


def risk_level(prob: float, threshold: float = 0.5) -> str:
    """Categorize prediction risk into High, Medium, or Low tiers."""
    if prob >= max(HIGH_RISK_THRESHOLD, threshold):
        return "High"
    if prob >= threshold:
        return "Medium"
    return "Low"


def predict_df(artifact: Dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    """Score a DataFrame of transactions using the saved model pipeline and tuned threshold."""
    pipeline = artifact.get("pipeline") or artifact.get("model")
    threshold = float(artifact.get("threshold", 0.5))

    df_scored = df.copy()

    # Ensure all expected raw features exist
    for col in RAW_FEATURES:
        if col not in df_scored.columns:
            df_scored[col] = 0.0

    # Apply feature engineering
    df_feat = engineer_features(df_scored)

    # Scikit-learn Pipeline expects the exact feature names it was trained on
    expected_cols = getattr(pipeline, "feature_names_in_", None)
    if expected_cols is not None:
        for c in expected_cols:
            if c not in df_feat.columns:
                df_feat[c] = 0.0
        X = df_feat[list(expected_cols)]
    else:
        X = df_feat

    # Get fraud probability (class 1)
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(X)[:, 1]
    elif hasattr(pipeline, "decision_function"):
        scores = pipeline.decision_function(X)
        probs = 1.0 / (1.0 + np.exp(-scores))
    else:
        probs = pipeline.predict(X).astype(float)

    probs = np.clip(probs, 0.0, 1.0)
    preds = (probs >= threshold).astype(int)

    df_scored["fraud_probability"] = np.round(probs, 4)
    df_scored["prediction"] = preds
    df_scored["risk_level"] = [risk_level(p, threshold) for p in probs]
    return df_scored
