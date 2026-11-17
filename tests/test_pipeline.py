"""Automated test suite for data loading, preprocessing, model inference, and API endpoints."""
import numpy as np
import pandas as pd
import pytest

from src.config import RAW_FEATURES, TARGET, V_COLS
from src.data_loader import generate_synthetic
from src.predict import risk_level
from src.preprocessing import clean_data, engineer_features


def test_synthetic_data_generator():
    """Verify synthetic dataset matches the expected schema and fraud distribution."""
    df = generate_synthetic(n_samples=1000, fraud_ratio=0.01, random_state=42)
    assert len(df) == 1000
    assert TARGET in df.columns
    assert "Time" in df.columns
    assert "Amount" in df.columns
    for c in V_COLS:
        assert c in df.columns
    assert df[TARGET].sum() > 0
    assert set(df[TARGET].unique()).issubset({0, 1})


def test_clean_data_duplicates():
    """Verify duplicate transaction removal."""
    df = pd.DataFrame({
        "Time": [100, 100, 200],
        "Amount": [50.0, 50.0, 75.0],
        "Class": [0, 0, 1],
    })
    for c in V_COLS:
        df[c] = 0.0

    df_cleaned = clean_data(df, drop_duplicates=True)
    assert len(df_cleaned) == 2


def test_feature_engineering():
    """Verify Hour and Log_Amount derivations."""
    df = pd.DataFrame({
        "Time": [3600 * 2, 3600 * 14.5],  # 2:00 AM, 2:30 PM
        "Amount": [0.0, 99.0],
    })
    for c in V_COLS:
        df[c] = 0.0

    df_feat = engineer_features(df)
    assert "Hour" in df_feat.columns
    assert "Log_Amount" in df_feat.columns
    assert df_feat.loc[0, "Hour"] == 2.0
    assert df_feat.loc[1, "Hour"] == 14.0
    assert df_feat.loc[0, "Log_Amount"] == 0.0
    assert np.isclose(df_feat.loc[1, "Log_Amount"], np.log1p(99.0))


def test_risk_level_tiering():
    """Verify risk classification rules."""
    assert risk_level(0.85, threshold=0.4) == "High"
    assert risk_level(0.55, threshold=0.4) == "Medium"
    assert risk_level(0.20, threshold=0.4) == "Low"


def test_model_artifact_and_predict():
    """Verify loading trained artifact and scoring transactions."""
    from src.predict import load_artifact, predict_df
    artifact = load_artifact()
    assert "pipeline" in artifact
    assert "threshold" in artifact

    # Test single transaction scoring
    sample_row = pd.DataFrame([{c: 0.0 for c in RAW_FEATURES}])
    sample_row["Amount"] = 120.0
    sample_row["Time"] = 45000.0
    scored = predict_df(artifact, sample_row)

    assert "fraud_probability" in scored.columns
    assert "prediction" in scored.columns
    assert "risk_level" in scored.columns
    prob = scored.loc[0, "fraud_probability"]
    assert 0.0 <= prob <= 1.0

