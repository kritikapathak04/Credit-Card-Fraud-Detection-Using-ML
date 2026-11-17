"""Data cleaning, feature engineering, and robust scaling pipeline."""
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler

from src.config import ALL_FEATURES, RAW_FEATURES, TARGET, V_COLS

SCALE_COLS = ["Time", "Amount", "Hour", "Log_Amount"]


def clean_data(df: pd.DataFrame, drop_duplicates: bool = True) -> pd.DataFrame:
    """Clean the raw transactions dataframe: drop duplicates and handle missing values."""
    df_clean = df.copy()
    if drop_duplicates:
        initial_count = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        dropped = initial_count - len(df_clean)
        if dropped > 0:
            print(f"Removed {dropped:,} duplicate transactions.")
    df_clean = df_clean.dropna()
    return df_clean


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add Hour of day and Log_Amount engineered features."""
    df_out = df.copy()
    # Hour of day (0-23)
    df_out["Hour"] = (np.floor(df_out["Time"] / 3600) % 24).astype(float)
    # Log_Amount: handles heavy skew in transaction amounts
    df_out["Log_Amount"] = np.log1p(df_out["Amount"].clip(lower=0.0)).astype(float)
    return df_out


def get_feature_matrix_and_target(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare feature matrix X and target y."""
    df_feat = engineer_features(df)
    feature_cols = [c for c in ALL_FEATURES if c in df_feat.columns]
    X = df_feat[feature_cols]
    y = df_feat[TARGET] if TARGET in df_feat.columns else None
    return X, y


def build_preprocessor(scale_cols: List[str] = SCALE_COLS) -> ColumnTransformer:
    """Build a ColumnTransformer that applies RobustScaler to non-PCA features

    and passes through PCA features (V1..V28) which are already scaled.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("scaler", RobustScaler(), scale_cols),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    return preprocessor
