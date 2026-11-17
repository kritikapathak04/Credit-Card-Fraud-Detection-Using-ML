"""Data loading and synthetic dataset generation for credit card fraud detection."""
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src.config import DATA_FILE, RAW_FEATURES, SYNTH_DATA, TARGET, V_COLS


def generate_synthetic(
    n_samples: int = 25000,
    fraud_ratio: float = 0.002,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate a high-fidelity synthetic demo dataset matching the Kaggle schema."""
    rng = np.random.default_rng(random_state)
    n_fraud = max(int(n_samples * fraud_ratio), 50)
    n_legit = n_samples - n_fraud

    # Time in seconds over 48 hours (0 to 172,800)
    time_legit = rng.uniform(0, 172800, size=n_legit)
    # Fraud transactions often cluster during night / off-peak hours
    time_fraud = rng.choice(
        np.concatenate([
            rng.uniform(3600 * 1, 3600 * 5, size=int(n_fraud * 0.6)),
            rng.uniform(0, 172800, size=n_fraud - int(n_fraud * 0.6)),
        ]),
        size=n_fraud,
        replace=False,
    )

    # V1-V28 PCA features (legitimate transactions centered around 0 with unit variance)
    v_legit = rng.normal(0.0, 1.0, size=(n_legit, 28))
    # Fraud transactions show specific shifts on key correlated features (V14, V4, V3, V11, V20)
    v_fraud = rng.normal(0.0, 1.2, size=(n_fraud, 28))
    # In the real dataset: V14 is strongly negative, V4 positive, V3 negative, V11 positive, V20 positive
    v_indices = {f"V{i}": i - 1 for i in range(1, 29)}
    v_fraud[:, v_indices["V14"]] -= 4.5
    v_fraud[:, v_indices["V4"]] += 3.8
    v_fraud[:, v_indices["V3"]] -= 3.5
    v_fraud[:, v_indices["V11"]] += 3.2
    v_fraud[:, v_indices["V20"]] += 1.5

    # Amounts: log-normal distribution
    amount_legit = np.exp(rng.normal(3.2, 1.2, size=n_legit))
    amount_fraud = np.exp(rng.normal(4.5, 1.6, size=n_fraud))

    # Combine
    df_legit = pd.DataFrame(v_legit, columns=V_COLS)
    df_legit["Time"] = time_legit
    df_legit["Amount"] = np.round(amount_legit, 2)
    df_legit[TARGET] = 0

    df_fraud = pd.DataFrame(v_fraud, columns=V_COLS)
    df_fraud["Time"] = time_fraud
    df_fraud["Amount"] = np.round(amount_fraud, 2)
    df_fraud[TARGET] = 1

    df = pd.concat([df_legit, df_fraud], ignore_index=True)
    # Shuffle
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    # Order columns as in Kaggle dataset: Time, V1..V28, Amount, Class
    ordered_cols = ["Time"] + V_COLS + ["Amount", TARGET]
    return df[ordered_cols]


def load_data(
    path: Optional[str] = None, force_synthetic: bool = False
) -> Tuple[pd.DataFrame, str]:
    """Load credit card dataset from CSV path, default Kaggle file, or synthetic demo."""
    if force_synthetic:
        if SYNTH_DATA.exists():
            return pd.read_csv(SYNTH_DATA), f"Synthetic demo dataset ({SYNTH_DATA})"
        SYNTH_DATA.parent.mkdir(parents=True, exist_ok=True)
        df = generate_synthetic()
        df.to_csv(SYNTH_DATA, index=False)
        return df, "Generated synthetic demo dataset"

    if path:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Requested dataset path does not exist: {p}")
        return pd.read_csv(p), f"Custom path ({p.resolve()})"

    if DATA_FILE.exists():
        return pd.read_csv(DATA_FILE), f"Official Kaggle dataset ({DATA_FILE})"

    if SYNTH_DATA.exists():
        return pd.read_csv(SYNTH_DATA), f"Synthetic demo dataset ({SYNTH_DATA})"

    # Generate synthetic fallback
    SYNTH_DATA.parent.mkdir(parents=True, exist_ok=True)
    df = generate_synthetic()
    df.to_csv(SYNTH_DATA, index=False)
    return df, "Synthetic demo dataset (auto-generated fallback)"
