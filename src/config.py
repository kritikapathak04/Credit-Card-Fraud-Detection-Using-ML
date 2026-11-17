"""Central configuration and paths for the Credit Card Fraud Detection system."""
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

DATA_FILE = DATA_DIR / "creditcard.csv"
SYNTH_DATA = DATA_DIR / "sample_creditcard.csv"
ARTIFACT_PATH = MODELS_DIR / "best_model.joblib"

# Feature definitions
TARGET = "Class"
V_COLS = [f"V{i}" for i in range(1, 29)]
RAW_FEATURES = ["Time"] + V_COLS + ["Amount"]
ENGINEERED_FEATURES = ["Hour", "Log_Amount"]
ALL_FEATURES = RAW_FEATURES + ENGINEERED_FEATURES

# Defaults & Hyperparameters
RANDOM_STATE = 42
DEFAULT_FRAUD_RATIO = 0.00172  # 0.172% matching Kaggle dataset
HIGH_RISK_THRESHOLD = 0.70
TEST_SIZE = 0.20
VAL_SIZE = 0.16  # 64% train / 16% val / 20% test
