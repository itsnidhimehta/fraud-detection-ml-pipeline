"""All settings in one place, so nothing is hard-coded inside the other files."""

from pathlib import Path

# Folders
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

DATA_PATH = DATA_DIR / "transactions.csv"
MODEL_PATH = MODELS_DIR / "fraud_model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"

# Reproducibility: same random numbers every run
RANDOM_STATE = 42

# Dataset size and train/test split
N_ROWS = 20_000
TEST_SIZE = 0.2

# Column names
TARGET = "is_fraud"

NUMERIC_FEATURES = [
    "amount",
    "hour",
    "distance_from_home_km",
    "txn_count_last_24h",
    "account_age_days",
    "avg_amount_30d",
    "is_international",
    "card_present",
    # engineered features (created in features.py)
    "amount_to_avg_ratio",
    "is_night",
    "log_amount",
]

CATEGORICAL_FEATURES = ["merchant_category"]

MERCHANT_CATEGORIES = ["grocery", "restaurant", "fuel", "online", "electronics", "travel"]

# Probability above which a transaction is flagged as fraud
DECISION_THRESHOLD = 0.5
