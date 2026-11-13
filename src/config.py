"""
Central configuration for the E-Commerce Product Recommendation Engine.
All paths, random seeds, and hyperparameters are defined here for 100% reproducibility.
"""
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
REPORTS_DIR = BASE_DIR / "reports"

# Ensure directories exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, ARTIFACTS_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# File Paths
PRODUCTS_FILE = RAW_DATA_DIR / "products.csv"
USERS_FILE = RAW_DATA_DIR / "users.csv"
INTERACTIONS_FILE = RAW_DATA_DIR / "interactions.csv"

TRAIN_INTERACTIONS_FILE = PROCESSED_DATA_DIR / "train_interactions.csv"
VAL_INTERACTIONS_FILE = PROCESSED_DATA_DIR / "val_interactions.csv"
TEST_INTERACTIONS_FILE = PROCESSED_DATA_DIR / "test_interactions.csv"
COLD_START_TEST_FILE = PROCESSED_DATA_DIR / "cold_start_test.csv"

# Artifact file paths
POPULARITY_MODEL_FILE = ARTIFACTS_DIR / "popularity_model.joblib"
COLLABORATIVE_MODEL_FILE = ARTIFACTS_DIR / "collaborative_model.joblib"
CONTENT_MODEL_FILE = ARTIFACTS_DIR / "content_model.joblib"
HYBRID_MODEL_FILE = ARTIFACTS_DIR / "hybrid_model.joblib"
METRICS_EVAL_FILE = ARTIFACTS_DIR / "evaluation_metrics.json"
PREPROCESSOR_FILE = ARTIFACTS_DIR / "preprocessor.joblib"

# Reproducibility
RANDOM_SEED = 42

# Data Generation Parameters
NUM_USERS = 1200
NUM_PRODUCTS = 250
NUM_INTERACTIONS = 18000
DAYS_SPAN = 90

# Interaction Event Weights (Implicit Feedback Scoring)
EVENT_WEIGHTS = {
    "view": 1.0,
    "cart": 2.5,
    "purchase": 5.0,
    "rating": 3.0  # fallback if raw rating is absent
}

# Train/Val/Test Chronological Split Ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Ranking Metrics Configuration
DEFAULT_TOP_K = 10
EVAL_K_LIST = [5, 10, 20]

# Model Hyperparameters
CF_LATENT_FACTORS = 30
CF_RANDOM_STATE = RANDOM_SEED
TFIDF_MAX_FEATURES = 600
HYBRID_DEFAULT_WEIGHTS = {
    "collaborative": 0.50,
    "content": 0.35,
    "popularity": 0.15
}
