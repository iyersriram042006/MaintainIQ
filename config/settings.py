"""Project settings and configuration for MaintainIQ.

Predictive Maintenance System based on the UCI AI4I 2020 dataset.
"""

from pathlib import Path

# Base project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RAW_DATA_FILE = RAW_DATA_DIR / "ai4i2020.csv"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "cleaned_data.csv"

# Models and reports directories
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"

# Default Model artifact path
MODEL_PATH = MODELS_DIR / "model.joblib"

# Dataset feature definitions (UCI AI4I 2020 Predictive Maintenance)
ID_COLUMNS = ["UDI", "Product ID"]
CATEGORICAL_FEATURES = ["Type"]
NUMERICAL_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# Primary target for binary classification (0: No failure, 1: Failure)
TARGET_COLUMN = "Machine failure"

# Specific failure type flags in the dataset
FAILURE_TYPE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]

# The 6 intended input features for modeling
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

# All 14 expected columns in the raw UCI dataset
ALL_EXPECTED_COLUMNS = (
    ID_COLUMNS
    + CATEGORICAL_FEATURES
    + NUMERICAL_FEATURES
    + [TARGET_COLUMN]
    + FAILURE_TYPE_COLUMNS
)

# Non-feature columns (identifiers, target, and failure-modes that cause target leakage)
EXCLUDED_FEATURES = ID_COLUMNS + FAILURE_TYPE_COLUMNS + [TARGET_COLUMN]

# Official UCI download URL
UCI_AI4I_URL = (
    "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"
)

# Modeling hyperparameters & reproducibility
RANDOM_STATE = 42
TEST_SIZE = 0.2

