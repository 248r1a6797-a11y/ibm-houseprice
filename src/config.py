"""Configuration loader and management for HouseWise AI."""
from pathlib import Path
from typing import Any, Dict
import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent

def load_yaml(file_path: Path) -> Dict[str, Any]:
    """Load a YAML configuration file safely."""
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# Load configurations
CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"
FEATURE_CONFIG_PATH = ROOT_DIR / "config" / "feature_config.yaml"

CONFIG = load_yaml(CONFIG_PATH)
FEATURE_CONFIG = load_yaml(FEATURE_CONFIG_PATH)

# Key configuration constants
PROJECT_NAME = CONFIG.get("project_name", "HouseWise AI")
DATA_CFG = CONFIG.get("data", {})
MODEL_CFG = CONFIG.get("modeling", {})
DASHBOARD_CFG = CONFIG.get("dashboard", {})

RAW_DATA_PATH = ROOT_DIR / DATA_CFG.get("raw_path", "data/raw/AmesHousing.csv")
DOCS_DATA_PATH = ROOT_DIR / DATA_CFG.get("documentation_path", "data/raw/DataDocumentation.txt")
PROCESSED_TRAIN_PATH = ROOT_DIR / DATA_CFG.get("processed_train_path", "data/processed/train.parquet")
PROCESSED_TEST_PATH = ROOT_DIR / DATA_CFG.get("processed_test_path", "data/processed/test.parquet")
DATA_PROFILE_PATH = ROOT_DIR / DATA_CFG.get("data_profile_path", "reports/data_profile.json")

TARGET_COLUMN = DATA_CFG.get("target_column", "SalePrice")
ID_COLUMNS = DATA_CFG.get("id_columns", ["Order", "PID"])
CURRENCY_SYMBOL = DATA_CFG.get("currency_symbol", "$")
CURRENCY = DATA_CFG.get("currency", "USD")

RANDOM_STATE = MODEL_CFG.get("random_state", 42)
TEST_SIZE = MODEL_CFG.get("test_size", 0.20)
VALIDATION_FOLDS = MODEL_CFG.get("validation_folds", 5)
PRIMARY_METRIC = MODEL_CFG.get("primary_metric", "RMSE")
PRICE_TRANSFORMATION = MODEL_CFG.get("price_transformation", "log1p")

PRODUCTION_MODEL_PATH = ROOT_DIR / MODEL_CFG.get("production_model_path", "models/production/production_model.joblib")
MODEL_METADATA_PATH = ROOT_DIR / MODEL_CFG.get("metadata_path", "models/production/model_metadata.json")
