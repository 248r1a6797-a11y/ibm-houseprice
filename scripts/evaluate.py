"""Comprehensive model evaluation script for HouseWise AI."""
import json
import logging
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    PRODUCTION_MODEL_PATH, MODEL_METADATA_PATH, PROCESSED_TEST_PATH, TARGET_COLUMN
)
from src.evaluation.metrics import compute_regression_metrics
from src.evaluation.error_analysis import (
    compute_residuals,
    analyze_worst_predictions,
    analyze_errors_by_segment
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def evaluate():
    logger.info("Executing production model holdout evaluation...")
    if not PRODUCTION_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model artifact not found at {PRODUCTION_MODEL_PATH}. Run scripts/train.py first.")

    model = joblib.load(PRODUCTION_MODEL_PATH)
    test_df = pd.read_parquet(PROCESSED_TEST_PATH)

    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN].values

    # Score
    y_pred = model.predict(X_test)
    y_pred = np.maximum(1000.0, y_pred)

    metrics = compute_regression_metrics(y_test, y_pred, prefix="Holdout_Test")
    logger.info("Holdout Test Performance Metrics:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v}")

    # Residuals & Segments
    df_res = compute_residuals(y_test, y_pred)
    
    # Neighborhood segment performance
    neigh_errors = analyze_errors_by_segment(df_res, X_test, "Neighborhood")
    logger.info("\nTop 5 Neighborhoods by Sample Count Error Performance:")
    print(neigh_errors.head(5).to_string())

    # Save summary
    eval_summary = {
        "metrics": metrics,
        "test_sample_size": len(test_df),
        "worst_mae_neighborhood": neigh_errors.iloc[0]["Neighborhood"] if not neigh_errors.empty else "N/A"
    }
    eval_path = Path("reports/tables/evaluation_summary.json")
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2)

    logger.info(f"\nEvaluation summary saved to: {eval_path}")

if __name__ == "__main__":
    evaluate()
