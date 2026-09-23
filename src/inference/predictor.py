"""Production inference engine for single-property and batch scoring."""
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import joblib
import numpy as np
import pandas as pd

from ..config import PRODUCTION_MODEL_PATH, MODEL_METADATA_PATH, CURRENCY_SYMBOL
from ..data.validator import validate_input_features
from ..features.engineering import engineer_housing_features
from ..explainability.shap_analysis import explain_single_prediction

logger = logging.getLogger(__name__)

_MODEL_CACHE = None
_METADATA_CACHE = None

def load_production_model() -> Tuple[Any, Dict[str, Any]]:
    """Load and cache the trained production model and associated metadata."""
    global _MODEL_CACHE, _METADATA_CACHE
    if _MODEL_CACHE is not None and _METADATA_CACHE is not None:
        return _MODEL_CACHE, _METADATA_CACHE

    if not PRODUCTION_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Production model not found at {PRODUCTION_MODEL_PATH}. "
            "Please execute 'python scripts/train.py' first."
        )

    logger.info(f"Loading production model from: {PRODUCTION_MODEL_PATH}")
    model = joblib.load(PRODUCTION_MODEL_PATH)
    
    metadata = {}
    if MODEL_METADATA_PATH.exists():
        with open(MODEL_METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    _MODEL_CACHE = model
    _METADATA_CACHE = metadata
    return model, metadata

def predict_single_property(
    input_data: Dict[str, Any],
    dataset_medians: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Generate an estimated market valuation, uncertainty interval, SHAP drivers,
    and profile comparison for a single residential property.
    """
    # 1. Validate payload
    is_valid, validation_errors = validate_input_features(input_data)
    if not is_valid:
        raise ValueError(f"Input validation failed: {'; '.join(validation_errors)}")

    # 2. Load model
    model, metadata = load_production_model()

    # 3. Create single-row DataFrame and align with model feature schema
    single_df = pd.DataFrame([input_data])
    
    # Populate any missing feature columns with NaN so preprocessor imputers handle them
    expected_cols = metadata.get("numerical_features", []) + metadata.get("categorical_features", [])
    for col in expected_cols:
        if col not in single_df.columns:
            single_df[col] = np.nan

    # 4. Feature engineering
    engineered_df = engineer_housing_features(single_df, is_training=False)

    # 5. Predict
    raw_pred = model.predict(engineered_df)
    predicted_price = float(np.maximum(1000.0, raw_pred[0]))

    # 6. Prediction interval calculation
    intervals = metadata.get("prediction_intervals", {})
    lower_offset = intervals.get("lower_residual_offset", -25000.0)
    upper_offset = intervals.get("upper_residual_offset", 25000.0)

    # Predicted price with calibrated bounds: Actual = Pred + Residual
    lower_bound = max(10000.0, predicted_price + lower_offset)
    upper_bound = predicted_price + upper_offset

    # 7. Explainability
    explanation = explain_single_prediction(model, engineered_df, top_k=4)

    # 8. Comparison against dataset medians
    comparisons = []
    if dataset_medians:
        liv_area = float(engineered_df["GrLivArea"].iloc[0]) if "GrLivArea" in engineered_df.columns else 0.0
        med_liv = dataset_medians.get("median_living_area", 1442.0)
        if liv_area > med_liv:
            diff_pct = ((liv_area - med_liv) / med_liv) * 100
            comparisons.append(f"Living Area ({liv_area:,.0f} sq ft) is {diff_pct:.0f}% above the Ames market median ({med_liv:,.0f} sq ft).")
        else:
            diff_pct = ((med_liv - liv_area) / med_liv) * 100
            comparisons.append(f"Living Area ({liv_area:,.0f} sq ft) is {diff_pct:.0f}% below the Ames market median ({med_liv:,.0f} sq ft).")

        qual = int(engineered_df["OverallQual"].iloc[0]) if "OverallQual" in engineered_df.columns else 6
        comparisons.append(f"Construction Quality rating is {qual}/10 (Market median: 6/10).")

    return {
        "predicted_price": round(predicted_price, 2),
        "formatted_price": f"{CURRENCY_SYMBOL}{predicted_price:,.0f}",
        "lower_bound": round(lower_bound, 2),
        "upper_bound": round(upper_bound, 2),
        "formatted_interval": f"{CURRENCY_SYMBOL}{lower_bound:,.0f} – {CURRENCY_SYMBOL}{upper_bound:,.0f}",
        "model_name": metadata.get("model_name", "Production Regressor"),
        "model_version": metadata.get("model_version", "1.0.0"),
        "test_mae": metadata.get("metrics", {}).get("holdout_test", {}).get("MAE", 15000.0),
        "positive_drivers": explanation.get("positive_drivers", []),
        "negative_drivers": explanation.get("negative_drivers", []),
        "comparisons": comparisons
    }

def predict_batch_dataframe(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Score a batch DataFrame, appending predicted_price, lower_bound, upper_bound.
    """
    model, metadata = load_production_model()

    df_scored = df_input.copy()
    expected_cols = metadata.get("numerical_features", []) + metadata.get("categorical_features", [])
    for col in expected_cols:
        if col not in df_scored.columns:
            df_scored[col] = np.nan

    engineered_df = engineer_housing_features(df_scored, is_training=False)

    predictions = model.predict(engineered_df)
    predictions = np.maximum(1000.0, predictions)

    intervals = metadata.get("prediction_intervals", {})
    lower_offset = intervals.get("lower_residual_offset", -25000.0)
    upper_offset = intervals.get("upper_residual_offset", 25000.0)

    df_scored["predicted_price"] = np.round(predictions, 2)
    df_scored["lower_bound"] = np.round(np.maximum(10000.0, predictions + lower_offset), 2)
    df_scored["upper_bound"] = np.round(predictions + upper_offset, 2)
    df_scored["model_version"] = metadata.get("model_version", "1.0.0")

    return df_scored
