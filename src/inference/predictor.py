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
            "Please train candidate models in 7. Model Laboratory first."
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
    dataset_medians: Optional[Dict[str, float]] = None,
    model: Optional[Any] = None,
    metadata: Optional[Dict[str, Any]] = None,
    expected_features: Optional[List[str]] = None,
    currency_symbol: str = CURRENCY_SYMBOL
) -> Dict[str, Any]:
    """
    Generate an estimated market valuation, uncertainty interval, SHAP drivers,
    and profile comparison for a single residential property.
    Supports active session models and arbitrary feature sets.
    """
    # 1. Resolve model and metadata
    if model is None:
        model, default_meta = load_production_model()
        meta = metadata or default_meta
    else:
        meta = metadata or {}

    # 2. Validate payload
    is_valid, validation_errors = validate_input_features(input_data, expected_features=expected_features)
    if not is_valid:
        raise ValueError(f"Input validation failed: {'; '.join(validation_errors)}")

    # 3. Create single-row DataFrame
    single_df = pd.DataFrame([input_data])
    
    # 4. Resolve expected feature schema from model or metadata
    model_features = None
    if hasattr(model, "feature_names_in_"):
        model_features = list(model.feature_names_in_)
    elif hasattr(model, "named_steps") and hasattr(model.named_steps.get("preprocessor"), "feature_names_in_"):
        model_features = list(model.named_steps["preprocessor"].feature_names_in_)
    elif hasattr(model, "regressor_") and hasattr(model.regressor_, "named_steps") and hasattr(model.regressor_.named_steps.get("preprocessor"), "feature_names_in_"):
        model_features = list(model.regressor_.named_steps["preprocessor"].feature_names_in_)
    elif expected_features:
        model_features = expected_features
    else:
        model_features = meta.get("numerical_features", []) + meta.get("categorical_features", [])

    # If the model was trained on Ames features, apply engineer_housing_features
    is_ames = any(f in ["GrLivArea", "OverallQual", "TotalBsmtSF"] for f in (model_features or []))
    if is_ames and not any(f in single_df.columns for f in ["TotalSF", "HouseAgeAtSale"]):
        single_df = engineer_housing_features(single_df, is_training=False)

    # Align columns: fill any missing columns with NaN so preprocessor imputers handle them
    if model_features:
        for col in model_features:
            if col not in single_df.columns:
                single_df[col] = np.nan
        # Ensure column ordering matches model expectations
        eval_df = single_df[[c for c in model_features if c in single_df.columns]].copy()
    else:
        eval_df = single_df.copy()

    # 5. Predict
    raw_pred = model.predict(eval_df)
    predicted_price = float(np.maximum(100.0, raw_pred[0]))

    # 6. Prediction interval calculation
    # Check if calibrated intervals are in metadata
    intervals = meta.get("prediction_intervals", {})
    lower_offset = intervals.get("lower_residual_offset")
    upper_offset = intervals.get("upper_residual_offset")

    if lower_offset is None or upper_offset is None:
        # Dynamic empirical standard error estimation: ~10% or from holdout test RMSE
        test_rmse = meta.get("metrics", {}).get("holdout_test", {}).get("RMSE", predicted_price * 0.10)
        lower_bound = max(100.0, predicted_price - (1.645 * test_rmse))
        upper_bound = predicted_price + (1.645 * test_rmse)
    else:
        lower_bound = max(100.0, predicted_price + lower_offset)
        upper_bound = predicted_price + upper_offset

    # 7. Explainability
    try:
        explanation = explain_single_prediction(model, eval_df, top_k=4)
    except Exception as e:
        logger.warning(f"Single prediction explanation failed: {e}")
        explanation = {"positive_drivers": [], "negative_drivers": []}

    # 8. Comparison against dataset medians
    comparisons = []
    if dataset_medians:
        for feat_name, med_val in dataset_medians.items():
            if feat_name in eval_df.columns and pd.api.types.is_numeric_dtype(eval_df[feat_name]):
                val = float(eval_df[feat_name].iloc[0])
                if med_val > 0:
                    pct = ((val - med_val) / med_val) * 100.0
                    sign = "above" if pct >= 0 else "below"
                    comparisons.append(f"**{feat_name}** ({val:,.1f}) is {abs(pct):.0f}% {sign} the dataset median ({med_val:,.1f}).")

    return {
        "predicted_price": round(predicted_price, 2),
        "formatted_price": f"{currency_symbol}{predicted_price:,.0f}",
        "lower_bound": round(lower_bound, 2),
        "upper_bound": round(upper_bound, 2),
        "formatted_interval": f"{currency_symbol}{lower_bound:,.0f} – {currency_symbol}{upper_bound:,.0f}",
        "model_name": meta.get("model_name", "Champion Model"),
        "model_version": meta.get("model_version", "1.0.0"),
        "test_mae": meta.get("metrics", {}).get("holdout_test", {}).get("MAE", 0.0),
        "positive_drivers": explanation.get("positive_drivers", []),
        "negative_drivers": explanation.get("negative_drivers", []),
        "comparisons": comparisons[:4]
    }

def predict_batch_dataframe(
    df_input: pd.DataFrame,
    model: Optional[Any] = None,
    metadata: Optional[Dict[str, Any]] = None,
    currency_symbol: str = CURRENCY_SYMBOL
) -> pd.DataFrame:
    """
    Score a batch DataFrame, appending predicted_price, lower_bound, upper_bound.
    """
    if model is None:
        model, default_meta = load_production_model()
        meta = metadata or default_meta
    else:
        meta = metadata or {}

    df_scored = df_input.copy()

    # Determine expected model features
    model_features = None
    if hasattr(model, "feature_names_in_"):
        model_features = list(model.feature_names_in_)
    elif hasattr(model, "named_steps") and hasattr(model.named_steps.get("preprocessor"), "feature_names_in_"):
        model_features = list(model.named_steps["preprocessor"].feature_names_in_)
    elif hasattr(model, "regressor_") and hasattr(model.regressor_, "named_steps") and hasattr(model.regressor_.named_steps.get("preprocessor"), "feature_names_in_"):
        model_features = list(model.regressor_.named_steps["preprocessor"].feature_names_in_)

    is_ames = any(f in ["GrLivArea", "OverallQual", "TotalBsmtSF"] for f in (model_features or []))
    if is_ames and not any(f in df_scored.columns for f in ["TotalSF", "HouseAgeAtSale"]):
        eval_df = engineer_housing_features(df_scored, is_training=False)
    else:
        eval_df = df_scored.copy()

    if model_features:
        for col in model_features:
            if col not in eval_df.columns:
                eval_df[col] = np.nan
        eval_df = eval_df[[c for c in model_features if c in eval_df.columns]]

    predictions = model.predict(eval_df)
    predictions = np.maximum(100.0, predictions)

    intervals = meta.get("prediction_intervals", {})
    lower_offset = intervals.get("lower_residual_offset")
    upper_offset = intervals.get("upper_residual_offset")

    if lower_offset is None or upper_offset is None:
        test_rmse = meta.get("metrics", {}).get("holdout_test", {}).get("RMSE", 20000.0)
        lower_bounds = np.maximum(100.0, predictions - (1.645 * test_rmse))
        upper_bounds = predictions + (1.645 * test_rmse)
    else:
        lower_bounds = np.maximum(100.0, predictions + lower_offset)
        upper_bounds = predictions + upper_offset

    df_scored["predicted_price"] = np.round(predictions, 2)
    df_scored["lower_bound"] = np.round(lower_bounds, 2)
    df_scored["upper_bound"] = np.round(upper_bounds, 2)
    df_scored["model_version"] = meta.get("model_version", "1.0.0")

    return df_scored
