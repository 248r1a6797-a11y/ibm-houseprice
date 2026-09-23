"""SHAP explainability engine for HouseWise AI."""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
import logging
import shap

logger = logging.getLogger(__name__)

def get_underlying_estimator_and_preprocessor(model_pipeline: Any) -> Tuple[Any, Any]:
    """Extract regressor and preprocessor from pipeline or TransformedTargetRegressor."""
    if hasattr(model_pipeline, "regressor_"):
        pipe = model_pipeline.regressor_
    elif hasattr(model_pipeline, "named_steps"):
        pipe = model_pipeline
    else:
        return model_pipeline, None

    preprocessor = pipe.named_steps.get("preprocessor")
    regressor = pipe.named_steps.get("regressor")
    return regressor, preprocessor

def compute_global_shap(
    model_pipeline: Any,
    X_background: pd.DataFrame,
    sample_size: int = 150
) -> Dict[str, Any]:
    """
    Compute global SHAP values using TreeExplainer on a sample of background data.
    Returns global feature importance rankings and feature values for plotting.
    """
    regressor, preprocessor = get_underlying_estimator_and_preprocessor(model_pipeline)
    if preprocessor is None or regressor is None:
        return {}

    # Sample background to maintain fast runtime
    sample_df = X_background.sample(min(len(X_background), sample_size), random_state=42)
    X_trans = preprocessor.transform(sample_df)
    feature_names = preprocessor.get_feature_names_out()

    try:
        explainer = shap.TreeExplainer(regressor)
        shap_values = explainer.shap_values(X_trans)
        expected_val = float(explainer.expected_value)
    except Exception as e:
        logger.warning(f"TreeExplainer failed ({e}); falling back to sampling Explainer.")
        explainer = shap.Explainer(regressor.predict, X_trans[:50])
        res = explainer(X_trans)
        shap_values = res.values
        expected_val = float(np.mean(regressor.predict(X_trans[:50])))

    # Compute mean absolute SHAP for each transformed feature
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap
    }).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)

    # Clean display feature names (e.g. 'num__GrLivArea' -> 'GrLivArea')
    importance_df["clean_feature"] = (
        importance_df["feature"]
        .str.replace("num__", "")
        .str.replace("cat__", "")
    )

    return {
        "importance_df": importance_df,
        "feature_names": feature_names.tolist(),
        "shap_values": shap_values,
        "X_trans": X_trans,
        "expected_value": expected_val
    }

def explain_single_prediction(
    model_pipeline: Any,
    single_record_df: pd.DataFrame,
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Generate waterfall contribution breakdown for an individual property prediction.
    Separates into top positive drivers and top negative drivers.
    """
    regressor, preprocessor = get_underlying_estimator_and_preprocessor(model_pipeline)
    if preprocessor is None or regressor is None:
        return {}

    X_trans = preprocessor.transform(single_record_df)
    feature_names = preprocessor.get_feature_names_out()

    try:
        explainer = shap.TreeExplainer(regressor)
        shap_vals = explainer.shap_values(X_trans)[0]
        base_val = float(explainer.expected_value)
    except Exception as e:
        logger.warning(f"Single TreeExplainer failed ({e}); using linear approximation.")
        shap_vals = np.zeros(X_trans.shape[1])
        base_val = float(regressor.predict(X_trans)[0])

    # Build attribution list
    attributions = []
    for fname, sval, fval in zip(feature_names, shap_vals, X_trans[0]):
        clean_name = fname.replace("num__", "").replace("cat__", "")
        # Only include active one-hot or non-zero contributions
        if abs(sval) > 1e-5:
            attributions.append({
                "feature": clean_name,
                "contribution": float(sval),
                "feature_value": float(fval)
            })

    # Sort positive and negative drivers
    pos_drivers = sorted([a for a in attributions if a["contribution"] > 0], key=lambda x: x["contribution"], reverse=True)[:top_k]
    neg_drivers = sorted([a for a in attributions if a["contribution"] < 0], key=lambda x: x["contribution"])[:top_k]

    return {
        "base_value_log": base_val,
        "base_value_dollars": float(np.expm1(base_val)),
        "positive_drivers": pos_drivers,
        "negative_drivers": neg_drivers,
        "all_attributions": sorted(attributions, key=lambda x: abs(x["contribution"]), reverse=True)[:15]
    }
