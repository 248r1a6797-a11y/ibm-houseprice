"""SHAP explainability engine for HouseWise AI."""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
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

    preprocessor = None
    regressor = None

    if hasattr(pipe, "named_steps"):
        preprocessor = pipe.named_steps.get("preprocessor")
        regressor = pipe.named_steps.get("regressor")
    else:
        regressor = pipe

    return regressor, preprocessor

def compute_global_shap(
    model_pipeline: Any,
    X_background: pd.DataFrame,
    sample_size: int = 150
) -> Dict[str, Any]:
    """
    Compute global SHAP values using TreeExplainer or Permutation Explainer on a sample of background data.
    Returns global feature importance rankings and feature values for plotting.
    """
    regressor, preprocessor = get_underlying_estimator_and_preprocessor(model_pipeline)
    if preprocessor is None or regressor is None:
        # Fallback if unpipelined
        if hasattr(regressor, "feature_importances_") and hasattr(X_background, "columns"):
            imp = regressor.feature_importances_
            df_imp = pd.DataFrame({"clean_feature": X_background.columns, "mean_abs_shap": imp}).sort_values("mean_abs_shap", ascending=False)
            return {"importance_df": df_imp, "feature_names": list(X_background.columns)}
        return {}

    # Sample background to maintain fast runtime
    sample_df = X_background.sample(min(len(X_background), sample_size), random_state=42)
    try:
        X_trans = preprocessor.transform(sample_df)
        if hasattr(X_trans, "toarray"):
            X_trans = X_trans.toarray()
        feature_names = preprocessor.get_feature_names_out()
    except Exception as e:
        logger.warning(f"Preprocessor transform failed in SHAP: {e}")
        return {}

    shap_values = None
    expected_val = 0.0

    try:
        explainer = shap.TreeExplainer(regressor)
        shap_values = explainer.shap_values(X_trans)
        expected_val = float(explainer.expected_value) if not isinstance(explainer.expected_value, (list, np.ndarray)) else float(explainer.expected_value[0])
    except Exception as e:
        logger.info(f"TreeExplainer not applicable ({e}); falling back to sampling Explainer.")
        try:
            background_sample = X_trans[:min(30, len(X_trans))]
            explainer = shap.Explainer(regressor.predict, background_sample)
            res = explainer(X_trans[:min(50, len(X_trans))])
            shap_values = res.values
            expected_val = float(np.mean(regressor.predict(background_sample)))
        except Exception as e2:
            logger.warning(f"Sampling explainer failed ({e2}); calculating feature importances or coefficients.")
            if hasattr(regressor, "feature_importances_"):
                mean_abs_shap = np.abs(regressor.feature_importances_)
            elif hasattr(regressor, "coef_"):
                mean_abs_shap = np.abs(regressor.coef_)
            else:
                mean_abs_shap = np.ones(len(feature_names)) / len(feature_names)
            shap_values = None

    if shap_values is not None:
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap
    }).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)

    # Clean display feature names (e.g. 'num__GrLivArea' -> 'GrLivArea')
    clean_names = []
    for f in importance_df["feature"]:
        clean = str(f).replace("num__", "").replace("cat__", "")
        clean_names.append(clean)
    importance_df["clean_feature"] = clean_names

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

    try:
        X_trans = preprocessor.transform(single_record_df)
        if hasattr(X_trans, "toarray"):
            X_trans = X_trans.toarray()
        feature_names = preprocessor.get_feature_names_out()
    except Exception as e:
        logger.warning(f"Single property transform failed: {e}")
        return {}

    try:
        explainer = shap.TreeExplainer(regressor)
        raw_shap = explainer.shap_values(X_trans)
        shap_vals = raw_shap[0] if len(raw_shap.shape) > 1 else raw_shap
        base_val = float(explainer.expected_value) if not isinstance(explainer.expected_value, (list, np.ndarray)) else float(explainer.expected_value[0])
    except Exception:
        # Fallback to feature importance approximation
        if hasattr(regressor, "coef_"):
            shap_vals = regressor.coef_ * X_trans[0]
            base_val = float(getattr(regressor, "intercept_", 0.0))
        elif hasattr(regressor, "feature_importances_"):
            shap_vals = regressor.feature_importances_ * X_trans[0]
            base_val = float(regressor.predict(X_trans)[0])
        else:
            shap_vals = np.zeros(X_trans.shape[1])
            base_val = float(regressor.predict(X_trans)[0])

    # Build attribution list
    attributions = []
    for fname, sval, fval in zip(feature_names, shap_vals, X_trans[0]):
        clean_name = fname.replace("num__", "").replace("cat__", "")
        # Only include active or non-zero contributions
        if abs(sval) > 1e-5:
            attributions.append({
                "feature": clean_name,
                "clean_feature": clean_name,
                "contribution": float(sval),
                "dollar_contribution": float(sval),
                "feature_value": float(fval)
            })

    # Sort positive and negative drivers
    pos_drivers = sorted([a for a in attributions if a["contribution"] > 0], key=lambda x: x["contribution"], reverse=True)[:top_k]
    neg_drivers = sorted([a for a in attributions if a["contribution"] < 0], key=lambda x: x["contribution"])[:top_k]

    return {
        "base_value_log": base_val,
        "base_value_dollars": float(np.expm1(base_val)) if base_val < 30 else float(base_val),
        "positive_drivers": pos_drivers,
        "negative_drivers": neg_drivers,
        "all_attributions": sorted(attributions, key=lambda x: abs(x["contribution"]), reverse=True)[:15]
    }
