"""Residual diagnostics, worst prediction forensics, and uncertainty estimation."""
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from .metrics import compute_regression_metrics

def compute_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """
    Calculate residuals and absolute/relative errors.
    Residual = Actual - Predicted (Positive = Underprediction, Negative = Overprediction).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = y_true - y_pred
    abs_errors = np.abs(residuals)
    pct_errors = (abs_errors / np.clip(y_true, 1.0, None)) * 100.0

    return pd.DataFrame({
        "actual": y_true,
        "predicted": y_pred,
        "residual": residuals,
        "absolute_error": abs_errors,
        "percentage_error": pct_errors
    })

def analyze_worst_predictions(
    df_eval: pd.DataFrame,
    df_features: pd.DataFrame,
    top_n: int = 10
) -> Dict[str, pd.DataFrame]:
    """
    Identify the top N largest underpredictions and overpredictions with property features.
    """
    merged = pd.concat([df_features.reset_index(drop=True), df_eval.reset_index(drop=True)], axis=1)

    # Underpredictions: Actual >> Predicted (residual > 0)
    worst_under = merged.sort_values(by="residual", ascending=False).head(top_n).copy()

    # Overpredictions: Actual << Predicted (residual < 0)
    worst_over = merged.sort_values(by="residual", ascending=True).head(top_n).copy()

    return {
        "worst_underpredictions": worst_under,
        "worst_overpredictions": worst_over
    }

def analyze_errors_by_segment(
    df_eval: pd.DataFrame,
    df_features: pd.DataFrame,
    segment_col: str
) -> pd.DataFrame:
    """Compute MAE, RMSE, and sample size for specific categorical segments."""
    merged = pd.concat([df_features.reset_index(drop=True), df_eval.reset_index(drop=True)], axis=1)
    if segment_col not in merged.columns:
        return pd.DataFrame()

    results = []
    for segment_val, group in merged.groupby(segment_col):
        if len(group) == 0:
            continue
        metrics = compute_regression_metrics(group["actual"], group["predicted"])
        results.append({
            segment_col: str(segment_val),
            "sample_count": len(group),
            "mae": metrics["MAE"],
            "rmse": metrics["RMSE"],
            "median_ae": metrics["MedianAE"],
            "mape_pct": metrics["MAPE_pct"]
        })

    res_df = pd.DataFrame(results).sort_values(by="sample_count", ascending=False).reset_index(drop=True)
    return res_df

def compute_calibrated_prediction_intervals(
    y_true_val: np.ndarray,
    y_pred_val: np.ndarray,
    coverage: float = 0.80
) -> Dict[str, float]:
    """
    Compute empirical residual quantiles on validation data for calibrated prediction intervals.
    For an 80% interval (alpha=0.20), extracts 10th and 90th percentiles of signed residuals.
    """
    residuals = y_true_val - y_pred_val
    alpha = 1.0 - coverage
    lower_pct = (alpha / 2.0) * 100.0
    upper_pct = (1.0 - alpha / 2.0) * 100.0

    lower_residual = float(np.percentile(residuals, lower_pct))
    upper_residual = float(np.percentile(residuals, upper_pct))

    return {
        "coverage": coverage,
        "lower_quantile": lower_pct,
        "upper_quantile": upper_pct,
        "lower_residual_offset": round(lower_residual, 2),
        "upper_residual_offset": round(upper_residual, 2),
        "typical_margin_of_error": round(float(np.median(np.abs(residuals))), 2)
    }
