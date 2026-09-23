"""Regression evaluation metrics for HouseWise AI."""
import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    median_absolute_error
)

def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str = ""
) -> Dict[str, float]:
    """
    Calculate comprehensive regression metrics on the original price scale ($).
    Guarantees non-negative targets before computing logarithmic metrics.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Avoid zero or negative values in predictions for logarithmic metrics
    y_pred_clipped = np.clip(y_pred, a_min=1.0, a_max=None)
    y_true_clipped = np.clip(y_true, a_min=1.0, a_max=None)

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    med_ae = median_absolute_error(y_true, y_pred)

    # RMSLE (Root Mean Squared Logarithmic Error)
    rmsle = np.sqrt(mean_squared_error(np.log1p(y_true_clipped), np.log1p(y_pred_clipped)))

    # MAPE (safe version: percentage relative to actual)
    mape = np.mean(np.abs((y_true - y_pred) / y_true_clipped)) * 100.0

    p = f"{prefix}_" if prefix else ""
    return {
        f"{p}MAE": round(float(mae), 2),
        f"{p}RMSE": round(float(rmse), 2),
        f"{p}R2": round(float(r2), 4),
        f"{p}RMSLE": round(float(rmsle), 4),
        f"{p}MAPE_pct": round(float(mape), 2),
        f"{p}MedianAE": round(float(med_ae), 2)
    }
