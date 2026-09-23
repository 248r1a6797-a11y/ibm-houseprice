"""Cross-validation benchmarking module for HouseWise AI."""
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.model_selection import KFold
import logging
from .metrics import compute_regression_metrics

logger = logging.getLogger(__name__)

def evaluate_model_cv(
    pipeline: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    cv_splits: int = 5,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Perform rigorous K-Fold cross-validation on the training set.
    All preprocessing and model fitting occur inside each fold.
    Metrics are computed on out-of-fold validation sets in original dollar ($) scale.
    """
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    y_arr = np.asarray(y, dtype=float)

    fold_metrics: List[Dict[str, float]] = []
    oof_predictions = np.zeros(len(y_arr))

    start_train_time = time.time()
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_arr)):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y_arr[train_idx], y_arr[val_idx]

        # Fit isolated pipeline on train fold
        pipeline.fit(X_train_fold, y_train_fold)

        # Predict on validation fold
        val_pred = pipeline.predict(X_val_fold)
        oof_predictions[val_idx] = val_pred

        metrics = compute_regression_metrics(y_val_fold, val_pred)
        fold_metrics.append(metrics)

    total_time = time.time() - start_train_time

    # Compute aggregate metrics across all out-of-fold predictions
    oof_metrics = compute_regression_metrics(y_arr, oof_predictions, prefix="OOF")

    rmse_list = [m["RMSE"] for m in fold_metrics]
    mae_list = [m["MAE"] for m in fold_metrics]
    r2_list = [m["R2"] for m in fold_metrics]

    return {
        "cv_rmse_mean": round(float(np.mean(rmse_list)), 2),
        "cv_rmse_std": round(float(np.std(rmse_list)), 2),
        "cv_mae_mean": round(float(np.mean(mae_list)), 2),
        "cv_mae_std": round(float(np.std(mae_list)), 2),
        "cv_r2_mean": round(float(np.mean(r2_list)), 4),
        "cv_r2_std": round(float(np.std(r2_list)), 4),
        "oof_rmse": oof_metrics["OOF_RMSE"],
        "oof_mae": oof_metrics["OOF_MAE"],
        "oof_r2": oof_metrics["OOF_R2"],
        "fold_metrics": fold_metrics,
        "total_cv_time_sec": round(total_time, 2)
    }
