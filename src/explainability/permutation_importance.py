"""Permutation feature importance for HouseWise AI."""
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.inspection import permutation_importance
import logging

logger = logging.getLogger(__name__)

def compute_permutation_importance(
    model_pipeline: Any,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    n_repeats: int = 5,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Calculate model-agnostic permutation feature importances on holdout data.
    Measures increase in prediction error (RMSE) when each feature is randomly shuffled.
    """
    logger.info(f"Computing permutation importance on {len(X_val)} validation samples...")
    res = permutation_importance(
        model_pipeline,
        X_val,
        y_val,
        n_repeats=n_repeats,
        scoring="neg_root_mean_squared_error",
        random_state=random_state,
        n_jobs=-1
    )

    df_imp = pd.DataFrame({
        "feature": X_val.columns,
        "importance_mean": res.importances_mean,
        "importance_std": res.importances_std
    }).sort_values(by="importance_mean", ascending=False).reset_index(drop=True)

    return df_imp
