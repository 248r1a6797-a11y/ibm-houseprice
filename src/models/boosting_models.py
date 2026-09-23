"""Gradient boosting regression models for HouseWise AI."""
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

def get_boosting_models(random_state: int = 42) -> Dict[str, Any]:
    """Return dictionary of gradient boosting regressors."""
    models = {
        "Hist Gradient Boosting": HistGradientBoostingRegressor(
            max_iter=150,
            learning_rate=0.08,
            max_leaf_nodes=31,
            random_state=random_state
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=4,
            subsample=0.85,
            random_state=random_state
        )
    }

    # Add XGBoost if available
    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = XGBRegressor(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=4,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=random_state,
            n_jobs=-1
        )
    except ImportError:
        logger.warning("XGBoost not available; omitting from boosting suite.")

    # Add LightGBM if available
    try:
        from lightgbm import LGBMRegressor
        models["LightGBM"] = LGBMRegressor(
            n_estimators=150,
            learning_rate=0.08,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1
        )
    except ImportError:
        logger.warning("LightGBM not available; omitting from boosting suite.")

    return models
