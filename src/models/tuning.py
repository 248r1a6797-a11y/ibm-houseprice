"""Hyperparameter optimization module for HouseWise AI."""
import numpy as np
import logging
from typing import Dict, Any
from sklearn.model_selection import RandomizedSearchCV, KFold

logger = logging.getLogger(__name__)

TUNING_PARAM_GRIDS = {
    "Ridge Regression": {
        "regressor__regressor__alpha": [0.1, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    },
    "Random Forest": {
        "regressor__regressor__n_estimators": [100, 150, 200],
        "regressor__regressor__max_depth": [10, 15, 20, None],
        "regressor__regressor__min_samples_split": [2, 5, 10],
        "regressor__regressor__min_samples_leaf": [1, 2, 4],
        "regressor__regressor__max_features": ["sqrt", 0.5, 0.8]
    },
    "Hist Gradient Boosting": {
        "regressor__regressor__max_iter": [100, 150, 200],
        "regressor__regressor__learning_rate": [0.03, 0.05, 0.08, 0.1],
        "regressor__regressor__max_leaf_nodes": [15, 31, 63],
        "regressor__regressor__min_samples_leaf": [10, 20, 30]
    },
    "Gradient Boosting": {
        "regressor__regressor__n_estimators": [100, 150, 200],
        "regressor__regressor__learning_rate": [0.03, 0.05, 0.08, 0.1],
        "regressor__regressor__max_depth": [3, 4, 5],
        "regressor__regressor__subsample": [0.75, 0.85, 1.0]
    },
    "XGBoost": {
        "regressor__regressor__n_estimators": [100, 150, 200],
        "regressor__regressor__learning_rate": [0.03, 0.05, 0.08, 0.1],
        "regressor__regressor__max_depth": [3, 4, 6],
        "regressor__regressor__subsample": [0.75, 0.85, 1.0],
        "regressor__regressor__colsample_bytree": [0.75, 0.85, 1.0]
    },
    "LightGBM": {
        "regressor__regressor__n_estimators": [100, 150, 200],
        "regressor__regressor__learning_rate": [0.03, 0.05, 0.08, 0.1],
        "regressor__regressor__num_leaves": [15, 31, 63],
        "regressor__regressor__subsample": [0.75, 0.85, 1.0],
        "regressor__regressor__colsample_bytree": [0.75, 0.85, 1.0]
    }
}

def tune_candidate_pipeline(
    pipeline: Any,
    model_name: str,
    X_train: Any,
    y_train: Any,
    n_iter: int = 15,
    cv: int = 5,
    random_state: int = 42
) -> Any:
    """Run controlled RandomizedSearchCV for a specified candidate model pipeline."""
    grid = TUNING_PARAM_GRIDS.get(model_name)
    if not grid:
        logger.info(f"No specific tuning grid defined for {model_name}; returning un-tuned model.")
        return pipeline

    # Check if pipeline is TransformedTargetRegressor or standard Pipeline
    is_ttr = hasattr(pipeline, "regressor")
    adjusted_grid = {}
    for param, values in grid.items():
        if not is_ttr and param.startswith("regressor__"):
            # Strip 'regressor__' prefix if standard pipeline
            clean_param = param.replace("regressor__regressor__", "regressor__")
            adjusted_grid[clean_param] = values
        else:
            adjusted_grid[param] = values

    logger.info(f"Initiating hyperparameter search for {model_name} (n_iter={n_iter}, cv={cv})...")
    cv_strategy = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=adjusted_grid,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=cv_strategy,
        random_state=random_state,
        n_jobs=-1,
        verbose=0
    )
    search.fit(X_train, y_train)
    logger.info(f"Tuning finished for {model_name}. Best CV Score: {-search.best_score_:.2f}")
    logger.info(f"Best Parameters: {search.best_params_}")
    return search.best_estimator_
