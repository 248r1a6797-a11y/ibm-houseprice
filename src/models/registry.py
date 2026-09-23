"""Model registry and unified pipeline assembler for HouseWise AI."""
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor

from .baselines import get_baseline_models
from .linear_models import get_linear_models
from .tree_models import get_tree_models
from .boosting_models import get_boosting_models
from .neural_network import get_other_models
from .ensemble import get_ensemble_models
from ..features.preprocessing import build_preprocessing_pipeline

def build_full_pipeline(
    estimator: Any,
    preprocessor: Any,
    use_log_target: bool = True
) -> Any:
    """
    Assemble preprocessor and regressor into an atomic Pipeline.
    Optionally wraps in TransformedTargetRegressor using log1p/expm1.
    """
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", estimator)
    ])

    if use_log_target:
        # Wrap in TransformedTargetRegressor so fit/predict handles log/exp automatically
        return TransformedTargetRegressor(
            regressor=pipeline,
            func=np.log1p,
            inverse_func=np.expm1
        )
    return pipeline

def get_all_candidate_models(
    random_state: int = 42,
    include_ensembles: bool = True
) -> Dict[str, Any]:
    """Compile dictionary of all candidate regressors across all model families."""
    models = {}
    models.update(get_baseline_models())
    models.update(get_linear_models(random_state=random_state))
    models.update(get_tree_models(random_state=random_state))
    models.update(get_boosting_models(random_state=random_state))
    models.update(get_other_models(random_state=random_state))

    if include_ensembles:
        # Select representative diverse base estimators for stacking
        base_estimators = [
            ("ridge", get_linear_models(random_state)["Ridge Regression"]),
            ("rf", get_tree_models(random_state)["Random Forest"]),
            ("gb", get_boosting_models(random_state)["Gradient Boosting"])
        ]
        boosting_dict = get_boosting_models(random_state)
        if "XGBoost" in boosting_dict:
            base_estimators.append(("xgb", boosting_dict["XGBoost"]))
        elif "LightGBM" in boosting_dict:
            base_estimators.append(("lgb", boosting_dict["LightGBM"]))

        ensembles = get_ensemble_models(base_estimators, random_state=random_state)
        models.update(ensembles)

    return models
