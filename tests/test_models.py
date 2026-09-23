"""Test candidate model pipelines and evaluation metrics."""
import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from src.features.preprocessing import get_feature_columns, build_preprocessing_pipeline
from src.models.registry import get_all_candidate_models, build_full_pipeline
from src.evaluation.metrics import compute_regression_metrics

def test_regression_metrics():
    """Verify RMSE, MAE, R2 calculation on known values."""
    y_true = np.array([100000.0, 200000.0, 300000.0])
    y_pred = np.array([110000.0, 190000.0, 310000.0])

    m = compute_regression_metrics(y_true, y_pred)
    assert m["MAE"] == 10000.0
    assert m["RMSE"] == 10000.0
    assert m["R2"] > 0.95

def test_candidate_models_compilation():
    """Verify model factory instantiates regressors across all families."""
    models = get_all_candidate_models(random_state=42, include_ensembles=True)
    assert "Ridge Regression" in models
    assert "Random Forest" in models
    assert "Hist Gradient Boosting" in models
    assert "XGBoost" in models
    assert "Stacking Ensemble" in models
    assert len(models) >= 12
