"""Baseline regression models for HouseWise AI."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor

class NeighborhoodMedianRegressor(BaseEstimator, RegressorMixin):
    """
    Predicts the median price of the property's neighborhood.
    Falls back to global training median if neighborhood was unseen during training.
    Strictly fitted ONLY on training data to prevent leakage.
    """
    def __init__(self, neighborhood_col: str = "Neighborhood"):
        self.neighborhood_col = neighborhood_col
        self.neighborhood_medians_ = {}
        self.global_median_ = 0.0

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        y_arr = np.asarray(y, dtype=float)
        self.global_median_ = float(np.median(y_arr))

        if isinstance(X, pd.DataFrame) and self.neighborhood_col in X.columns:
            temp_df = pd.DataFrame({
                "neigh": X[self.neighborhood_col].astype(str),
                "target": y_arr
            })
            self.neighborhood_medians_ = temp_df.groupby("neigh")["target"].median().to_dict()
        else:
            self.neighborhood_medians_ = {}
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not isinstance(X, pd.DataFrame) or self.neighborhood_col not in X.columns:
            return np.full(len(X), self.global_median_)

        preds = [
            self.neighborhood_medians_.get(str(val), self.global_median_)
            for val in X[self.neighborhood_col]
        ]
        return np.asarray(preds, dtype=float)

def get_baseline_models():
    """Return dictionary of baseline estimators."""
    return {
        "Median Baseline": DummyRegressor(strategy="median"),
        "Mean Baseline": DummyRegressor(strategy="mean"),
        "Neighborhood Median Baseline": NeighborhoodMedianRegressor(neighborhood_col="Neighborhood")
    }
