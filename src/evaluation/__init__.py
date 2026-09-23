"""Model evaluation, cross-validation, and error diagnostics package."""
from .metrics import compute_regression_metrics
from .cross_validation import evaluate_model_cv
from .error_analysis import (
    compute_residuals,
    analyze_worst_predictions,
    analyze_errors_by_segment,
    compute_calibrated_prediction_intervals
)

__all__ = [
    "compute_regression_metrics",
    "evaluate_model_cv",
    "compute_residuals",
    "analyze_worst_predictions",
    "analyze_errors_by_segment",
    "compute_calibrated_prediction_intervals"
]
