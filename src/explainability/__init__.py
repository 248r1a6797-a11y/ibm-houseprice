"""Model explainability package supporting SHAP and permutation importance."""
from .shap_analysis import compute_global_shap, explain_single_prediction
from .permutation_importance import compute_permutation_importance

__all__ = [
    "compute_global_shap",
    "explain_single_prediction",
    "compute_permutation_importance"
]
