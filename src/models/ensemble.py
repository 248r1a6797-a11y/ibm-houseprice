"""Ensemble and meta-learning models for HouseWise AI."""
from sklearn.ensemble import VotingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from typing import Dict, Any, List, Tuple

def get_ensemble_models(
    base_estimators: List[Tuple[str, Any]],
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Build Voting and Stacking ensembles from evaluated base estimators.
    Stacking uses Ridge meta-learner with 5-fold cross validation on training data.
    """
    if not base_estimators or len(base_estimators) < 2:
        return {}

    voting = VotingRegressor(estimators=base_estimators, n_jobs=-1)
    stacking = StackingRegressor(
        estimators=base_estimators,
        final_estimator=Ridge(alpha=10.0, random_state=random_state),
        cv=5,
        n_jobs=-1
    )

    return {
        "Voting Ensemble": voting,
        "Stacking Ensemble": stacking
    }
