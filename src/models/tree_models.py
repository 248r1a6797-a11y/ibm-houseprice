"""Decision tree and tree-ensemble models for HouseWise AI."""
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from typing import Dict, Any

def get_tree_models(random_state: int = 42) -> Dict[str, Any]:
    """Return dictionary of tree-based estimators."""
    return {
        "Decision Tree": DecisionTreeRegressor(
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=random_state
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=150,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=random_state
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=150,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=random_state
        )
    }
