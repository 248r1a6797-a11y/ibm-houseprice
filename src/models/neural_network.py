"""Neural network and non-parametric regressors for HouseWise AI."""
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from typing import Dict, Any

def get_other_models(random_state: int = 42) -> Dict[str, Any]:
    """Return dictionary containing MLP and KNN regressors."""
    return {
        "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=7, weights="distance"),
        "Artificial Neural Network (MLP)": MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            alpha=0.01,
            learning_rate_init=0.005,
            max_iter=300,
            early_stopping=True,
            random_state=random_state
        )
    }
