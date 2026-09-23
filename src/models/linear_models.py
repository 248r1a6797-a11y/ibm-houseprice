"""Linear and regularized regression models for HouseWise AI."""
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from typing import Dict, Any

def get_linear_models(random_state: int = 42) -> Dict[str, Any]:
    """Return dictionary of linear and regularized regressors."""
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=10.0, random_state=random_state),
        "Lasso Regression": Lasso(alpha=0.001, max_iter=5000, random_state=random_state),
        "ElasticNet": ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000, random_state=random_state)
    }
