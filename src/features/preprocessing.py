"""Reusable, leakage-free preprocessing pipeline builder."""
from typing import List, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import logging

logger = logging.getLogger(__name__)

EXCLUDE_COLUMNS = [
    "Order", "PID", "SalePrice", "PricePerSqFt", "IsExtremeOutlier"
]

def get_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Partition columns into numerical and categorical lists, excluding target and IDs."""
    candidate_cols = [c for c in df.columns if c not in EXCLUDE_COLUMNS]
    
    num_cols = []
    cat_cols = []
    for col in candidate_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            num_cols.append(col)
        else:
            cat_cols.append(col)
            
    return num_cols, cat_cols

def build_preprocessing_pipeline(
    num_cols: List[str],
    cat_cols: List[str],
    scale_numeric: bool = True
) -> ColumnTransformer:
    """
    Construct a scikit-learn ColumnTransformer for numerical and categorical features.
    Guarantees:
    - Numerical: Median imputation, optional standard scaling.
    - Categorical: Constant 'None' imputation, OneHotEncoding with handle_unknown='ignore'.
    """
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    numeric_pipe = Pipeline(num_steps)

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols)
        ],
        remainder="drop"
    )

    return preprocessor
