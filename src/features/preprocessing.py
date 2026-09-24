"""Reusable, leakage-free preprocessing pipeline builder."""
from typing import List, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import logging
from ..data.schema import infer_target_column, infer_id_columns

logger = logging.getLogger(__name__)

DEFAULT_EXCLUDE_COLUMNS = [
    "Order", "PID", "SalePrice", "PricePerSqFt", "IsExtremeOutlier"
]

def get_feature_columns(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    exclude_cols: Optional[List[str]] = None
) -> Tuple[List[str], List[str]]:
    """
    Partition columns into numerical and categorical lists, strictly excluding target and IDs.
    Guarantees zero target leakage regardless of column naming.
    """
    # 1. Determine target to exclude
    effective_target = target_col or infer_target_column(df)
    
    # 2. Determine ID columns to exclude
    auto_ids = infer_id_columns(df)
    
    # 3. Compile full exclusion set
    exclusions = set(DEFAULT_EXCLUDE_COLUMNS)
    if effective_target:
        exclusions.add(effective_target)
    for id_c in auto_ids:
        exclusions.add(id_c)
    if exclude_cols:
        for c in exclude_cols:
            exclusions.add(c)

    candidate_cols = [c for c in df.columns if c not in exclusions]
    
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
    - Categorical: Constant 'Missing' imputation, OneHotEncoder with handle_unknown='ignore'
      and min_frequency=0.01 to prevent dimensional explosion on high cardinality columns.
    """
    transformers = []

    if num_cols:
        num_steps = [("imputer", SimpleImputer(strategy="median"))]
        if scale_numeric:
            num_steps.append(("scaler", StandardScaler()))
        numeric_pipe = Pipeline(num_steps)
        transformers.append(("num", numeric_pipe, num_cols))

    if cat_cols:
        # Use min_frequency=0.01 to bound categorical expansion gracefully
        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=0.01))
        ])
        transformers.append(("cat", categorical_pipe, cat_cols))

    if not transformers:
        # Fallback empty transformer
        transformers.append(("empty", "passthrough", []))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    return preprocessor
