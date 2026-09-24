"""
HouseWise AI — Dynamic Dataset Context & Semantic Schema Inference Engine
========================================================================
Provides schema-adaptive introspection for arbitrary tabular housing datasets:
- Auto-detects candidate target variables (prices, values, costs).
- Auto-detects identifier/index columns to prevent leakage.
- Partitions features into continuous numeric, discrete numeric, categorical, and temporal.
- Identifies spatial (neighborhood/location), size (sqft/area), and age (vintage) roles.
- Computes distribution summaries for dynamic UI form and slider generation.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Common target column name patterns in real estate datasets
TARGET_NAME_PATTERNS = [
    "saleprice", "sale_price", "price", "val", "value", "cost", 
    "target", "medv", "housing_median_age", "median_house_value", 
    "listing_price", "actual_price", "house_price"
]

# Common ID column name patterns
ID_NAME_PATTERNS = ["id", "order", "pid", "index", "property_id", "parcel_id", "unnamed: 0"]

# Common spatial / neighborhood patterns
LOCATION_PATTERNS = ["neighborhood", "location", "suburb", "city", "district", "zipcode", "zip_code", "zone", "area_name", "region"]

# Common size / area patterns
SIZE_PATTERNS = ["grlivarea", "living_area", "sqft", "area", "size", "surface", "totalsf", "total_sqft", "lotarea", "lot_size"]

# Common vintage / age patterns
AGE_PATTERNS = ["yearbuilt", "year_built", "built", "age", "house_age", "yr_built", "vintage"]

@dataclass
class DatasetContext:
    """Encapsulates the active dataset, its inferred schema, and distribution metadata."""
    dataset_name: str
    target_col: str
    problem_type: str = "Regression"
    id_cols: List[str] = field(default_factory=list)
    num_cols: List[str] = field(default_factory=list)
    cat_cols: List[str] = field(default_factory=list)
    date_cols: List[str] = field(default_factory=list)
    selected_features: List[str] = field(default_factory=list)
    location_col: Optional[str] = None
    size_col: Optional[str] = None
    age_col: Optional[str] = None
    feature_distributions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    total_rows: int = 0
    total_cols: int = 0

def infer_target_column(df: pd.DataFrame) -> Optional[str]:
    """Smart candidate target identification based on naming heuristics and numeric types."""
    if df.empty:
        return None
        
    cols = list(df.columns)
    # Check for exact matches
    for col in cols:
        clean = col.lower().replace(" ", "").replace("_", "")
        for pat in TARGET_NAME_PATTERNS:
            if clean == pat and pd.api.types.is_numeric_dtype(df[col]):
                return col
                
    # Check for substring matches in numeric columns
    for col in cols:
        clean = col.lower().replace(" ", "").replace("_", "")
        for pat in TARGET_NAME_PATTERNS:
            if pat in clean and pd.api.types.is_numeric_dtype(df[col]):
                return col
                
    # Fallback to last numeric column
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    if num_cols:
        return num_cols[-1]
    return cols[-1]

def infer_id_columns(df: pd.DataFrame) -> List[str]:
    """Detects potential row identifiers or primary keys to prevent target leakage."""
    id_cols = []
    n_rows = len(df)
    for col in df.columns:
        c_clean = col.lower().strip()
        if any(c_clean == pat or c_clean.startswith(f"{pat}_") for pat in ID_NAME_PATTERNS):
            id_cols.append(col)
        elif n_rows > 10 and df[col].nunique() == n_rows and pd.api.types.is_integer_dtype(df[col]):
            # 100% unique integer sequence -> likely an arbitrary ID / index
            id_cols.append(col)
    return id_cols

def infer_semantic_roles(df: pd.DataFrame, candidate_cols: List[str]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Identifies candidate columns for location/submarket, area/size, and age/vintage."""
    loc_col = None
    size_col = None
    age_col = None

    for col in candidate_cols:
        c_clean = col.lower().replace(" ", "").replace("_", "")
        # Location candidate (must be categorical or object)
        if loc_col is None and any(pat in c_clean for pat in LOCATION_PATTERNS):
            if not pd.api.types.is_numeric_dtype(df[col]) or df[col].nunique() < 50:
                loc_col = col

        # Size candidate (must be numeric)
        if size_col is None and any(pat in c_clean for pat in SIZE_PATTERNS):
            if pd.api.types.is_numeric_dtype(df[col]):
                size_col = col

        # Age candidate (numeric)
        if age_col is None and any(pat in c_clean for pat in AGE_PATTERNS):
            if pd.api.types.is_numeric_dtype(df[col]):
                age_col = col

    return loc_col, size_col, age_col

def compute_feature_distributions(df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Dict[str, Any]]:
    """Calculates min, max, median, and categorical levels for runtime dynamic UI synthesis."""
    distributions = {}
    for col in feature_cols:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if series.empty:
            continue

        if pd.api.types.is_numeric_dtype(series):
            c_min = float(series.min())
            c_max = float(series.max())
            c_med = float(series.median())
            c_mean = float(series.mean())
            c_std = float(series.std()) if len(series) > 1 else 0.0

            # Reasonable step calculation for UI sliders/number inputs
            span = c_max - c_min
            if pd.api.types.is_integer_dtype(series) or (span > 0 and span < 20 and c_min.is_integer() and c_max.is_integer()):
                is_int = True
                step = 1
            else:
                is_int = False
                step = round(span / 100.0, 2) if span > 0 else 0.1
                if step == 0:
                    step = 0.01

            distributions[col] = {
                "type": "numeric",
                "is_integer": is_int,
                "min": c_min,
                "max": c_max,
                "median": c_med,
                "mean": c_mean,
                "std": c_std,
                "step": step,
                "default": c_med
            }
        else:
            # Categorical
            top_vals = series.value_counts().head(50).index.tolist()
            # Ensure strings
            categories = [str(v) for v in top_vals]
            distributions[col] = {
                "type": "categorical",
                "categories": categories if categories else ["None"],
                "default": str(series.mode().iloc[0]) if not series.empty else "None",
                "nunique": int(series.nunique())
            }

    return distributions

def build_dataset_context(
    df: pd.DataFrame,
    dataset_name: str = "Uploaded Dataset",
    target_col: Optional[str] = None
) -> DatasetContext:
    """Constructs a complete DatasetContext from a raw DataFrame."""
    if df.empty:
        return DatasetContext(dataset_name=dataset_name, target_col="")

    # 1. Infer Target
    if target_col is None or target_col not in df.columns:
        target_col = infer_target_column(df)

    # 2. Infer IDs
    id_cols = infer_id_columns(df)

    # 3. Partition remaining columns into numerical and categorical
    candidate_features = [c for c in df.columns if c != target_col and c not in id_cols]
    num_cols = []
    cat_cols = []
    date_cols = []

    for col in candidate_features:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_cols.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            # If numeric has only 2 unique values and object-like, can be binary numeric
            num_cols.append(col)
        else:
            cat_cols.append(col)

    # 4. Infer semantic roles
    loc_col, size_col, age_col = infer_semantic_roles(df, candidate_features)

    # 5. Compute distribution metadata
    distributions = compute_feature_distributions(df, candidate_features)

    return DatasetContext(
        dataset_name=dataset_name,
        target_col=target_col,
        problem_type="Regression",
        id_cols=id_cols,
        num_cols=num_cols,
        cat_cols=cat_cols,
        date_cols=date_cols,
        selected_features=candidate_features,
        location_col=loc_col,
        size_col=size_col,
        age_col=age_col,
        feature_distributions=distributions,
        total_rows=len(df),
        total_cols=len(df.columns)
    )
