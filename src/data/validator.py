"""Schema and data integrity validator for HouseWise AI."""
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
import logging
from ..config import TARGET_COLUMN

logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Custom exception raised when data validation fails."""
    pass

def validate_housing_schema(
    df: pd.DataFrame, 
    require_target: bool = True,
    target_col: Optional[str] = None
) -> Tuple[bool, List[str]]:
    """
    Validate dataset schema, data types, and logical bounds.
    Adaptive to arbitrary datasets while maintaining backward compatibility with Ames.
    Returns (is_valid, list_of_errors).
    """
    errors = []
    
    if df.empty:
        errors.append("Dataset contains 0 rows.")
        return False, errors

    # Check target
    target = target_col or (TARGET_COLUMN if TARGET_COLUMN in df.columns else (df.columns[-1] if len(df.columns) > 0 else None))
    if require_target and target:
        if target not in df.columns:
            errors.append(f"Required target column '{target}' is missing.")
        else:
            # Check target is numeric and positive
            if not pd.api.types.is_numeric_dtype(df[target]):
                errors.append(f"Target column '{target}' must be numeric, got {df[target].dtype}.")
            elif (df[target] <= 0).any():
                non_pos_count = int((df[target] <= 0).sum())
                errors.append(f"Target column '{target}' contains {non_pos_count} non-positive values.")

    # Check key numeric fields have no negative values where impossible
    non_negative_keywords = ["area", "size", "sf", "car", "room", "qual", "cond", "price", "bath", "bed"]
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            c_low = col.lower()
            if any(k in c_low for k in non_negative_keywords) and not c_low.startswith("residual") and not c_low.startswith("error"):
                neg_count = int((df[col] < 0).sum())
                if neg_count > 0:
                    errors.append(f"Column '{col}' has {neg_count} invalid negative values.")

    # Check year ranges if temporal columns detected
    year_keywords = ["year", "yr", "built", "vintage"]
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            c_low = col.lower()
            if any(k in c_low for k in year_keywords) and not "age" in c_low:
                series_non_null = df[col].dropna()
                if not series_non_null.empty and series_non_null.median() > 1800:
                    invalid_years = ((series_non_null < 1700) | (series_non_null > 2035))
                    if invalid_years.any():
                        errors.append(f"Column '{col}' contains {int(invalid_years.sum())} invalid years out of bounds [1700, 2035].")

    is_valid = len(errors) == 0
    return is_valid, errors

def validate_input_features(
    features: Dict[str, Any],
    expected_features: Optional[List[str]] = None,
    feature_distributions: Optional[Dict[str, Any]] = None
) -> Tuple[bool, List[str]]:
    """
    Validate a single property inference payload before sending to model pipeline.
    If expected_features is specified, validates against the active model's feature set.
    Otherwise checks generic Ames compatibility.
    """
    errors = []
    
    if expected_features:
        # Schema-adaptive validation: ensure numerical fields are valid numbers
        for f_name in expected_features:
            val = features.get(f_name)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                # If feature is in distributions, it will be imputed; else note warning
                continue
            if feature_distributions and f_name in feature_distributions:
                dist = feature_distributions[f_name]
                if dist.get("type") == "numeric":
                    try:
                        num_val = float(val)
                        if num_val < 0 and any(k in f_name.lower() for k in ["area", "size", "price", "bath", "bed", "car", "room"]):
                            errors.append(f"Feature '{f_name}' cannot be negative (got {num_val}).")
                    except (ValueError, TypeError):
                        errors.append(f"Feature '{f_name}' must be numeric (got {val}).")
        return len(errors) == 0, errors

    # Legacy Ames validation fallback if no expected_features passed
    living_area = features.get("GrLivArea") or features.get("Gr Liv Area")
    if living_area is not None:
        try:
            val = float(living_area)
            if val <= 0:
                errors.append("Above-ground living area (GrLivArea) must be strictly greater than 0.")
            elif val > 15000:
                errors.append("Above-ground living area exceeds realistic residential limit (15,000 sq ft).")
        except (ValueError, TypeError):
            errors.append("Above-ground living area must be a valid number.")
    elif "GrLivArea" in features:
        errors.append("Living area (GrLivArea) is required.")

    qual = features.get("OverallQual") or features.get("Overall Qual")
    if qual is not None:
        try:
            q_val = int(qual)
            if q_val < 1 or q_val > 10:
                errors.append("Overall Quality must be an integer between 1 and 10.")
        except (ValueError, TypeError):
            errors.append("Overall Quality must be an integer between 1 and 10.")

    yb = features.get("YearBuilt") or features.get("Year Built")
    if yb is not None:
        try:
            yb_val = int(yb)
            if yb_val < 1800 or yb_val > 2030:
                errors.append("Year Built must be between 1800 and 2030.")
        except (ValueError, TypeError):
            errors.append("Year Built must be a valid year.")

    return len(errors) == 0, errors
