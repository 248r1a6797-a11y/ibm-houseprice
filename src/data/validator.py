"""Schema and data integrity validator for HouseWise AI."""
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
import logging
from ..config import TARGET_COLUMN

logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Custom exception raised when data validation fails."""
    pass

def validate_housing_schema(df: pd.DataFrame, require_target: bool = True) -> Tuple[bool, List[str]]:
    """
    Validate dataset schema, data types, and logical bounds.
    Returns (is_valid, list_of_errors).
    """
    errors = []
    
    if df.empty:
        errors.append("Dataset contains 0 rows.")
        return False, errors

    # Check target
    target = TARGET_COLUMN
    if require_target:
        if target not in df.columns:
            errors.append(f"Required target column '{target}' is missing.")
        else:
            # Check target is numeric and positive
            if not pd.api.types.is_numeric_dtype(df[target]):
                errors.append(f"Target column '{target}' must be numeric, got {df[target].dtype}.")
            elif (df[target] <= 0).any():
                non_pos_count = (df[target] <= 0).sum()
                errors.append(f"Target column '{target}' contains {non_pos_count} non-positive values.")

    # Check key numeric fields have no negative values where impossible
    non_negative_fields = [
        "GrLivArea", "LotArea", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", 
        "GarageArea", "GarageCars", "OverallQual", "OverallCond"
    ]
    for col in non_negative_fields:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            neg_count = (df[col] < 0).sum()
            if neg_count > 0:
                errors.append(f"Column '{col}' has {neg_count} invalid negative values.")

    # Check year ranges
    year_fields = ["YearBuilt", "YearRemodAdd", "GarageYrBlt", "YrSold"]
    for y_col in year_fields:
        if y_col in df.columns and pd.api.types.is_numeric_dtype(df[y_col]):
            invalid_years = ((df[y_col] < 1800) | (df[y_col] > 2030)) & df[y_col].notnull()
            if invalid_years.any():
                errors.append(f"Column '{y_col}' contains invalid years out of bounds [1800, 2030].")

    is_valid = len(errors) == 0
    return is_valid, errors

def validate_input_features(features: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate a single property inference payload before sending to model pipeline.
    """
    errors = []
    
    # Must have positive living area
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
    else:
        errors.append("Living area (GrLivArea) is required.")

    # Overall quality must be between 1 and 10
    qual = features.get("OverallQual") or features.get("Overall Qual")
    if qual is not None:
        try:
            q_val = int(qual)
            if q_val < 1 or q_val > 10:
                errors.append("Overall Quality must be an integer between 1 and 10.")
        except (ValueError, TypeError):
            errors.append("Overall Quality must be an integer between 1 and 10.")

    # Year built check
    yb = features.get("YearBuilt") or features.get("Year Built")
    if yb is not None:
        try:
            yb_val = int(yb)
            if yb_val < 1800 or yb_val > 2030:
                errors.append("Year Built must be between 1800 and 2030.")
        except (ValueError, TypeError):
            errors.append("Year Built must be a valid year.")

    return len(errors) == 0, errors
