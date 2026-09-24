"""Domain feature engineering for HouseWise AI."""
import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

def engineer_housing_features(df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
    """
    Compute domain-specific features without leaking future or target information.
    - If specific Ames columns exist, computes standard Ames domain features.
    - Avoids injecting zero-variance constant columns when source features are missing.
    - Historical age uses YrSold (sale date), avoiding current year leakage.
    - Replaces negative age anomalies (new construction sold mid-year) with 0.
    """
    df_feat = df.copy()

    # Total Living & Basement Square Footage (only if at least one area column exists)
    has_gr_liv = "GrLivArea" in df_feat.columns
    has_bsmt = "TotalBsmtSF" in df_feat.columns
    if has_gr_liv or has_bsmt:
        gr_liv = df_feat["GrLivArea"] if has_gr_liv else 0.0
        bsmt_sf = df_feat["TotalBsmtSF"] if has_bsmt else 0.0
        df_feat["TotalSF"] = gr_liv + bsmt_sf

    # Fractional Combined Bathrooms
    bath_cols = ["FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"]
    if any(c in df_feat.columns for c in bath_cols):
        full_bath = df_feat["FullBath"] if "FullBath" in df_feat.columns else 0.0
        half_bath = df_feat["HalfBath"] if "HalfBath" in df_feat.columns else 0.0
        bsmt_full = df_feat["BsmtFullBath"] if "BsmtFullBath" in df_feat.columns else 0.0
        bsmt_half = df_feat["BsmtHalfBath"] if "BsmtHalfBath" in df_feat.columns else 0.0
        df_feat["TotalBath"] = full_bath + (0.5 * half_bath) + bsmt_full + (0.5 * bsmt_half)

    # Total Combined Porch Area
    porch_cols = ["OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"]
    if any(c in df_feat.columns for c in porch_cols):
        open_porch = df_feat["OpenPorchSF"] if "OpenPorchSF" in df_feat.columns else 0.0
        enc_porch = df_feat["EnclosedPorch"] if "EnclosedPorch" in df_feat.columns else 0.0
        three_porch = df_feat["3SsnPorch"] if "3SsnPorch" in df_feat.columns else 0.0
        screen_porch = df_feat["ScreenPorch"] if "ScreenPorch" in df_feat.columns else 0.0
        df_feat["TotalPorchSF"] = open_porch + enc_porch + three_porch + screen_porch

    # Property Age at Time of Sale
    if "YrSold" in df_feat.columns and "YearBuilt" in df_feat.columns:
        age = df_feat["YrSold"] - df_feat["YearBuilt"]
        df_feat["HouseAgeAtSale"] = np.maximum(0, age)
    elif "YearBuilt" in df_feat.columns:
        df_feat["HouseAgeAtSale"] = np.maximum(0, 2010 - df_feat["YearBuilt"])

    # Remodel Age at Time of Sale
    if "YrSold" in df_feat.columns and "YearRemodAdd" in df_feat.columns:
        remod_age = df_feat["YrSold"] - df_feat["YearRemodAdd"]
        df_feat["RemodAgeAtSale"] = np.maximum(0, remod_age)
    elif "YearRemodAdd" in df_feat.columns:
        df_feat["RemodAgeAtSale"] = np.maximum(0, 2010 - df_feat["YearRemodAdd"])

    # Remodel Indicator
    if "YearRemodAdd" in df_feat.columns and "YearBuilt" in df_feat.columns:
        df_feat["IsRemodeled"] = (df_feat["YearRemodAdd"] != df_feat["YearBuilt"]).astype(int)

    # Quality & Condition Interaction Score
    if "OverallQual" in df_feat.columns and "OverallCond" in df_feat.columns:
        df_feat["OverallGradeScore"] = df_feat["OverallQual"] * df_feat["OverallCond"]

    # Binary amenity flags (only if column exists)
    if "PoolArea" in df_feat.columns:
        df_feat["HasPool"] = (df_feat["PoolArea"] > 0).astype(int)
    if "Fireplaces" in df_feat.columns:
        df_feat["HasFireplace"] = (df_feat["Fireplaces"] > 0).astype(int)
    if "GarageCars" in df_feat.columns:
        df_feat["HasGarage"] = (df_feat["GarageCars"] > 0).astype(int)

    return df_feat

def engineer_custom_algebraic_feature(
    df: pd.DataFrame, 
    new_col_name: str, 
    col1: str, 
    operator: str, 
    col2: str
) -> pd.DataFrame:
    """Safely constructs a custom algebraic feature from two existing numerical columns."""
    df_out = df.copy()
    if col1 not in df_out.columns or col2 not in df_out.columns:
        return df_out

    s1 = pd.to_numeric(df_out[col1], errors="coerce").fillna(0.0)
    s2 = pd.to_numeric(df_out[col2], errors="coerce").fillna(0.0)

    if operator == "+":
        df_out[new_col_name] = s1 + s2
    elif operator == "-":
        df_out[new_col_name] = s1 - s2
    elif operator == "*":
        df_out[new_col_name] = s1 * s2
    elif operator == "/":
        # Avoid division by zero
        s2_safe = np.where(s2 == 0, 1e-6, s2)
        df_out[new_col_name] = s1 / s2_safe
    else:
        logger.warning(f"Unsupported algebraic operator: {operator}")

    return df_out
