"""Data cleaning and structural imputation for HouseWise AI."""
import pandas as pd
import numpy as np
import logging
from typing import Optional
from pathlib import Path
from ..config import FEATURE_CONFIG, TARGET_COLUMN

logger = logging.getLogger(__name__)

def clean_housing_data(
    df: pd.DataFrame,
    remove_extreme_outliers: bool = False,
    fill_structural_nulls: bool = True
) -> pd.DataFrame:
    """
    Clean raw Ames Housing data.
    - Imputes domain structural nulls (e.g. absence of garage/basement/pool).
    - Cleans whitespace in string fields.
    - Optionally flags or removes the 5 De Cock extreme outliers (GrLivArea > 4000).
    """
    df_clean = df.copy()

    # Strip whitespace from string columns
    for col in df_clean.select_dtypes(include=["object"]).columns:
        df_clean[col] = df_clean[col].astype(str).str.strip()
        # Convert string 'nan', 'NA', 'None' back to proper NaN
        df_clean[col] = df_clean[col].replace(["nan", "NA", "None", ""], np.nan)

    if fill_structural_nulls:
        struct_cfg = FEATURE_CONFIG.get("structural_missing_imputations", {})
        cat_none_cols = struct_cfg.get("categorical_none", [])
        num_zero_cols = struct_cfg.get("numeric_zero", [])

        # Normalize column lookup names
        norm_map = {c.replace(" ", "").replace("/", "").lower(): c for c in df_clean.columns}

        for cat_col in cat_none_cols:
            clean_key = cat_col.replace(" ", "").replace("/", "").lower()
            if clean_key in norm_map:
                actual_col = norm_map[clean_key]
                df_clean[actual_col] = df_clean[actual_col].fillna("None")

        for num_col in num_zero_cols:
            clean_key = num_col.replace(" ", "").replace("/", "").lower()
            if clean_key in norm_map:
                actual_col = norm_map[clean_key]
                df_clean[actual_col] = df_clean[actual_col].fillna(0.0)

    # Fix clerical data entry errors (e.g. GarageYrBlt = 2207 typo for 2007)
    if "GarageYrBlt" in df_clean.columns:
        typo_mask = df_clean["GarageYrBlt"] > 2030
        if typo_mask.any():
            logger.info(f"Correcting {typo_mask.sum()} clerical errors in GarageYrBlt (>2030)...")
            # If YearBuilt is available, use YearBuilt or 2007
            df_clean.loc[typo_mask & (df_clean["GarageYrBlt"] == 2207), "GarageYrBlt"] = 2007.0
            # For any remaining > 2030, cap at YearBuilt
            df_clean.loc[df_clean["GarageYrBlt"] > 2030, "GarageYrBlt"] = df_clean.loc[df_clean["GarageYrBlt"] > 2030, "YearBuilt"]

    # Impute LotFrontage by Neighborhood median if Neighborhood exists
    if "LotFrontage" in df_clean.columns and "Neighborhood" in df_clean.columns:
        df_clean["LotFrontage"] = df_clean.groupby("Neighborhood")["LotFrontage"].transform(
            lambda x: x.fillna(x.median())
        )
        # Any remaining fill with overall median
        df_clean["LotFrontage"] = df_clean["LotFrontage"].fillna(df_clean["LotFrontage"].median())

    # Audit De Cock outliers: GrLivArea > 4000
    if "GrLivArea" in df_clean.columns:
        outlier_mask = df_clean["GrLivArea"] > 4000
        outlier_count = int(outlier_mask.sum())
        logger.info(f"Dean De Cock Outlier Audit: Found {outlier_count} properties with GrLivArea > 4,000 sq ft.")
        df_clean["IsExtremeOutlier"] = outlier_mask.astype(int)

        if remove_extreme_outliers and outlier_count > 0:
            logger.warning(f"Removing {outlier_count} extreme outlier observations (>4000 sq ft) per Dean De Cock recommendation.")
            df_clean = df_clean[~outlier_mask].reset_index(drop=True)

    return df_clean
