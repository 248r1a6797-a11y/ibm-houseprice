"""Data cleaning and structural imputation for HouseWise AI."""
import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
from ..config import FEATURE_CONFIG, TARGET_COLUMN

logger = logging.getLogger(__name__)

def clean_housing_data(
    df: pd.DataFrame,
    remove_extreme_outliers: bool = False,
    fill_structural_nulls: bool = True
) -> pd.DataFrame:
    """
    Clean housing data with Ames-specific domain structural rules when applicable.
    - Imputes domain structural nulls (e.g. absence of garage/basement/pool).
    - Cleans whitespace in string fields.
    - Optionally flags or removes extreme outliers (GrLivArea > 4000).
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
            df_clean.loc[typo_mask & (df_clean["GarageYrBlt"] == 2207), "GarageYrBlt"] = 2007.0
            if "YearBuilt" in df_clean.columns:
                df_clean.loc[df_clean["GarageYrBlt"] > 2030, "GarageYrBlt"] = df_clean.loc[df_clean["GarageYrBlt"] > 2030, "YearBuilt"]

    # Impute LotFrontage by Neighborhood median if Neighborhood exists
    if "LotFrontage" in df_clean.columns and "Neighborhood" in df_clean.columns:
        neigh_median = df_clean.groupby("Neighborhood")["LotFrontage"].transform("median")
        df_clean["LotFrontage"] = df_clean["LotFrontage"].fillna(neigh_median)
        df_clean["LotFrontage"] = df_clean["LotFrontage"].fillna(df_clean["LotFrontage"].median())

    # De Cock 5 extreme outliers filter (optional)
    if remove_extreme_outliers and "GrLivArea" in df_clean.columns and "SalePrice" in df_clean.columns:
        extreme_mask = (df_clean["GrLivArea"] > 4000) & (df_clean["SalePrice"] < 300000)
        n_outliers = extreme_mask.sum()
        if n_outliers > 0:
            logger.info(f"Removing {n_outliers} De Cock extreme outliers (GrLivArea > 4000 & SalePrice < 300k)...")
            df_clean = df_clean[~extreme_mask].reset_index(drop=True)

    return df_clean

def clean_dataset_adaptive(
    df: pd.DataFrame,
    options: Optional[Dict[str, Any]] = None,
    target_col: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Universal, schema-adaptive cleaning engine.
    Applies configurable deduplication, imputation, and outlier treatment.
    Returns (cleaned_df, cleaning_audit_dict).
    """
    if df.empty:
        return df, {"initial_rows": 0, "clean_rows": 0, "duplicates_removed": 0, "nulls_imputed": 0, "outliers_treated": 0}

    opts = options or {}
    num_strat = opts.get("num_impute_strategy", "Median")
    cat_strat = opts.get("cat_impute_strategy", "Most Frequent")
    remove_dup = opts.get("remove_duplicates", True)
    outlier_method = opts.get("outlier_method", "IQR")
    outlier_treatment = opts.get("outlier_treatment", "Keep")

    initial_rows = len(df)
    initial_nulls = int(df.isnull().sum().sum())
    
    cleaned = df.copy()

    # 1. Whitespace normalization
    for col in cleaned.select_dtypes(include=["object"]).columns:
        cleaned[col] = cleaned[col].astype(str).str.strip()
        cleaned[col] = cleaned[col].replace(["nan", "NA", "None", "null", "NULL", ""], np.nan)

    # 2. Deduplication
    dups_removed = 0
    if remove_dup:
        dup_mask = cleaned.duplicated()
        dups_removed = int(dup_mask.sum())
        if dups_removed > 0:
            cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    # 3. Missing Value Imputation
    for col in cleaned.columns:
        if cleaned[col].isnull().sum() == 0:
            continue
        if pd.api.types.is_numeric_dtype(cleaned[col]):
            if num_strat == "Mean":
                fill_val = cleaned[col].mean()
            elif num_strat == "Zero":
                fill_val = 0.0
            else: # Median
                fill_val = cleaned[col].median()
            cleaned[col] = cleaned[col].fillna(fill_val)
        else:
            if cat_strat == 'Constant "Missing"' or cat_strat == 'Constant "Unknown"':
                cleaned[col] = cleaned[col].fillna("Missing")
            else: # Most Frequent
                mode_val = cleaned[col].mode()
                fill_val = mode_val.iloc[0] if not mode_val.empty else "Missing"
                cleaned[col] = cleaned[col].fillna(fill_val)

    final_nulls = int(cleaned.isnull().sum().sum())
    nulls_imputed = max(0, initial_nulls - final_nulls)

    # 4. Outlier Handling on numerical columns (excluding ID columns)
    outliers_treated = 0
    if outlier_treatment in ["Cap / Winsorize", "Remove Records"]:
        num_cols = cleaned.select_dtypes(include=[np.number]).columns.tolist()
        # If target_col specified, include or treat target
        outlier_indices = set()
        
        for col in num_cols:
            if cleaned[col].nunique() <= 5:
                continue # Skip discrete or binary features
                
            series = cleaned[col]
            if "iqr" in outlier_method.lower():
                q25 = series.quantile(0.25)
                q75 = series.quantile(0.75)
                iqr = q75 - q25
                lower_bound = q25 - (1.5 * iqr)
                upper_bound = q75 + (1.5 * iqr)
            else: # Z-Score
                mean = series.mean()
                std = series.std()
                if std > 0:
                    lower_bound = mean - (3.0 * std)
                    upper_bound = mean + (3.0 * std)
                else:
                    continue

            # Non-negative guard for physical quantities
            if (series >= 0).all():
                lower_bound = max(0.0, lower_bound)

            mask = (series < lower_bound) | (series > upper_bound)
            count = int(mask.sum())
            if count > 0:
                if outlier_treatment == "Cap / Winsorize":
                    cleaned[col] = cleaned[col].clip(lower=lower_bound, upper=upper_bound)
                    outliers_treated += count
                elif outlier_treatment == "Remove Records":
                    outlier_indices.update(cleaned[mask].index.tolist())

        if outlier_treatment == "Remove Records" and outlier_indices:
            outliers_treated = len(outlier_indices)
            cleaned = cleaned.drop(index=list(outlier_indices)).reset_index(drop=True)

    audit = {
        "initial_rows": initial_rows,
        "clean_rows": len(cleaned),
        "duplicates_removed": dups_removed,
        "nulls_imputed": nulls_imputed,
        "outliers_treated": outliers_treated
    }

    return cleaned, audit
