"""Domain feature engineering for HouseWise AI."""
import pandas as pd
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def engineer_housing_features(df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
    """
    Compute domain-specific features without leaking future or target information.
    - Historical age uses YrSold (sale date), avoiding current year leakage.
    - Replaces negative age anomalies (new construction sold mid-year) with 0.
    """
    df_feat = df.copy()

    # Total Living & Basement Square Footage
    gr_liv = df_feat["GrLivArea"] if "GrLivArea" in df_feat.columns else 0.0
    bsmt_sf = df_feat["TotalBsmtSF"] if "TotalBsmtSF" in df_feat.columns else 0.0
    df_feat["TotalSF"] = gr_liv + bsmt_sf

    # Fractional Combined Bathrooms
    full_bath = df_feat["FullBath"] if "FullBath" in df_feat.columns else 0.0
    half_bath = df_feat["HalfBath"] if "HalfBath" in df_feat.columns else 0.0
    bsmt_full = df_feat["BsmtFullBath"] if "BsmtFullBath" in df_feat.columns else 0.0
    bsmt_half = df_feat["BsmtHalfBath"] if "BsmtHalfBath" in df_feat.columns else 0.0
    df_feat["TotalBath"] = full_bath + (0.5 * half_bath) + bsmt_full + (0.5 * bsmt_half)

    # Total Combined Porch Area
    open_porch = df_feat["OpenPorchSF"] if "OpenPorchSF" in df_feat.columns else 0.0
    enc_porch = df_feat["EnclosedPorch"] if "EnclosedPorch" in df_feat.columns else 0.0
    three_porch = df_feat["3SsnPorch"] if "3SsnPorch" in df_feat.columns else 0.0
    screen_porch = df_feat["ScreenPorch"] if "ScreenPorch" in df_feat.columns else 0.0
    df_feat["TotalPorchSF"] = open_porch + enc_porch + three_porch + screen_porch

    # Property Age at Time of Sale
    if "YrSold" in df_feat.columns and "YearBuilt" in df_feat.columns:
        age = df_feat["YrSold"] - df_feat["YearBuilt"]
        # Handle cases where property sold in construction year before completion
        df_feat["HouseAgeAtSale"] = np.maximum(0, age)
    elif "YearBuilt" in df_feat.columns:
        # Fallback for inference if YrSold not specified: use baseline 2010
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

    # Binary amenity flags
    if "GarageCars" in df_feat.columns:
        df_feat["HasGarage"] = (df_feat["GarageCars"] > 0).astype(int)
    if "TotalBsmtSF" in df_feat.columns:
        df_feat["HasBasement"] = (df_feat["TotalBsmtSF"] > 0).astype(int)
    if "PoolArea" in df_feat.columns:
        df_feat["HasPool"] = (df_feat["PoolArea"] > 0).astype(int)

    # Derived descriptive metric: Price per Sq Ft (Descriptive ONLY, never an ML feature)
    if "SalePrice" in df_feat.columns and "GrLivArea" in df_feat.columns:
        valid_area = df_feat["GrLivArea"] > 0
        df_feat["PricePerSqFt"] = np.where(valid_area, df_feat["SalePrice"] / df_feat["GrLivArea"], np.nan)

    return df_feat
