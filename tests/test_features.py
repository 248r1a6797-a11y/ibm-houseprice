"""Test feature engineering formulas and anti-leakage logic."""
import pytest
import pandas as pd
import numpy as np
from src.features.engineering import engineer_housing_features

def test_engineered_features_formulas():
    """Verify feature formulas for TotalSF, TotalBath, HouseAgeAtSale."""
    df = pd.DataFrame({
        "GrLivArea": [1500],
        "TotalBsmtSF": [800],
        "FullBath": [2],
        "HalfBath": [1],
        "BsmtFullBath": [1],
        "BsmtHalfBath": [0],
        "OpenPorchSF": [50],
        "EnclosedPorch": [20],
        "3SsnPorch": [0],
        "ScreenPorch": [0],
        "YrSold": [2008],
        "YearBuilt": [2000],
        "YearRemodAdd": [2004],
        "OverallQual": [7],
        "OverallCond": [6]
    })
    feat_df = engineer_housing_features(df, is_training=False)

    assert feat_df["TotalSF"].iloc[0] == 2300
    assert feat_df["TotalBath"].iloc[0] == 3.5
    assert feat_df["TotalPorchSF"].iloc[0] == 70
    assert feat_df["HouseAgeAtSale"].iloc[0] == 8
    assert feat_df["RemodAgeAtSale"].iloc[0] == 4
    assert feat_df["IsRemodeled"].iloc[0] == 1
    assert feat_df["OverallGradeScore"].iloc[0] == 42

def test_negative_age_handling():
    """Verify new construction sold in build year before completion does not result in negative age."""
    df = pd.DataFrame({
        "YrSold": [2007],
        "YearBuilt": [2008]
    })
    feat_df = engineer_housing_features(df, is_training=False)
    assert feat_df["HouseAgeAtSale"].iloc[0] == 0
