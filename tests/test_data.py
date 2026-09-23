"""Test data loading, validation, and cleaning."""
import pytest
import pandas as pd
import numpy as np
from src.config import RAW_DATA_PATH, TARGET_COLUMN
from src.data.loader import load_raw_data, normalize_columns
from src.data.validator import validate_housing_schema
from src.data.cleaner import clean_housing_data

def test_load_raw_data_exists():
    """Verify raw dataset exists and loads with correct dimensions."""
    df = load_raw_data(RAW_DATA_PATH, normalize=True)
    assert not df.empty
    assert len(df) == 2930
    assert TARGET_COLUMN in df.columns

def test_normalize_columns():
    """Test column normalization removes whitespace and slashes."""
    cols = pd.Index(["MS SubClass", "Year Remod/Add", "Sale Price "])
    norm = normalize_columns(cols)
    assert norm == ["MSSubClass", "YearRemodAdd", "SalePrice"]

def test_schema_validator_valid_data():
    """Verify schema validator passes on valid dataset."""
    df = load_raw_data(RAW_DATA_PATH, normalize=True)
    df_clean = clean_housing_data(df)
    is_valid, errors = validate_housing_schema(df_clean, require_target=True)
    assert is_valid
    assert len(errors) == 0

def test_schema_validator_catches_negative_target():
    """Verify schema validator rejects non-positive targets."""
    df = pd.DataFrame({
        "SalePrice": [100000, -5000, 200000],
        "GrLivArea": [1200, 1400, 1600]
    })
    is_valid, errors = validate_housing_schema(df, require_target=True)
    assert not is_valid
    assert any("non-positive" in e for e in errors)

def test_clean_housing_data_structural_nulls():
    """Verify structural nulls are filled with 'None' and 0."""
    df = pd.DataFrame({
        "PoolQC": [np.nan, "Ex"],
        "GarageCars": [np.nan, 2.0],
        "GarageYrBlt": [2207.0, 2005.0],
        "YearBuilt": [2006.0, 2005.0]
    })
    df_clean = clean_housing_data(df, fill_structural_nulls=True)
    assert df_clean["PoolQC"].iloc[0] == "None"
    assert df_clean["GarageCars"].iloc[0] == 0.0
    # Verify clerical error 2207 is corrected to 2007
    assert df_clean["GarageYrBlt"].iloc[0] == 2007.0
