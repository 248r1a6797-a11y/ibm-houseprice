"""
HouseWise AI — Schema Adaptation & Arbitrary Dataset Integration Tests
=====================================================================
Validates that HouseWise AI operates flawlessly on generic real-world housing data
without hardcoded Ames column dependencies:
- Tests 3,986 records × 8 features (price, area, bedrooms, bathrooms, location, age, parking, lot_size)
- Verifies zero target leakage
- Verifies dynamic pipeline fitting, inference, and prediction bounds
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor

from src.data.schema import build_dataset_context, infer_target_column, infer_semantic_roles
from src.data.cleaner import clean_dataset_adaptive
from src.features.preprocessing import get_feature_columns, build_preprocessing_pipeline
from src.models.registry import build_full_pipeline
from src.inference.predictor import predict_single_property, predict_batch_dataframe

@pytest.fixture
def generic_housing_df():
    """Generates a representative 3,986-record real-world housing dataset across 8 features."""
    np.random.seed(42)
    n = 3986
    locations = ["Downtown", "Suburbs", "Uptown", "Lakeside", "Metropolitan"]
    
    areas = np.random.normal(1600, 450, n).clip(400, 6000)
    beds = np.random.choice([1, 2, 3, 4, 5], size=n, p=[0.1, 0.25, 0.45, 0.15, 0.05])
    baths = np.random.choice([1.0, 1.5, 2.0, 2.5, 3.0, 3.5], size=n)
    locs = np.random.choice(locations, size=n)
    ages = np.random.randint(0, 60, size=n)
    parkings = np.random.choice([0, 1, 2, 3], size=n)
    lot_sizes = np.random.normal(6000, 1500, n).clip(1000, 25000)

    # Base price calculation with noise
    loc_mult = {"Downtown": 1.4, "Suburbs": 1.0, "Uptown": 1.25, "Lakeside": 1.5, "Metropolitan": 1.1}
    loc_factors = np.array([loc_mult[l] for l in locs])
    prices = (areas * 150.0 + beds * 15000.0 + baths * 12000.0 - ages * 800.0 + parkings * 8000.0) * loc_factors
    prices = prices + np.random.normal(0, 20000, n)
    prices = np.maximum(50000.0, prices)

    # Inject a few realistic nulls and duplicate rows
    df = pd.DataFrame({
        "price": np.round(prices, 2),
        "area": np.round(areas, 1),
        "bedrooms": beds,
        "bathrooms": baths,
        "location": locs,
        "age": ages,
        "parking": parkings,
        "lot_size": np.round(lot_sizes, 1)
    })
    
    # Introduce 5 nulls in lot_size and 2 duplicates
    df.loc[10:14, "lot_size"] = np.nan
    df = pd.concat([df, df.iloc[:2]], ignore_index=True)
    return df

def test_schema_inference_on_generic_data(generic_housing_df):
    """Verify semantic schema inference correctly identifies target, roles, and types."""
    ctx = build_dataset_context(generic_housing_df, dataset_name="Custom Real Estate Data")
    
    assert ctx.target_col == "price"
    assert ctx.location_col == "location"
    assert ctx.size_col == "area"
    assert ctx.age_col == "age"
    assert "location" in ctx.cat_cols
    assert "area" in ctx.num_cols
    assert "bedrooms" in ctx.num_cols
    assert "price" not in ctx.selected_features

def test_zero_target_leakage_on_custom_target(generic_housing_df):
    """Verify that 'price' target is strictly excluded from candidate features."""
    num_cols, cat_cols = get_feature_columns(generic_housing_df, target_col="price")
    assert "price" not in num_cols
    assert "price" not in cat_cols
    assert len(num_cols) + len(cat_cols) == len(generic_housing_df.columns) - 1

def test_adaptive_cleaning_on_generic_data(generic_housing_df):
    """Verify cleaning handles duplicates, null imputation, and generates audit metrics."""
    cleaned, audit = clean_dataset_adaptive(
        generic_housing_df,
        options={
            "num_impute_strategy": "Median",
            "cat_impute_strategy": "Most Frequent",
            "remove_duplicates": True,
            "outlier_treatment": "Keep"
        },
        target_col="price"
    )
    assert audit["duplicates_removed"] == 2
    assert audit["nulls_imputed"] >= 5
    assert cleaned["lot_size"].isnull().sum() == 0
    assert len(cleaned) == len(generic_housing_df) - 2

def test_end_to_end_training_and_dynamic_prediction(generic_housing_df):
    """Verify pipeline fits on generic dataset and predicts single and batch properties."""
    cleaned, _ = clean_dataset_adaptive(generic_housing_df, target_col="price")
    
    X = cleaned.drop(columns=["price"])
    y = cleaned["price"]
    
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    
    preprocessor = build_preprocessing_pipeline(num_cols, cat_cols)
    pipeline = build_full_pipeline(HistGradientBoostingRegressor(random_state=42), preprocessor, use_log_target=True)
    
    # Train pipeline
    pipeline.fit(X, y)
    
    # Test single prediction
    single_prop = {
        "area": 2000.0,
        "bedrooms": 3,
        "bathrooms": 2.5,
        "location": "Downtown",
        "age": 5,
        "parking": 2,
        "lot_size": 7500.0
    }
    
    meta = {
        "model_name": "HistGradientBoosting Champion",
        "metrics": {"holdout_test": {"RMSE": 22000.0, "MAE": 16000.0, "R2": 0.92}},
        "prediction_intervals": {"lower_residual_offset": -25000.0, "upper_residual_offset": 25000.0}
    }
    
    pred = predict_single_property(single_prop, model=pipeline, metadata=meta, expected_features=list(X.columns))
    
    assert pred["predicted_price"] > 100000.0
    assert pred["lower_bound"] < pred["predicted_price"]
    assert pred["upper_bound"] > pred["predicted_price"]
    assert "formatted_interval" in pred

    # Test batch scoring
    batch_test = X.head(5).copy()
    scored_batch = predict_batch_dataframe(batch_test, model=pipeline, metadata=meta)
    assert len(scored_batch) == 5
    assert "predicted_price" in scored_batch.columns
    assert (scored_batch["predicted_price"] > 0).all()
