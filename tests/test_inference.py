"""Test production model inference and batch prediction."""
import pytest
import pandas as pd
from src.inference.predictor import (
    load_production_model,
    predict_single_property,
    predict_batch_dataframe
)

def test_production_model_loads():
    """Verify saved production model and metadata load properly."""
    model, metadata = load_production_model()
    assert model is not None
    assert "model_name" in metadata
    assert "metrics" in metadata

def test_predict_single_property():
    """Verify single property prediction generates finite positive estimate and intervals."""
    prop_data = {
        "GrLivArea": 1600,
        "LotArea": 9000,
        "OverallQual": 7,
        "OverallCond": 5,
        "YearBuilt": 2005,
        "YearRemodAdd": 2005,
        "TotalBsmtSF": 850,
        "FullBath": 2,
        "HalfBath": 1,
        "BedroomAbvGr": 3,
        "Neighborhood": "CollgCr"
    }
    res = predict_single_property(prop_data)
    assert res["predicted_price"] > 50000.0
    assert res["lower_bound"] < res["predicted_price"]
    assert res["upper_bound"] > res["predicted_price"]
    assert len(res["positive_drivers"]) > 0 or len(res["negative_drivers"]) > 0

def test_predict_batch_dataframe():
    """Verify batch DataFrame scoring appends predicted_price and bounds."""
    df = pd.DataFrame([
        {"GrLivArea": 1400, "OverallQual": 6, "YearBuilt": 1995, "Neighborhood": "NAmes"},
        {"GrLivArea": 2200, "OverallQual": 8, "YearBuilt": 2006, "Neighborhood": "NridgHt"}
    ])
    scored = predict_batch_dataframe(df)
    assert "predicted_price" in scored.columns
    assert "lower_bound" in scored.columns
    assert "upper_bound" in scored.columns
    assert len(scored) == 2
    assert (scored["predicted_price"] > 0).all()

def test_invalid_input_rejected():
    """Verify validator rejects negative living area."""
    bad_prop = {
        "GrLivArea": -500,
        "OverallQual": 7,
        "YearBuilt": 2000
    }
    with pytest.raises(ValueError) as excinfo:
        predict_single_property(bad_prop)
    assert "strictly greater than 0" in str(excinfo.value)
