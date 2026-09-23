"""Test Streamlit application modules and page imports."""
import pytest
import importlib

def test_app_pages_import_cleanly():
    """Verify all 12 modular Streamlit pages import without syntax or import errors."""
    page_modules = [
        "app.pages.01_executive_overview",
        "app.pages.02_market_analytics",
        "app.pages.03_property_predictor",
        "app.pages.04_model_lab",
        "app.pages.05_model_explainability",
        "app.pages.06_risk_analysis",
        "app.pages.07_insights_actions",
        "app.pages.08_data_quality",
        "app.pages.09_methodology",
        "app.pages.10_monitoring",
        "app.pages.11_documentation",
        "app.pages.12_about",
        "app.components.header",
        "app.components.kpi_cards"
    ]
    for mod_name in page_modules:
        mod = importlib.import_module(mod_name)
        assert mod is not None
