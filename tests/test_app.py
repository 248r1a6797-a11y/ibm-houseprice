"""Test Streamlit application modules, state engine, and 21 modular pages."""
import pytest
import importlib
import pandas as pd
from unittest.mock import MagicMock

def test_all_21_pages_import_cleanly():
    """Verify all 21 modular workbench pages import without error and have render_page."""
    page_modules = [
        "01_data_workspace",
        "02_data_profiling",
        "03_data_cleaning_studio",
        "04_feature_engineering_studio",
        "05_eda_laboratory",
        "06_experiment_setup",
        "07_model_laboratory",
        "08_hyperparameter_tuning",
        "09_experiment_history",
        "10_model_explainability",
        "11_property_predictor",
        "12_what_if_simulator",
        "13_prediction_diagnostics",
        "14_market_analytics",
        "15_neighborhood_intelligence",
        "16_risk_analysis",
        "17_insights_actions",
        "18_executive_overview",
        "19_sql_studio",
        "20_methodology",
        "21_documentation"
    ]
    for page_name in page_modules:
        mod = importlib.import_module(f"app.pages.{page_name}")
        assert mod is not None
        assert hasattr(mod, "render_page"), f"{page_name} must implement render_page()"

def test_components_import_cleanly():
    """Verify reusable Streamlit components import properly."""
    comp_modules = [
        "app.components.header",
        "app.components.kpi_cards",
        "app.components.pipeline_banner"
    ]
    for comp in comp_modules:
        mod = importlib.import_module(comp)
        assert mod is not None

def test_session_state_lifecycle_and_invalidation():
    """Verify session state initialization and reactive downstream invalidation."""
    from app.state import init_session_state, invalidate_downstream, reset_session_state

    # Mock Streamlit session state
    class MockSessionState(dict):
        def __getattr__(self, key):
            try:
                return self[key]
            except KeyError:
                raise AttributeError(key)

        def __setattr__(self, key, value):
            self[key] = value

        def __delattr__(self, key):
            try:
                del self[key]
            except KeyError:
                raise AttributeError(key)

    mock_st = MagicMock()
    mock_st.session_state = MockSessionState()

    # 1. Initialize
    init_session_state(mock_st)
    assert mock_st.session_state.initialized is True
    assert "raw_df" in mock_st.session_state
    assert "target_col" in mock_st.session_state
    assert "pipeline_status" in mock_st.session_state
    assert mock_st.session_state.needs_retraining is False

    # 2. Invalidate on cleaning change
    invalidate_downstream(mock_st, "cleaning")
    assert mock_st.session_state.needs_retraining is True
    assert mock_st.session_state.pipeline_status["TRAINING"] == "⏳ Pending"
    assert mock_st.session_state.pipeline_status["PRODUCTION"] == "⏳ Pending"

    # 3. Invalidate on data change
    mock_st.session_state.needs_retraining = False
    invalidate_downstream(mock_st, "data")
    assert mock_st.session_state.needs_retraining is True
    assert mock_st.session_state.pipeline_status["CLEANING"] == "⏳ Pending"
    assert mock_st.session_state.pipeline_status["FEATURES"] == "⏳ Pending"

    # 4. Reset
    reset_session_state(mock_st)
    assert mock_st.session_state.initialized is True
    assert mock_st.session_state.needs_retraining is False
    assert mock_st.session_state.pipeline_status["TRAINING"] == "✓ Completed"
