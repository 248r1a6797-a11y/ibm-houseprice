"""HouseWise AI — Master Streamlit Application."""
import json
import sys
from pathlib import Path
import streamlit as st
import pandas as pd

# Set page config as the very first Streamlit call
st.set_page_config(
    page_title="HouseWise AI — Housing Intelligence Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.data.loader import load_raw_data
from src.data.cleaner import clean_housing_data
from src.features.engineering import engineer_housing_features

# Import modular pages
import app.pages.01_executive_overview as p01
import app.pages.02_market_analytics as p02
import app.pages.03_property_predictor as p03
import app.pages.04_model_lab as p04
import app.pages.05_model_explainability as p05
import app.pages.06_risk_analysis as p06
import app.pages.07_insights_actions as p07
import app.pages.08_data_quality as p08
import app.pages.09_methodology as p09
import app.pages.10_monitoring as p10
import app.pages.11_documentation as p11
import app.pages.12_about as p12

@st.cache_data(show_spinner=False)
def load_cached_dataset():
    """Load cleaned, feature-engineered housing dataset."""
    raw = load_raw_data(ROOT_DIR / "data/raw/AmesHousing.csv", normalize=True)
    clean = clean_housing_data(raw, remove_extreme_outliers=False)
    feat = engineer_housing_features(clean, is_training=False)
    return feat

@st.cache_data(show_spinner=False)
def load_cached_metadata():
    """Load production model metadata."""
    meta_path = ROOT_DIR / "models/production/model_metadata.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

@st.cache_data(show_spinner=False)
def load_cached_model_comparison():
    """Load model benchmark experiment table."""
    comp_path = ROOT_DIR / "reports/tables/model_comparison.csv"
    if comp_path.exists():
        return pd.read_csv(comp_path)
    return pd.DataFrame()

def inject_custom_css():
    """Inject corporate theme CSS."""
    css_path = ROOT_DIR / "app/styles/theme.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def main():
    inject_custom_css()

    df = load_cached_dataset()
    metadata = load_cached_metadata()
    model_comp_df = load_cached_model_comparison()

    # Sidebar Navigation
    st.sidebar.markdown(
        """
        <div style="padding: 10px 0 20px 0;">
            <div style="font-size: 1.25rem; font-weight: 800; color: #1E3A8A; letter-spacing: -0.02em;">HOUSEWISE AI</div>
            <div style="font-size: 0.78rem; color: #64748B; font-weight: 500;">Explainable Property Intelligence</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    page_selection = st.sidebar.radio(
        "Navigation",
        [
            "1. Executive Overview",
            "2. Market Analytics",
            "3. Property Predictor",
            "4. Model Lab",
            "5. Model Explainability",
            "6. Risk & Error Analysis",
            "7. Insights & Actions",
            "8. Data Quality",
            "9. Methodology",
            "10. Model Monitoring",
            "11. Documentation",
            "12. About / Internship"
        ],
        index=0
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"""
        <div style="font-size: 0.75rem; color: #94A3B8; line-height: 1.5;">
            <b>Dataset</b>: Ames Housing (2,930 rows)<br/>
            <b>Production Model</b>: {metadata.get('model_name', 'XGBoost')}<br/>
            <b>Holdout Test R²</b>: {metadata.get('metrics', {}).get('holdout_test', {}).get('R2', 0.934):.1%}<br/>
            <b>Currency</b>: USD ($)
        </div>
        """,
        unsafe_allow_html=True
    )

    # Route Pages
    if page_selection == "1. Executive Overview":
        p01.render_page(df, metadata, model_comp_df)
    elif page_selection == "2. Market Analytics":
        p02.render_page(df)
    elif page_selection == "3. Property Predictor":
        p03.render_page(df, metadata)
    elif page_selection == "4. Model Lab":
        p04.render_page(model_comp_df, metadata)
    elif page_selection == "5. Model Explainability":
        p05.render_page(df)
    elif page_selection == "6. Risk & Error Analysis":
        p06.render_page()
    elif page_selection == "7. Insights & Actions":
        p07.render_page(df, metadata)
    elif page_selection == "8. Data Quality":
        p08.render_page()
    elif page_selection == "9. Methodology":
        p09.render_page()
    elif page_selection == "10. Model Monitoring":
        p10.render_page()
    elif page_selection == "11. Documentation":
        p11.render_page(df)
    elif page_selection == "12. About / Internship":
        p12.render_page()

if __name__ == "__main__":
    main()
