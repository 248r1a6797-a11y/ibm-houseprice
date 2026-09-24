"""
HouseWise AI — Master Streamlit Application & Reactive ML Workbench
===================================================================
A production-grade, 21-module End-to-End Machine Learning Analytics Workbench.
Adheres strictly to the operational principle:
USER ACTION -> COMPUTATION -> RESULT -> VISUALIZATION -> NEXT ACTION
"""

import sys
import importlib
from pathlib import Path
import streamlit as st
import pandas as pd

# 1. Page Configuration (Must be the very first Streamlit call)
st.set_page_config(
    page_title="HouseWise AI — Machine Learning Analytics Workbench",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Path Setup: Ensure project root is prioritized and prevent app/ directory from shadowing package app
ROOT_DIR = Path(__file__).resolve().parent.parent
APP_DIR = Path(__file__).resolve().parent

# Remove app directory from sys.path if Python/Streamlit added it, then prepend ROOT_DIR
sys.path = [p for p in sys.path if Path(p).resolve() != APP_DIR]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from app.state import init_session_state, reset_session_state
    from app.components.pipeline_banner import render_pipeline_banner
except (ModuleNotFoundError, ImportError):
    from state import init_session_state, reset_session_state
    from components.pipeline_banner import render_pipeline_banner


# 3. Theme CSS Injection
def inject_custom_css():
    """Injects corporate theme and component styling."""
    css_path = ROOT_DIR / "app/styles/theme.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# 4. Navigation Registry (21 Modules across 5 Operational Tiers)
WORKBENCH_SECTIONS = {
    "All Modules (1–21)": [
        ("01_data_workspace", "01. Data Ingestion & Target", "📥"),
        ("02_data_profiling", "02. Data Profiling & Quality", "🔍"),
        ("03_data_cleaning_studio", "03. Data Cleaning Studio", "🧹"),
        ("04_feature_engineering_studio", "04. Feature Engineering Studio", "⚙️"),
        ("05_eda_laboratory", "05. EDA Laboratory", "📊"),
        ("06_experiment_setup", "06. Experiment Setup & Split", "🧪"),
        ("07_model_laboratory", "07. Model Benchmark Lab", "🏆"),
        ("08_hyperparameter_tuning", "08. Hyperparameter Tuning", "🎛️"),
        ("09_experiment_history", "09. Experiment History", "📜"),
        ("10_model_explainability", "10. Model Explainability (SHAP)", "💡"),
        ("11_property_predictor", "11. Property Valuation Predictor", "🏡"),
        ("12_what_if_simulator", "12. What-If Scenario Simulator", "🔄"),
        ("13_prediction_diagnostics", "13. Prediction Diagnostics", "🎯"),
        ("14_market_analytics", "14. Market Analytics & Correlations", "📈"),
        ("15_neighborhood_intelligence", "15. Neighborhood Intelligence", "🏘️"),
        ("16_risk_analysis", "16. Sliced Risk & Uncertainty", "🛡️"),
        ("17_insights_actions", "17. Strategic Insights & Actions", "📋"),
        ("18_executive_overview", "18. Executive Intelligence Overview", "🏛️"),
        ("19_sql_studio", "19. Enterprise Relational SQL Studio", "💻"),
        ("20_methodology", "20. Scientific Methodology", "📐"),
        ("21_documentation", "21. Technical Documentation", "📖")
    ],
    "I. Data Workspace (1–5)": [
        ("01_data_workspace", "01. Data Ingestion & Target", "📥"),
        ("02_data_profiling", "02. Data Profiling & Quality", "🔍"),
        ("03_data_cleaning_studio", "03. Data Cleaning Studio", "🧹"),
        ("04_feature_engineering_studio", "04. Feature Engineering Studio", "⚙️"),
        ("05_eda_laboratory", "05. EDA Laboratory", "📊")
    ],
    "II. Machine Learning (6–10)": [
        ("06_experiment_setup", "06. Experiment Setup & Split", "🧪"),
        ("07_model_laboratory", "07. Model Benchmark Lab", "🏆"),
        ("08_hyperparameter_tuning", "08. Hyperparameter Tuning", "🎛️"),
        ("09_experiment_history", "09. Experiment History", "📜"),
        ("10_model_explainability", "10. Model Explainability (SHAP)", "💡")
    ],
    "III. Prediction & Scenarios (11–13)": [
        ("11_property_predictor", "11. Property Valuation Predictor", "🏡"),
        ("12_what_if_simulator", "12. What-If Scenario Simulator", "🔄"),
        ("13_prediction_diagnostics", "13. Prediction Diagnostics", "🎯")
    ],
    "IV. Market Intelligence (14–17)": [
        ("14_market_analytics", "14. Market Analytics & Correlations", "📈"),
        ("15_neighborhood_intelligence", "15. Neighborhood Intelligence", "🏘️"),
        ("16_risk_analysis", "16. Sliced Risk & Uncertainty", "🛡️"),
        ("17_insights_actions", "17. Strategic Insights & Actions", "📋")
    ],
    "V. Governance & Reporting (18–21)": [
        ("18_executive_overview", "18. Executive Intelligence Overview", "🏛️"),
        ("19_sql_studio", "19. Enterprise Relational SQL Studio", "💻"),
        ("20_methodology", "20. Scientific Methodology", "📐"),
        ("21_documentation", "21. Technical Documentation", "📖")
    ]
}

def main():
    # Initialize styling and session state
    inject_custom_css()
    init_session_state(st)

    # Sidebar Header
    header_html = (
        '<div style="padding: 4px 0 16px 0; border-bottom: 1px solid #E2E8F0; margin-bottom: 16px;">'
        '<div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 4px;">'
        '<div style="display: flex; align-items: center; gap: 8px;">'
        '<div style="width: 32px; height: 32px; background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%); border-radius: 8px; display: flex; align-items: center; justify-content: center; color: white; font-size: 1.1rem; box-shadow: 0 2px 6px rgba(37,99,235,0.25);">🏡</div>'
        '<div style="font-size: 1.25rem; font-weight: 800; color: #0F172A; letter-spacing: -0.02em;">HouseWise <span style="color: #2563EB;">AI</span></div>'
        '</div>'
        '<span style="background: #EFF6FF; color: #1D4ED8; font-size: 0.68rem; font-weight: 700; padding: 2px 7px; border-radius: 6px; border: 1px solid #BFDBFE;">PRO</span>'
        '</div>'
        '<div style="font-size: 0.76rem; color: #64748B; font-weight: 500; margin-left: 40px;">Predictive ML Workbench</div>'
        '</div>'
    )
    if hasattr(st.sidebar, "html"):
        st.sidebar.html(header_html)
    else:
        st.sidebar.markdown(header_html, unsafe_allow_html=True)

    # Tier Filter Dropdown
    st.sidebar.markdown("<p style='font-size: 0.72rem; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;'>Operational Stage</p>", unsafe_allow_html=True)
    selected_tier = st.sidebar.selectbox(
        "Operational Tier:",
        list(WORKBENCH_SECTIONS.keys()),
        index=0,
        label_visibility="collapsed"
    )

    tier_pages = WORKBENCH_SECTIONS[selected_tier]
    page_labels = [f"{icon} {label}" for _, label, icon in tier_pages]
    page_keys = [key for key, _, _ in tier_pages]

    # Remember active selection across tier toggles
    current_key = st.session_state.get("active_page_key", "01_data_workspace")
    default_index = 0
    if current_key in page_keys:
        default_index = page_keys.index(current_key)

    st.sidebar.markdown("<p style='font-size: 0.72rem; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 10px; margin-bottom: 4px;'>Navigation</p>", unsafe_allow_html=True)
    selected_label = st.sidebar.radio(
        "Navigation",
        page_labels,
        index=default_index,
        label_visibility="collapsed"
    )

    # Determine selected module
    selected_idx = page_labels.index(selected_label)
    selected_module = page_keys[selected_idx]
    st.session_state["active_page_key"] = selected_module

    # Active Session Summary Card in Sidebar
    raw_df = st.session_state.get("raw_df", pd.DataFrame())
    active_champ = st.session_state.get("champion_name", "None")
    target_name = st.session_state.get("target_col", "None")
    needs_retrain = st.session_state.get("needs_retraining", False)
    dataset_name = st.session_state.get("dataset_name", "None")
    display_dataset = (dataset_name[:20] + "...") if len(dataset_name) > 23 else dataset_name

    status_dot = "#EF4444" if needs_retrain else "#10B981"
    status_text = "Retrain Required" if needs_retrain else "Synced & Ready"

    status_card_html = (
        '<div style="background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%); border: 1px solid #334155; border-radius: 10px; padding: 12px 14px; color: #E2E8F0; box-shadow: 0 4px 10px rgba(0,0,0,0.08); margin-top: 16px;">'
        '<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; border-bottom: 1px solid #334155; padding-bottom: 6px;">'
        '<span style="font-size: 0.68rem; font-weight: 700; color: #94A3B8; letter-spacing: 0.06em; text-transform: uppercase;">Workspace State</span>'
        f'<span style="display: inline-flex; align-items: center; gap: 5px; font-size: 0.7rem; font-weight: 600; color: {status_dot};">'
        f'<span style="width: 6px; height: 6px; border-radius: 50%; background: {status_dot}; display: inline-block;"></span>'
        f'{status_text}'
        '</span>'
        '</div>'
        '<div style="font-size: 0.74rem; line-height: 1.6; color: #CBD5E1;">'
        f'<div style="display: flex; justify-content: space-between; margin-bottom: 2px;"><span style="color: #94A3B8;">Dataset:</span> <span style="font-weight: 600; color: #F8FAFC;" title="{dataset_name}">{display_dataset}</span></div>'
        f'<div style="display: flex; justify-content: space-between; margin-bottom: 2px;"><span style="color: #94A3B8;">Records:</span> <span style="font-weight: 600; color: #F8FAFC;">{len(raw_df):,} × {raw_df.shape[1] if not raw_df.empty else 0}</span></div>'
        f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2px;"><span style="color: #94A3B8;">Target:</span> <span style="font-family: monospace; background: #334155; padding: 1px 6px; border-radius: 4px; color: #93C5FD; font-size: 0.7rem;">{target_name}</span></div>'
        f'<div style="display: flex; justify-content: space-between;"><span style="color: #94A3B8;">Champion:</span> <span style="font-weight: 700; color: #34D399;">{active_champ}</span></div>'
        '</div>'
        '</div>'
    )
    if hasattr(st.sidebar, "html"):
        st.sidebar.html(status_card_html)
    else:
        st.sidebar.markdown(status_card_html, unsafe_allow_html=True)

    st.sidebar.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    if st.sidebar.button("🔄 Reset Entire Workbench", use_container_width=True):
        reset_session_state(st)
        st.sidebar.success("Workbench session state reset successfully.")
        st.rerun()

    # Render Main Lifecycle Progress Banner
    render_pipeline_banner()

    # Dynamically Import and Render Selected Page
    try:
        page_mod = importlib.import_module(f"app.pages.{selected_module}")
        if hasattr(page_mod, "render_page"):
            page_mod.render_page()
        else:
            st.error(f"Module `app.pages.{selected_module}` does not implement `render_page()`.")
    except Exception as e:
        st.error(f"Error loading module `{selected_module}`: {e}")
        with st.expander("🛠️ View Error Traceback", expanded=True):
            import traceback
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
