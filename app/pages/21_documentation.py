"""Page 21 — Technical Documentation & Interactive Data Dictionary."""
import streamlit as st
import pandas as pd
from pathlib import Path
from app.components.header import render_header

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

def render_page(df: pd.DataFrame = None):
    render_header(
        title="Technical Documentation & Data Dictionary",
        subtitle="Searchable documentation for dataset variables, provenance records, and regulatory notices.",
        badge="TECHNICAL DOCUMENTATION"
    )

    tab1, tab2, tab3 = st.tabs([
        "📖 Interactive Data Dictionary",
        "🏛️ Data Provenance & Academic Citations",
        "⚙️ System Architecture & Technology Stack"
    ])

    with tab1:
        st.markdown("<div class='card-section'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Authoritative Variable Codebook</div>", unsafe_allow_html=True)
        
        search_query = st.text_input("Filter variables by keyword (e.g. 'area', 'qual', 'garage', 'bath'):", "")
        
        doc_md_path = ROOT_DIR / "docs/data_dictionary.md"
        if doc_md_path.exists():
            with open(doc_md_path, "r", encoding="utf-8") as f:
                content = f.read()
            if search_query:
                # Filter lines matching search
                lines = content.split("\n")
                filtered = [l for l in lines if search_query.lower() in l.lower() or l.startswith("#")]
                st.markdown("\n".join(filtered))
            else:
                st.markdown(content)
        else:
            st.info("Data dictionary file (docs/data_dictionary.md) not found.")
        st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        st.markdown("<div class='card-section'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Data Provenance & Citations</div>", unsafe_allow_html=True)
        st.markdown("""
        - **Primary Academic Citation**: De Cock, Dean. *"Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project."* Journal of Statistics Education, Volume 19, Number 3 (2011).
        - **Archive Source**: American Statistical Association (ASA) JSE Data Archive.
        - **Data Custodian**: Ames City Assessor's Office, Story County, Iowa.
        - **Geographic Scope**: Ames, Iowa, USA (incorporating 28 distinct neighborhoods).
        - **Temporal Observation Window**: January 2006 – July 2010.
        - **Authoritative Text Documentation**: Local copy stored at `data/raw/DataDocumentation.txt`.
        """)
        st.markdown("</div>", unsafe_allow_html=True)

    with tab3:
        st.markdown("<div class='card-section'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>System Architecture & Technology Stack</div>", unsafe_allow_html=True)
        st.markdown("""
        | Layer | Technology | Purpose |
        |---|---|---|
        | **Frontend UI** | Streamlit 1.40+ | Reactive single-page web application with multi-tier sidebar navigation |
        | **State Engine** | `st.session_state` + Invalidation Engine | Reactive data science pipeline with automatic downstream invalidation |
        | **Analytics & SQL** | SQLite 3 + Python `sqlite3` | 12 production analytical scripts with sub-millisecond query performance |
        | **Visualizations** | Plotly Express & Plotly Graph Objects | Interactive charts with tooltips, box plots, histograms, and heatmaps |
        | **Machine Learning** | Scikit-Learn 1.6+, XGBoost, LightGBM, CatBoost | 12 candidate regressors with 5-fold cross-validation and hyperparameter tuning |
        | **Model Explainability** | SHAP (TreeExplainer) | Global feature attribution beeswarms and single-property dollar waterfalls |
        | **Testing & CI** | Pytest | Automated test coverage for state management, cleaning, features, and modeling |
        """)
        st.markdown("</div>", unsafe_allow_html=True)
