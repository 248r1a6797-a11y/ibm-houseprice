"""Page 12 — About / Internship Project Credentials."""
import streamlit as st
from app.components.header import render_header

def render_page():
    render_header(
        title="About HouseWise AI & Project Credentials",
        subtitle="Portfolio credentials, technical competencies demonstrated, and Responsible AI governance framework.",
        badge="INTERNSHIP CAPSTONE"
    )

    st.markdown("<div class='card-section'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>Project Profile</div>", unsafe_allow_html=True)
    st.markdown("""
    - **Project Title**: HouseWise AI — Explainable House Price Intelligence & Prediction Platform
    - **Alternative Title**: House Price Intelligence: From Raw Housing Data to Predictive Business Decisions
    - **Context**: Developed as an intensive internship data science & analytics portfolio capstone.
    - **Core Mission**: Bridge predictive machine learning regression with executive decision support and transparent explainability.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("<div class='card-section'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Technical Competencies Demonstrated</div>", unsafe_allow_html=True)
        st.markdown("""
        1. **Exploratory Data Analysis & Statistics**: Univariate, bivariate, correlation matrices, skewness analysis.
        2. **Data Cleaning & Governance**: Domain structural null handling, clerical entry forensics (`GarageYrBlt` 2207 $\\to$ 2007).
        3. **Anti-Leakage Feature Engineering**: Domain combinations (`TotalSF`, `TotalBath`), temporal age calculation (`YrSold - YearBuilt`).
        4. **Machine Learning & Benchmarking**: 18 regression models evaluated across 6 families (OLS, Ridge, Lasso, Trees, Ensembles, Boosters, MLP).
        5. **Scientific Cross-Validation**: 5-Fold K-Fold cross validation with out-of-fold scoring in true dollar ($\$) scale.
        6. **Model Explainability (XAI)**: Global and local SHAP attributions, feature waterfall attributions.
        7. **Software Engineering & MLOps**: Modular code structure (`src/`, `config/`), configuration-driven, unit tests (`pytest`), Streamlit UI.
        """)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='card-section'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Responsible AI & Ethical Governance</div>", unsafe_allow_html=True)
        st.markdown("""
        - **Intended Purpose**: Decision support, educational benchmarking, and preliminary appraisal screening.
        - **Prohibited Uses**: This platform must **NOT** be utilized for automated mortgage credit approval, credit underwriting, insurance denial, or tenant background screening without certified legal compliance.
        - **Fairness & Demographic Neutrality**: The dataset strictly incorporates physical property characteristics and zoning definitions; no protected demographic attributes are used.
        - **Uncertainty Transparency**: Every valuation includes an 80% empirical calibrated interval to convey prediction variance honestly.
        """)
        st.markdown("</div>", unsafe_allow_html=True)
