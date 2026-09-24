"""Page 9 — Methodology & Pipeline Architecture."""
import streamlit as st
from app.components.header import render_header

def render_page():
    render_header(
        title="Scientific Methodology & Pipeline Architecture",
        subtitle="End-to-end data science lifecycle, mathematical metric formulations, and anti-leakage engineering protocols.",
        badge="METHODOLOGY"
    )

    st.markdown("<div class='card-section'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>The 14-Stage Machine Learning Pipeline</div>", unsafe_allow_html=True)

    st.markdown("""
    ```
    1. Raw Data Ingestion (AmesHousing.csv, 2,930 rows, 82 columns)
       │
    2. Schema & Bounds Validation (dtypes, positive price checks, year bounds)
       │
    3. Structural Missingness Imputation (NA -> 'None' for amenities, 0.0 for counts)
       │
    4. Anti-Leakage Feature Engineering (TotalSF, TotalBath, HouseAgeAtSale = YrSold - YearBuilt)
       │
    5. Clean Holdout Splitting (80% Train/CV Pool, 20% Untouched Holdout Test)
       │
    6. Reusable Preprocessor Assembly (ColumnTransformer: Median Imputer + StandardScaler + OHE)
       │
    7. Multi-Model Benchmark Suite (18 Regressors across 6 distinct model families)
       │
    8. 5-Fold Cross-Validation (Out-of-fold scoring in true dollar $ scale)
       │
    9. Target Transformation Experiment (Raw target vs log1p(y) with expm1 inverse)
       │
    10. Hyperparameter Optimization (RandomizedSearchCV on top gradient boosters)
       │
    11. Model Selection Gate (Rigorous criteria: holdout RMSE, stability, overfit gap)
       │
    12. Model Explainability Layer (TreeExplainer SHAP waterfall & global beeswarm)
       │
    13. Error Diagnostics & Calibrated Intervals (80% empirical residual coverage)
       │
    14. Inference & Decision Support (Single property scoring, batch CSV processing, BI insights)
    ```
    """)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card-section'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>Mathematical Metric Formulations</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Root Mean Squared Error (RMSE)")
        st.latex(r"\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}")
        st.markdown("Measures standard deviation of residuals. Penalizes large errors heavily; primary metric for valuation stability.")

        st.markdown("#### Mean Absolute Error (MAE)")
        st.latex(r"\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|")
        st.markdown("Linear average of absolute errors. Easily interpretable as the typical dollar valuation error.")

    with c2:
        st.markdown("#### Coefficient of Determination ($R^2$)")
        st.latex(r"R^2 = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}")
        st.markdown("Proportion of target variance explained by model relative to the naive mean baseline.")

        st.markdown("#### Target Transformation & Inversion")
        st.latex(r"z_i = \ln(1 + y_i), \quad \hat{y}_i = \exp(\hat{z}_i) - 1")
        st.markdown("Eliminates severe positive skewness during gradient updates while evaluating all reported metrics in actual dollars.")
    st.markdown("</div>", unsafe_allow_html=True)
