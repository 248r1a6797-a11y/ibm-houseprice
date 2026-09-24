"""Page 20 — Scientific Methodology & Pipeline Architecture."""
import streamlit as st
from app.components.header import render_header

def render_page():
    render_header(
        title="Scientific Methodology & Pipeline Architecture",
        subtitle="End-to-end data science lifecycle, mathematical metric formulations, and anti-leakage engineering protocols.",
        badge="METHODOLOGY & GOVERNANCE"
    )

    tab1, tab2, tab3 = st.tabs([
        "🔄 14-Stage ML Architecture",
        "📐 Mathematical Metric Formulations",
        "🛡️ Data Leakage & Validation Protocols"
    ])

    # TAB 1: 14-STAGE PIPELINE ARCHITECTURE
    with tab1:
        st.markdown("<div class='card-section'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>End-to-End Enterprise Data Science Lifecycle</div>", unsafe_allow_html=True)
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
        7. Multi-Model Benchmark Suite (12 Regressors across 4 distinct model families)
           │
        8. 5-Fold Cross-Validation (Out-of-fold scoring in true dollar $ scale)
           │
        9. Target Transformation Experiment (Raw target vs log1p(y) with expm1 inverse)
           │
        10. Hyperparameter Optimization (RandomizedSearchCV & GridSearchCV on top gradient boosters)
           │
        11. Model Selection Gate (Rigorous criteria: holdout RMSE, stability, overfit gap)
           │
        12. Model Explainability Layer (TreeExplainer SHAP waterfall & global beeswarm)
           │
        13. Error Diagnostics & Calibrated Intervals (80% empirical residual coverage)
           │
        14. Inference & Decision Support (Single property scoring, what-if simulation, BI insights)
        ```
        """)
        st.markdown("</div>", unsafe_allow_html=True)

    # TAB 2: MATHEMATICAL FORMULATIONS
    with tab2:
        st.markdown("<div class='card-section'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Statistical Metric Definitions</div>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Root Mean Squared Error (RMSE)")
            st.latex(r"\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}")
            st.markdown("Measures the standard deviation of residuals. Heavily penalizes large variance outliers; primary benchmark ranking metric.")

            st.markdown("#### Mean Absolute Error (MAE)")
            st.latex(r"\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|")
            st.markdown("Linear average of absolute errors. Directly interpretable as the typical dollar valuation error.")

            st.markdown("#### Mean Absolute Percentage Error (MAPE)")
            st.latex(r"\text{MAPE} = \frac{100\%}{n} \sum_{i=1}^n \left|\frac{y_i - \hat{y}_i}{y_i}\right|")
            st.markdown("Scale-independent relative error percentage across differing price tiers.")

        with c2:
            st.markdown("#### Coefficient of Determination ($R^2$)")
            st.latex(r"R^2 = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}")
            st.markdown("Proportion of target variance explained by the model relative to the naive mean baseline. Replaces generic 'accuracy' in regression.")

            st.markdown("#### Target Transformation & Inversion")
            st.latex(r"z_i = \ln(1 + y_i), \quad \hat{y}_i = \exp(\hat{z}_i) - 1")
            st.markdown("Eliminates severe positive skewness during gradient updates while evaluating all reported metrics in actual dollars.")

            st.markdown("#### Calibrated 80% Prediction Interval")
            st.latex(r"\hat{y}_i \pm 1.28 \cdot \sigma_{\text{residuals}}")
            st.markdown("Empirical error bounds providing actionable confidence bands for real estate underwriting.")
        st.markdown("</div>", unsafe_allow_html=True)

    # TAB 3: DATA LEAKAGE PROTOCOLS
    with tab3:
        st.markdown("<div class='card-section'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Strict Anti-Leakage & Governance Standards</div>", unsafe_allow_html=True)
        st.markdown(r"""
        1. **Strict Partition Isolation**: The 20% holdout test partition is sequestered immediately after raw ingestion and never touched during preprocessing or cross-validation.
        2. **Pipeline Preprocessing Encapsulation**: All imputers, standard scalers, and one-hot encoders are fitted exclusively on training folds. The holdout test set only receives `.transform()`.
        3. **Target Inversion Integrity**: When models are trained on $\ln(1+y)$, predictions are inverted via $\exp(\hat{z})-1$ before computing RMSE, MAE, and dollar-denominated residuals.
        4. **Zero Look-Ahead Bias**: Temporal features (e.g. `HouseAgeAtSale`) are strictly computed as $\text{YrSold} - \text{YearBuilt}$, preserving point-in-time assessment accuracy.
        """)
        st.markdown("</div>", unsafe_allow_html=True)
