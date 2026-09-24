"""Page 3 — Interactive Data Cleaning Studio."""
import streamlit as st
import pandas as pd
import numpy as np
from app.components.header import render_header
from app.state import invalidate_downstream
from src.data.cleaner import clean_dataset_adaptive

def render_page():
    render_header(
        title="Interactive Data Cleaning Studio",
        subtitle="Configure missing value imputations, deduplication, and outlier treatments with dynamic before/after verification.",
        badge="CLEANING LAYER"
    )

    df_raw = st.session_state.raw_df
    if df_raw.empty:
        st.warning("No dataset loaded in session. Please navigate to 1. Data Workspace.")
        return

    st.markdown("<div class='card-section'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>🛠️ Data Cleaning Configuration Controls</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("##### 1. Missing Value Strategy")
        num_strat = st.radio("Numerical Imputation:", ["Median", "Mean", "Zero"], index=0)
        cat_strat = st.radio("Categorical Imputation:", ["Most Frequent", 'Constant "Missing"'], index=0)

    with col2:
        st.markdown("##### 2. Duplicate Row Handling")
        dup_count = int(df_raw.duplicated().sum())
        st.markdown(f"Detected exact duplicate rows: **{dup_count}**")
        remove_dup = st.checkbox("Remove Duplicate Rows", value=True)

    with col3:
        st.markdown("##### 3. Outlier Handling")
        outlier_method = st.selectbox("Outlier Detection Method:", ["IQR (1.5 × Spread)", "Z-Score (> 3.0 Std)"])
        outlier_treatment = st.radio("Treatment Action:", ["Keep (No Treatment)", "Cap / Winsorize", "Remove Records"], index=0)

    st.markdown("---")
    
    # Cleaning Actions
    opts = {
        "num_impute_strategy": num_strat,
        "cat_impute_strategy": cat_strat,
        "remove_duplicates": remove_dup,
        "outlier_method": outlier_method,
        "outlier_treatment": outlier_treatment
    }

    if st.button("🚀 Apply Cleaning Transformations", type="primary"):
        cleaned, audit = clean_dataset_adaptive(df_raw, options=opts, target_col=st.session_state.target_col)
        
        # Store in session state
        st.session_state.clean_df = cleaned
        st.session_state.cleaning_params = opts
        st.session_state.cleaning_audit = audit
        st.session_state.pipeline_status["CLEANING"] = "✓ Completed"
        invalidate_downstream(st, "cleaning")
        st.success(f"Cleaning transformations applied successfully! Cleaned Dataset: {len(cleaned):,} rows × {len(cleaned.columns)} columns.")

    st.markdown("</div>", unsafe_allow_html=True)

    # Before vs After Comparison Card
    audit = st.session_state.get("cleaning_audit", {})
    initial_rows = audit.get("initial_rows", len(df_raw))
    initial_nulls = int(df_raw.isnull().sum().sum())
    is_cleaned = not st.session_state.clean_df.empty

    clean_rows = len(st.session_state.clean_df) if is_cleaned else initial_rows
    final_nulls = int(st.session_state.clean_df.isnull().sum().sum()) if is_cleaned else initial_nulls

    st.markdown("#### 📊 Transformation Verification Audit")
    b_col1, b_col2, b_col3, b_col4 = st.columns(4)
    with b_col1:
        st.metric(
            "Missing Values",
            f"{final_nulls:,}",
            delta=f"-{initial_nulls - final_nulls:,} imputed" if is_cleaned and final_nulls < initial_nulls else ("Awaiting Run" if not is_cleaned else "0 remaining"),
            delta_color="normal"
        )
    with b_col2:
        st.metric(
            "Total Records",
            f"{clean_rows:,}",
            delta=f"{clean_rows - initial_rows:,} rows" if is_cleaned and clean_rows != initial_rows else ("Awaiting Run" if not is_cleaned else "Preserved"),
            delta_color="normal"
        )
    with b_col3:
        st.metric(
            "Duplicates Removed",
            f"{audit.get('duplicates_removed', 0):,}" if is_cleaned else "0",
            delta="Clean" if is_cleaned else "Pending"
        )
    with b_col4:
        st.metric(
            "Outliers Treated",
            f"{audit.get('outliers_treated', 0):,}" if is_cleaned else "0",
            delta="Processed" if is_cleaned else "Pending"
        )

    # Preview Cleaned Data
    if not st.session_state.clean_df.empty:
        st.markdown("<div class='card-section'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>🔍 Cleaned Data Sample (First 10 Rows)</div>", unsafe_allow_html=True)
        st.dataframe(st.session_state.clean_df.head(10), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
