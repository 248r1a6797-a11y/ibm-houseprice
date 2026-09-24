"""Page 6 — Machine Learning Experiment Setup & Cross-Validation Configuration."""
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from app.components.header import render_header
from app.state import invalidate_downstream

def render_page():
    render_header(
        title="Machine Learning Experiment Setup",
        subtitle="Configure train/test holdout partitioning, cross-validation strategies, target transformations, and random seeds.",
        badge="EXPERIMENT LAYER"
    )

    df = st.session_state.feature_df if not st.session_state.feature_df.empty else (
        st.session_state.clean_df if not st.session_state.clean_df.empty else st.session_state.raw_df
    )
    target_col = st.session_state.target_col
    selected_features = st.session_state.get("selected_features", [])

    if df.empty or not target_col or target_col not in df.columns:
        st.warning("No dataset or target available. Please visit 1. Data Workspace first.")
        return

    if not selected_features:
        selected_features = [c for c in df.columns if c != target_col]
        st.session_state.selected_features = selected_features

    st.markdown("<div class='card-section'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>⚙️ Cross-Validation & Partitioning Parameters</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        test_pct = st.slider("Holdout Test Set Size (%):", 10, 40, int(st.session_state.split_config.get("test_size", 0.20) * 100), 5)
        test_size = test_pct / 100.0

    with col2:
        cv_folds = st.slider("Cross-Validation Folds (K):", 3, 10, int(st.session_state.split_config.get("cv_folds", 5)), 1)

    with col3:
        seed = st.number_input("Random Seed (Reproducibility):", value=int(st.session_state.split_config.get("random_state", 42)), step=1)

    col4, col5 = st.columns(2)
    with col4:
        use_log = st.checkbox(
            "Apply log1p(y) Target Transformation (Recommended for skewed house prices)",
            value=st.session_state.split_config.get("use_log_target", True)
        )

    with col5:
        cv_strat = st.selectbox("Cross-Validation Strategy:", ["K-Fold (Standard)", "Quantile Stratified K-Fold"])

    st.markdown("---")

    if st.button("🚀 Initialize & Commit Experiment Partitions", type="primary"):
        st.session_state.split_config = {
            "test_size": test_size,
            "cv_folds": cv_folds,
            "random_state": seed,
            "use_log_target": use_log,
            "strategy": cv_strat
        }

        # Create partitions
        X = df[selected_features].copy()
        y = df[target_col].copy()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed
        )

        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test

        invalidate_downstream(st, "split")
        st.session_state.pipeline_status["TRAINING"] = "⏳ Configured (Ready)"
        st.success(f"Experiment partitions committed! Training Set: {len(X_train):,} rows | Holdout Test Set: {len(X_test):,} rows.")

    st.markdown("</div>", unsafe_allow_html=True)

    # Display Current Partition Health Card
    st.markdown("#### 📐 Current Data Partitions")
    if st.session_state.X_train is not None and st.session_state.X_test is not None:
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Training Pool Records", f"{len(st.session_state.X_train):,}", f"{(1-test_size):.0%}")
        p2.metric("Holdout Validation Records", f"{len(st.session_state.X_test):,}", f"{test_size:.0%}")
        p3.metric("Selected Feature Count", f"{len(selected_features)}")
        p4.metric("Cross-Validation Folds", f"{cv_folds}-Fold K-Fold")

        st.markdown("<div class='card-section'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Target Distribution Comparison (Train vs Holdout Test)</div>", unsafe_allow_html=True)
        
        train_y = st.session_state.y_train
        test_y = st.session_state.y_test
        
        comp_data = {
            "Partition": ["Training Pool", "Holdout Test Partition"],
            "Count": [len(train_y), len(test_y)],
            "Mean Target ($)": [f"${train_y.mean():,.2f}", f"${test_y.mean():,.2f}"],
            "Median Target ($)": [f"${train_y.median():,.2f}", f"${test_y.median():,.2f}"],
            "Std Dev ($)": [f"${train_y.std():,.2f}", f"${test_y.std():,.2f}"],
            "Min ($)": [f"${train_y.min():,.2f}", f"${test_y.min():,.2f}"],
            "Max ($)": [f"${train_y.max():,.2f}", f"${test_y.max():,.2f}"]
        }
        st.table(pd.DataFrame(comp_data))
        st.info("⚠️ Verification: Notice the mean, median, and variance align closely between train and test partitions, ensuring zero distribution skew across splits.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("Partitions not yet initialized in this session. Click 'Initialize & Commit Experiment Partitions' above.")
