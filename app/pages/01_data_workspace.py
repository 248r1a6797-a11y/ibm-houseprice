"""Page 1 — Data Workspace (Dataset Ingestion & Target Definition)."""
import io
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from app.components.header import render_header
from app.state import invalidate_downstream, set_active_dataset
from src.data.loader import load_raw_data

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

def render_page():
    render_header(
        title="Data Workspace & Dataset Management",
        subtitle="Ingest custom property datasets, inspect schema foundations, define target variables, and control pipeline state.",
        badge="WORKSPACE LAYER"
    )

    col_up, col_actions = st.columns([1.2, 0.8])

    with col_up:
        st.markdown("<div class='card-section'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>📁 Ingest Dataset (CSV or Excel)</div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload Property File:",
            type=["csv", "xlsx", "xls"],
            help="Upload any residential property dataset in CSV or Excel format (arbitrary columns supported)."
        )

        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".csv"):
                    # Detect delimiter and encoding
                    new_df = pd.read_csv(uploaded_file)
                else:
                    new_df = pd.read_excel(uploaded_file)

                # Strip whitespace from column headers
                new_df.columns = [str(c).strip() for c in new_df.columns]

                if st.session_state.dataset_name != uploaded_file.name or len(new_df) != len(st.session_state.raw_df):
                    set_active_dataset(st, new_df, dataset_name=uploaded_file.name)
                    st.success(f"Successfully loaded custom dataset: `{uploaded_file.name}` ({len(new_df):,} rows × {len(new_df.columns)} columns)")
                    st.rerun()
            except Exception as e:
                st.error(f"Error parsing uploaded file: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

    with col_actions:
        st.markdown("<div class='card-section'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>⚙️ Workspace Actions</div>", unsafe_allow_html=True)
        
        c_act1, c_act2 = st.columns(2)
        with c_act1:
            if st.button("🔄 Reset to Default Ames", use_container_width=True):
                raw_path = ROOT_DIR / "data/raw/AmesHousing.csv"
                df_ames = load_raw_data(raw_path, normalize=True)
                set_active_dataset(st, df_ames, dataset_name="AmesHousing.csv (Default Benchmark)", target_col="SalePrice")
                st.rerun()

        with c_act2:
            if not st.session_state.raw_df.empty:
                csv_bytes = st.session_state.raw_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Export Dataset (CSV)",
                    data=csv_bytes,
                    file_name=f"exported_{st.session_state.dataset_name.split()[0]}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        st.markdown("---")
        st.caption(f"**Active Source**: `{st.session_state.dataset_name}`")
        st.markdown("</div>", unsafe_allow_html=True)

    df = st.session_state.raw_df
    if df.empty:
        st.warning("No dataset loaded in session. Please upload a file above or click 'Reset to Default Ames'.")
        return

    # Target & Problem Type Configuration Card
    st.markdown("<div class='card-section'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>🎯 Target Variable & Machine Learning Task Configuration</div>", unsafe_allow_html=True)
    
    t_col1, t_col2 = st.columns([1.2, 1])
    with t_col1:
        all_cols = list(df.columns)
        num_cols = [c for c in all_cols if pd.api.types.is_numeric_dtype(df[c])]
        default_target = st.session_state.target_col
        if default_target in all_cols:
            default_idx = all_cols.index(default_target)
        elif num_cols:
            default_idx = all_cols.index(num_cols[-1])
        else:
            default_idx = len(all_cols) - 1

        selected_target = st.selectbox(
            "Select Target Variable to Predict:",
            all_cols,
            index=default_idx,
            help="Choose the continuous numeric variable (e.g. Price, SalePrice, MEDV) to predict."
        )
        if selected_target != st.session_state.target_col:
            set_active_dataset(st, df, dataset_name=st.session_state.dataset_name, target_col=selected_target)
            st.rerun()

    with t_col2:
        is_num = pd.api.types.is_numeric_dtype(df[st.session_state.target_col])
        n_unique = df[st.session_state.target_col].nunique()
        auto_type = "Regression" if is_num and n_unique > 15 else "Classification"
        
        prob_type = st.radio(
            "Prediction Task Type:",
            ["Regression", "Classification"],
            index=0 if auto_type == "Regression" else 1,
            horizontal=True
        )
        st.session_state.problem_type = prob_type

        target_card_html = (
            '<div style="background: #F8FAFC; border: 1px solid #E2E8F0; padding: 10px 14px; border-radius: 6px; font-size: 0.82rem; color: #475569;">'
            f'<b>Target Overview</b>: <code>{st.session_state.target_col}</code><br/>'
            f'<b>Data Type</b>: <code>{df[st.session_state.target_col].dtype}</code><br/>'
            f'<b>Unique Values</b>: {n_unique:,}'
            '</div>'
        )
        if hasattr(st, "html"):
            st.html(target_card_html)
        else:
            st.markdown(target_card_html, unsafe_allow_html=True)

        if not is_num:
            st.warning(f"⚠️ `{st.session_state.target_col}` is non-numeric. For price prediction and regression modeling, please select a numeric column.")

    st.markdown("</div>", unsafe_allow_html=True)

    # Dataset Summary Metrics Row
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]
    total_nulls = int(df.isnull().sum().sum())
    dup_rows = int(df.duplicated().sum())

    st.markdown("#### Active Dataset Overview")
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Total Records", f"{len(df):,}")
    k2.metric("Total Features", f"{len(df.columns)}")
    k3.metric("Numeric Columns", f"{len(num_cols)}")
    k4.metric("Categorical Columns", f"{len(cat_cols)}")
    k5.metric("Missing Values", f"{total_nulls:,}")
    k6.metric("Duplicate Rows", f"{dup_rows:,}")

    # Data Preview
    st.markdown("<div class='card-section'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>🔍 Raw Dataset Sample (First 15 Rows)</div>", unsafe_allow_html=True)
    st.dataframe(df.head(15), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
