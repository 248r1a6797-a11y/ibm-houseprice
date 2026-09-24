"""Page 4 — Interactive Feature Engineering & Feature Selection."""
import streamlit as st
import pandas as pd
import numpy as np
from app.components.header import render_header
from app.state import invalidate_downstream
from src.features.engineering import engineer_custom_algebraic_feature

def render_page():
    render_header(
        title="Interactive Feature Engineering & Selection Studio",
        subtitle="Construct domain-specific derived features, apply formula templates, and curate feature subsets.",
        badge="ENGINEERING LAYER"
    )

    df_clean = st.session_state.clean_df
    target_col = st.session_state.target_col

    if df_clean.empty:
        if not st.session_state.raw_df.empty:
            st.info("ℹ️ Cleaning transformations have not been committed yet. You can auto-initialize directly from raw data or visit **3. Data Cleaning Studio**.")
            if st.button("⚡ Use Raw Data Directly for Feature Engineering", type="primary"):
                st.session_state.clean_df = st.session_state.raw_df.copy()
                st.rerun()
        else:
            st.warning("No dataset available in session. Please visit 1. Data Workspace first.")
        return

    feat_tab, derived_tab = st.tabs(["1. Feature Selection & Automated Ranking", "2. Derived Feature Creator & Templates"])

    # =========================================================================
    # TAB 1: FEATURE SELECTION
    # =========================================================================
    with feat_tab:
        st.markdown("<div class='card-section'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>🎛️ Active Feature Selection Matrix</div>", unsafe_allow_html=True)

        candidate_cols = [c for c in df_clean.columns if c != target_col]
        
        c_ctrl1, c_ctrl2, c_ctrl3 = st.columns([1, 1, 1.2])
        with c_ctrl1:
            if st.button("Select All Features", use_container_width=True):
                st.session_state.selected_features = list(candidate_cols)
                invalidate_downstream(st, "features")
                st.rerun()

        with c_ctrl2:
            if st.button("Deselect All", use_container_width=True):
                st.session_state.selected_features = []
                invalidate_downstream(st, "features")
                st.rerun()

        with c_ctrl3:
            filter_type = st.radio("Display Filter:", ["All", "Numerical Only", "Categorical Only"], horizontal=True)

        search_f = st.text_input("🔍 Search Feature:", value="")

        display_candidates = candidate_cols
        if filter_type == "Numerical Only":
            display_candidates = [c for c in candidate_cols if pd.api.types.is_numeric_dtype(df_clean[c])]
        elif filter_type == "Categorical Only":
            display_candidates = [c for c in candidate_cols if not pd.api.types.is_numeric_dtype(df_clean[c])]

        if search_f:
            display_candidates = [c for c in display_candidates if search_f.lower() in c.lower()]

        # Selection Checkboxes
        cols_per_row = 4
        selected_set = set(st.session_state.selected_features)
        new_selected = list(selected_set)

        st.caption(f"Currently Selected: **{len(selected_set)}** of **{len(candidate_cols)}** candidate features.")

        # Display in rows of 4
        for i in range(0, len(display_candidates), cols_per_row):
            row_cols = display_candidates[i:i+cols_per_row]
            ui_cols = st.columns(cols_per_row)
            for j, c_name in enumerate(row_cols):
                with ui_cols[j]:
                    checked = c_name in selected_set
                    is_chk = st.checkbox(c_name, value=checked, key=f"sel_feat_{c_name}")
                    if is_chk and c_name not in new_selected:
                        new_selected.append(c_name)
                    elif not is_chk and c_name in new_selected:
                        new_selected.remove(c_name)

        if sorted(new_selected) != sorted(st.session_state.selected_features):
            st.session_state.selected_features = new_selected
            invalidate_downstream(st, "features")
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

        # Feature Ranking Table (Correlation with Target)
        if target_col in df_clean.columns and pd.api.types.is_numeric_dtype(df_clean[target_col]):
            st.markdown("<div class='card-section'>", unsafe_allow_html=True)
            st.markdown(f"<div class='card-title'>📊 Numerical Feature Correlation with Target ({target_col})</div>", unsafe_allow_html=True)
            num_candidates = [c for c in candidate_cols if pd.api.types.is_numeric_dtype(df_clean[c])]
            if num_candidates:
                corrs = []
                for col in num_candidates:
                    try:
                        r = df_clean[col].corr(df_clean[target_col])
                        if not np.isnan(r):
                            corrs.append({"Feature": col, "Pearson Correlation": round(r, 4), "Abs Correlation": abs(r)})
                    except Exception:
                        pass
                if corrs:
                    corr_df = pd.DataFrame(corrs).sort_values(by="Abs Correlation", ascending=False).reset_index(drop=True)
                    st.dataframe(corr_df[["Feature", "Pearson Correlation"]], use_container_width=True, height=280)
            st.markdown("</div>", unsafe_allow_html=True)

    # =========================================================================
    # TAB 2: DERIVED FEATURE CREATOR
    # =========================================================================
    with derived_tab:
        st.markdown("<div class='card-section'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>🧪 Custom Algebraic Feature Creator</div>", unsafe_allow_html=True)
        st.caption("Construct a new derived feature by performing arithmetic operations on any two existing numerical columns:")

        num_cols = [c for c in df_clean.columns if pd.api.types.is_numeric_dtype(df_clean[c])]
        
        if len(num_cols) >= 2:
            c1, c2, c3, c4 = st.columns([1.2, 1, 0.6, 1])
            with c1:
                new_f_name = st.text_input("New Feature Name:", value="CustomRatioIndex")
            with c2:
                col_a = st.selectbox("Column A (Operand 1):", num_cols, index=0)
            with c3:
                op = st.selectbox("Operator:", ["+", "-", "*", "/"], index=0)
            with c4:
                col_b = st.selectbox("Column B (Operand 2):", num_cols, index=1 if len(num_cols) > 1 else 0)

            if st.button("➕ Create & Inject Derived Feature", type="primary"):
                if not new_f_name:
                    st.error("Please specify a feature name.")
                else:
                    st.session_state.clean_df = engineer_custom_algebraic_feature(st.session_state.clean_df, new_f_name, col_a, op, col_b)
                    st.session_state.engineered_features[new_f_name] = f"{col_a} {op} {col_b}"
                    if new_f_name not in st.session_state.selected_features:
                        st.session_state.selected_features.append(new_f_name)
                    invalidate_downstream(st, "features")
                    st.success(f"Derived feature `{new_f_name}` created and added to active features!")
                    st.rerun()

        st.markdown("---")
        st.markdown("##### 💡 Schema-Adaptive Feature Templates")
        t_col1, t_col2 = st.columns(2)
        with t_col1:
            if "TotalBsmtSF" in df_clean.columns and "1stFlrSF" in df_clean.columns:
                if st.button("➕ Generate 'TotalSF' (Basement + 1st Floor)", use_container_width=True):
                    st.session_state.clean_df["TotalSF"] = df_clean["TotalBsmtSF"] + df_clean["1stFlrSF"]
                    st.session_state.engineered_features["TotalSF"] = "TotalBsmtSF + 1stFlrSF"
                    if "TotalSF" not in st.session_state.selected_features:
                        st.session_state.selected_features.append("TotalSF")
                    invalidate_downstream(st, "features")
                    st.success("Generated TotalSF!")
                    st.rerun()

            if "FullBath" in df_clean.columns and "HalfBath" in df_clean.columns:
                if st.button("➕ Generate 'TotalBathrooms' (Full + 0.5*Half)", use_container_width=True):
                    st.session_state.clean_df["TotalBathrooms"] = df_clean["FullBath"] + 0.5 * df_clean["HalfBath"]
                    st.session_state.engineered_features["TotalBathrooms"] = "FullBath + 0.5*HalfBath"
                    if "TotalBathrooms" not in st.session_state.selected_features:
                        st.session_state.selected_features.append("TotalBathrooms")
                    invalidate_downstream(st, "features")
                    st.success("Generated TotalBathrooms!")
                    st.rerun()

        with t_col2:
            if "YearBuilt" in df_clean.columns:
                if st.button("➕ Generate 'PropertyAge' (2026 - YearBuilt)", use_container_width=True):
                    st.session_state.clean_df["PropertyAge"] = np.maximum(0, 2026 - df_clean["YearBuilt"])
                    st.session_state.engineered_features["PropertyAge"] = "2026 - YearBuilt"
                    if "PropertyAge" not in st.session_state.selected_features:
                        st.session_state.selected_features.append("PropertyAge")
                    invalidate_downstream(st, "features")
                    st.success("Generated PropertyAge!")
                    st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)
