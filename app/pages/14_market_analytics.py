"""Page 14 — Market Analytics & Feature Relationship Intelligence."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from app.components.header import render_header

def render_page():
    render_header(
        title="Market Analytics & Feature Relationships",
        subtitle="Univariate correlations, non-linear dependencies, and price-per-square-foot dynamics across property cohorts.",
        badge="MARKET INTELLIGENCE"
    )

    df = st.session_state.clean_df if not st.session_state.clean_df.empty else st.session_state.raw_df
    target_col = st.session_state.target_col

    if df.empty:
        st.warning("No data available in session. Please visit 1. Data Workspace.")
        return

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # 1. Correlation Matrix Studio
    st.markdown("<div class='card-section'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>📐 Multidimensional Feature Correlation Matrix</div>", unsafe_allow_html=True)

    c_m1, c_m2 = st.columns([1, 2])
    with c_m1:
        corr_method = st.radio("Correlation Metric:", ["Pearson (Linear)", "Spearman (Rank / Monotonic)"], horizontal=True)
        method_str = "pearson" if "Pearson" in corr_method else "spearman"

    with c_m2:
        default_corr_cols = [c for c in [target_col, "GrLivArea", "OverallQual", "TotalBsmtSF", "1stFlrSF", "GarageCars", "YearBuilt", "FullBath"] if c in num_cols]
        if len(default_corr_cols) < 2:
            default_corr_cols = num_cols[:min(8, len(num_cols))]
        selected_corr_cols = st.multiselect("Select Numerical Columns to Correlate:", num_cols, default=default_corr_cols)

    if len(selected_corr_cols) >= 2:
        corr_mat = df[selected_corr_cols].corr(method=method_str).round(2)
        fig_heat = px.imshow(
            corr_mat,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Blues",
            labels=dict(color=f"{method_str.title()} Correlation")
        )
        fig_heat.update_layout(height=450, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_heat, use_container_width=True)

    st.info("⚠️ Methodological Notice: Statistical correlation reflects empirical linear or monotonic association within the dataset period and must not be conflated with causal impact.")
    st.markdown("</div>", unsafe_allow_html=True)

    # 2. Top Relationships with Target
    if target_col in num_cols:
        st.markdown("<div class='card-section'>", unsafe_allow_html=True)
        st.markdown(f"<div class='card-title'>🏆 Top Feature Associations with Target ({target_col})</div>", unsafe_allow_html=True)

        other_nums = [c for c in num_cols if c != target_col]
        all_r = []
        for col in other_nums:
            r = df[col].corr(df[target_col], method=method_str)
            all_r.append({"Feature": col, "Correlation": r, "Abs_Corr": abs(r)})

        r_df = pd.DataFrame(all_r).dropna().sort_values(by="Correlation", ascending=False)

        col_pos, col_neg = st.columns(2)
        with col_pos:
            st.markdown("##### 🟢 Top Positive Predictive Relationships")
            st.dataframe(r_df.head(6)[["Feature", "Correlation"]].style.format({"Correlation": "{:+.3f}"}), use_container_width=True)

        with col_neg:
            st.markdown("##### 🔴 Top Negative Predictive Relationships")
            st.dataframe(r_df.tail(6).sort_values(by="Correlation", ascending=True)[["Feature", "Correlation"]].style.format({"Correlation": "{:+.3f}"}), use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)
