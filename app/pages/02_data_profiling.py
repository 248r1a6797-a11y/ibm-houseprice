"""Page 2 — Interactive Data Profiling & Quality Scoring."""
import streamlit as st
import pandas as pd
import numpy as np
from app.components.header import render_header

def render_page():
    render_header(
        title="Interactive Data Profiling & Quality Audit",
        subtitle="Exhaustive univariate schema health, missingness distribution, cardinality, and automated quality scoring.",
        badge="AUDIT LAYER"
    )

    df = st.session_state.raw_df
    target_col = st.session_state.target_col

    if df.empty:
        st.warning("No dataset loaded in session. Please navigate to 1. Data Workspace.")
        return

    # Calculate Data Quality Score
    total_cells = df.shape[0] * df.shape[1]
    total_nulls = int(df.isnull().sum().sum())
    completeness_score = max(0.0, 100.0 * (1.0 - total_nulls / total_cells)) if total_cells > 0 else 100.0
    
    dup_rows = int(df.duplicated().sum())
    uniqueness_score = max(0.0, 100.0 * (1.0 - dup_rows / len(df))) if len(df) > 0 else 100.0

    constant_cols = [c for c in df.columns if df[c].nunique() <= 1]
    validity_penalty = len(constant_cols) * 2.0
    overall_quality_score = int(np.clip(0.6 * completeness_score + 0.4 * uniqueness_score - validity_penalty, 0, 100))

    # Top Quality Score Card
    score_html = (
        '<div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-left: 6px solid #1E3A8A; border-radius: 8px; padding: 18px 24px; margin-bottom: 20px;">'
        '<div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">'
        '<div>'
        '<div style="font-size: 0.8rem; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 0.05em;">COMPUTED DATA QUALITY SCORE</div>'
        f'<div style="font-size: 2.4rem; font-weight: 800; color: #0F172A; margin: 4px 0;">{overall_quality_score} <span style="font-size: 1.1rem; color: #64748B; font-weight: 500;">/ 100</span></div>'
        f'<div style="font-size: 0.84rem; color: #475569;">Completeness: <b>{completeness_score:.1f}%</b> | Uniqueness: <b>{uniqueness_score:.1f}%</b> | Constant Columns: <b>{len(constant_cols)}</b></div>'
        '</div>'
        '<div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 12px 18px; text-align: right;">'
        '<div style="font-size: 0.78rem; color: #64748B;">Total Records</div>'
        f'<div style="font-size: 1.25rem; font-weight: 700; color: #1E3A8A;">{len(df):,}</div>'
        f'<div style="font-size: 0.78rem; color: #64748B; margin-top: 4px;">Features Inspected: <b>{len(df.columns)}</b></div>'
        '</div>'
        '</div>'
        '</div>'
    )
    if hasattr(st, "html"):
        st.html(score_html)
    else:
        st.markdown(score_html, unsafe_allow_html=True)

    # Build Column Profiling Table
    profile_rows = []
    for col in df.columns:
        series = df[col]
        dtype = str(series.dtype)
        is_num = pd.api.types.is_numeric_dtype(series)
        null_count = int(series.isnull().sum())
        null_pct = (null_count / len(df)) * 100.0
        unique_cnt = series.nunique(dropna=True)

        is_id = False
        if unique_cnt == len(df) or "id" in col.lower() or "pid" in col.lower():
            is_id = True

        is_leakage = False
        if col != target_col and is_num and target_col in df.columns and pd.api.types.is_numeric_dtype(df[target_col]):
            try:
                corr = series.corr(df[target_col])
                if abs(corr) > 0.90:
                    is_leakage = True
            except Exception:
                pass

        tag = "Numeric" if is_num else "Categorical"
        if col == target_col:
            tag = "🎯 Target"
        elif is_leakage:
            tag = "⚠️ Leakage Risk"
        elif is_id:
            tag = "🔑 Identifier"

        row = {
            "Feature": col,
            "Type": tag,
            "Missing": null_count,
            "Missing %": round(null_pct, 2),
            "Unique": unique_cnt,
            "Mean": round(float(series.mean()), 2) if is_num else None,
            "Median": round(float(series.median()), 2) if is_num else None,
            "Std Dev": round(float(series.std()), 2) if is_num else None,
            "Min": round(float(series.min()), 2) if is_num else None,
            "Max": round(float(series.max()), 2) if is_num else None
        }
        profile_rows.append(row)

    profile_df = pd.DataFrame(profile_rows)

    # Interactive Filter Controls
    st.markdown("<div class='card-section'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>📋 Column-Level Profiling Directory</div>", unsafe_allow_html=True)
    
    f_col1, f_col2 = st.columns([1.2, 0.8])
    with f_col1:
        filter_mode = st.radio(
            "Filter Columns by Category:",
            ["All Columns", "Numerical Only", "Categorical Only", "Missing Values Only (>0)", "High Cardinality (>25)", "Potential ID / Leakage"],
            horizontal=True
        )

    with f_col2:
        search_query = st.text_input("🔍 Search Feature by Name:", value="")

    filtered_profile = profile_df.copy()
    if filter_mode == "Numerical Only":
        filtered_profile = filtered_profile[filtered_profile["Type"].str.contains("Numeric|Target")]
    elif filter_mode == "Categorical Only":
        filtered_profile = filtered_profile[filtered_profile["Type"] == "Categorical"]
    elif filter_mode == "Missing Values Only (>0)":
        filtered_profile = filtered_profile[filtered_profile["Missing"] > 0]
    elif filter_mode == "High Cardinality (>25)":
        filtered_profile = filtered_profile[filtered_profile["Unique"] > 25]
    elif filter_mode == "Potential ID / Leakage":
        filtered_profile = filtered_profile[filtered_profile["Type"].str.contains("Identifier|Leakage")]

    if search_query:
        filtered_profile = filtered_profile[filtered_profile["Feature"].str.contains(search_query, case=False)]

    st.markdown(f"Showing **{len(filtered_profile)}** of **{len(profile_df)}** columns.")
    st.dataframe(filtered_profile, use_container_width=True, height=450)
    st.markdown("</div>", unsafe_allow_html=True)
