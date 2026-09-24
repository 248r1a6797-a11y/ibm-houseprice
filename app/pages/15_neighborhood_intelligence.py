"""Page 15 — Neighborhood Intelligence & Price-Per-Square-Foot Analytics."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from app.components.header import render_header

def render_page():
    render_header(
        title="Neighborhood Intelligence & Price/SqFt Deep Dive",
        subtitle="Granular spatial valuation benchmarks, submarket price-per-square-foot dispersion, and quality cross-tabulations.",
        badge="SPATIAL INTELLIGENCE"
    )

    df = st.session_state.clean_df if not st.session_state.clean_df.empty else st.session_state.raw_df
    target_col = st.session_state.target_col

    if df.empty or not target_col or target_col not in df.columns:
        st.warning("No data available in session. Please visit 1. Data Workspace.")
        return

    # Dynamic Column Resolution
    ctx = st.session_state.get("dataset_context")
    default_loc = ctx.location_col if ctx and ctx.location_col else ("Neighborhood" if "Neighborhood" in df.columns else None)
    default_area = ctx.size_col if ctx and ctx.size_col else ("GrLivArea" if "GrLivArea" in df.columns else None)

    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    st.markdown("<div class='card-section'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>⚙️ Spatial & Floorplan Feature Mapping</div>", unsafe_allow_html=True)
    
    col_map1, col_map2 = st.columns(2)
    with col_map1:
        loc_candidates = [c for c in df.columns if df[c].nunique() <= 50 and c != target_col]
        if not loc_candidates:
            loc_candidates = cat_cols if cat_cols else list(df.columns)
        def_loc_idx = loc_candidates.index(default_loc) if default_loc in loc_candidates else 0
        loc_col = st.selectbox("Select Location / Submarket Grouping Column:", loc_candidates, index=def_loc_idx)

    with col_map2:
        area_candidates = [c for c in num_cols if c != target_col]
        if area_candidates:
            def_area_idx = area_candidates.index(default_area) if default_area in area_candidates else 0
            area_col = st.selectbox("Select Usable Living Area / Size Column:", area_candidates, index=def_area_idx)
        else:
            area_col = None
            st.selectbox("Select Usable Living Area / Size Column:", ["(None available)"], index=0, disabled=True)

    st.markdown("</div>", unsafe_allow_html=True)

    if not loc_col or not area_col:
        st.info("Please select a location and area column to generate submarket intelligence.")
        return

    df_calc = df.copy()
    valid_mask = (df_calc[area_col] > 0) & (df_calc[target_col] > 0)
    df_calc = df_calc[valid_mask].copy()

    if df_calc.empty:
        st.warning("No valid positive records found for area and target columns.")
        return

    df_calc["Observed_PPSF"] = df_calc[target_col] / df_calc[area_col]

    # Overall PPSF KPIs
    overall_med_ppsf = df_calc["Observed_PPSF"].median()
    overall_high_ppsf = df_calc["Observed_PPSF"].max()
    overall_low_ppsf = df_calc["Observed_PPSF"].min()

    st.markdown("#### 📏 Observed Price per Square Foot Benchmarks")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Overall Median PPSF", f"${overall_med_ppsf:.2f}/sq ft")
    k2.metric("Highest PPSF Observed", f"${overall_high_ppsf:.2f}/sq ft")
    k3.metric("Lowest PPSF Observed", f"${overall_low_ppsf:.2f}/sq ft")
    k4.metric("Submarkets Tracked", f"{df_calc[loc_col].nunique()}")

    # Submarket Summary Table
    st.markdown("<div class='card-section'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>🏘️ Submarket Comparative Intelligence Directory</div>", unsafe_allow_html=True)

    neigh_summary = df_calc.groupby(loc_col).agg(
        Property_Count=(target_col, "count"),
        Median_Price=(target_col, "median"),
        Mean_Price=(target_col, "mean"),
        Median_Living_Area=(area_col, "median"),
        Median_PPSF=("Observed_PPSF", "median")
    ).reset_index().sort_values(by="Median_PPSF", ascending=False)

    st.dataframe(
        neigh_summary.style.format({
            "Median_Price": "${:,.0f}",
            "Mean_Price": "${:,.0f}",
            "Median_Living_Area": "{:,.0f} sq ft",
            "Median_PPSF": "${:,.2f}/sq ft",
            "Property_Count": "{:,}"
        }),
        use_container_width=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Submarket Visual Comparison
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='card-section'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Submarket Median Price per Sq Ft</div>", unsafe_allow_html=True)
        fig_bar = px.bar(
            neigh_summary.head(15),
            x="Median_PPSF",
            y=loc_col,
            orientation="h",
            color="Median_PPSF",
            color_continuous_scale="Blues",
            labels={"Median_PPSF": "Median PPSF ($/sq ft)"}
        )
        fig_bar.update_layout(yaxis=dict(autorange="reversed"), height=380, margin=dict(l=20, r=20, t=20, b=20), coloraxis_showscale=False)
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card-section'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Price Dispersion Across Top Submarkets</div>", unsafe_allow_html=True)
        top_submarkets = neigh_summary.head(8)[loc_col].tolist()
        df_top_sub = df_calc[df_calc[loc_col].isin(top_submarkets)]
        fig_box = px.box(
            df_top_sub,
            x=loc_col,
            y=target_col,
            color=loc_col,
            labels={target_col: "Target Valuation ($)", loc_col: "Submarket"}
        )
        fig_box.update_layout(height=380, margin=dict(l=20, r=20, t=20, b=20), showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
