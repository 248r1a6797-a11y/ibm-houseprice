"""Page 18 — Executive Strategic Overview Dashboard."""
import json
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from app.components.header import render_header
from app.components.kpi_cards import render_kpi_card
from src.analytics.descriptive import compute_market_kpis
from src.explainability.shap_analysis import compute_global_shap

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

def render_page(df: pd.DataFrame = None, metadata: dict = None, model_comp_df: pd.DataFrame = None):
    render_header(
        title="Executive Intelligence & Production Model Overview",
        subtitle="Market-level KPIs, predictive performance health, top price drivers, and strategic risk assessment.",
        badge="EXECUTIVE DASHBOARD"
    )

    # Resolve active dataset
    if df is None:
        df = st.session_state.get("feature_df")
        if df is None or df.empty:
            df = st.session_state.get("clean_df")
        if df is None or df.empty:
            df = st.session_state.get("raw_df")

    target_col = st.session_state.get("target_col", "SalePrice")

    # Resolve metadata
    if metadata is None:
        metadata = st.session_state.get("champion_metadata", {})
        if not metadata:
            meta_path = ROOT_DIR / "models/production/model_metadata.json"
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)

    # Resolve benchmark comparison
    if model_comp_df is None:
        model_comp_df = st.session_state.get("benchmark_results")
        if model_comp_df is None or model_comp_df.empty:
            comp_path = ROOT_DIR / "reports/tables/model_comparison.csv"
            if comp_path.exists():
                model_comp_df = pd.read_csv(comp_path)
            else:
                model_comp_df = pd.DataFrame()

    if df is None or df.empty:
        st.warning("No dataset loaded in session. Please visit 1. Data Workspace.")
        return

    kpis = compute_market_kpis(df, target_col=target_col)
    metrics = metadata.get("metrics", {}).get("holdout_test", {})

    active_champ = st.session_state.get("champion_name", metadata.get("model_name", "Champion Model"))
    
    # Active holdout metrics
    if model_comp_df is not None and not model_comp_df.empty:
        champ_row = model_comp_df.iloc[0]
        r2_val = champ_row.get("Test_R2", champ_row.get("CV_R2", 0.0))
        rmse_val = champ_row.get("Test_RMSE", champ_row.get("CV_RMSE", 0.0))
        mae_val = champ_row.get("Test_MAE", champ_row.get("CV_MAE", 0.0))
    else:
        r2_val = metrics.get("R2", 0.0)
        rmse_val = metrics.get("RMSE", 0.0)
        mae_val = metrics.get("MAE", 0.0)

    # Top KPI Row (Dynamic labels)
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        render_kpi_card("Total Properties", f"{kpis['total_properties']:,}", f"{st.session_state.dataset_name.split()[0]}")
    with col2:
        render_kpi_card("Median Target Value", f"${kpis['median_sale_price']:,.0f}", f"Mean: ${kpis['average_sale_price']:,.0f}")
    with col3:
        render_kpi_card("Median Living Area", f"{kpis['median_living_area']:,.0f} sq ft" if kpis['median_living_area'] > 0 else "N/A", f"Unit: ${kpis['median_price_per_sqft']:.0f}/sq ft" if kpis['median_price_per_sqft'] > 0 else "")
    with col4:
        render_kpi_card("Champion Model", str(active_champ), f"v{metadata.get('model_version', '1.0.0')}")
    with col5:
        render_kpi_card("Holdout Test RMSE", f"${rmse_val:,.0f}" if rmse_val > 0 else "Awaiting Run", f"MAE: ${mae_val:,.0f}" if mae_val > 0 else "")
    with col6:
        render_kpi_card("Explained Variance", f"{r2_val:.3f}" if r2_val > 0 else "Awaiting Run", "Cross-Sectional R²")

    st.markdown("<div class='section-spacer'></div>", unsafe_allow_html=True)

    # Middle Row: Market Valuation Distribution & Model Tournament
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("<div class='card-section'>", unsafe_allow_html=True)
        st.markdown(f"<div class='card-title'>📈 Target Distribution & Skewness ({target_col})</div>", unsafe_allow_html=True)
        if target_col in df.columns and pd.api.types.is_numeric_dtype(df[target_col]):
            fig_hist = px.histogram(
                df,
                x=target_col,
                nbins=35,
                color_discrete_sequence=["#1E3A8A"],
                labels={target_col: f"Target Valuation ({target_col})"}
            )
            fig_hist.update_layout(height=320, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_hist, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_chart2:
        st.markdown("<div class='card-section'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>🏆 Model Tournament Comparison</div>", unsafe_allow_html=True)
        if model_comp_df is not None and not model_comp_df.empty:
            top_comp = model_comp_df.head(6)
            y_col = "Test_RMSE" if "Test_RMSE" in top_comp.columns else "CV_RMSE"
            fig_bar = px.bar(
                top_comp,
                x="Model",
                y=y_col,
                color=y_col,
                color_continuous_scale="Blues_r",
                labels={y_col: f"{y_col} ($) [Lower is Better]"}
            )
            fig_bar.update_layout(height=320, margin=dict(l=20, r=20, t=20, b=20), coloraxis_showscale=False)
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("Execute benchmark tournament in 7. Model Laboratory to display comparison.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Bottom Row: Top Drivers & Pipeline Status
    c_bot1, c_bot2 = st.columns([1.1, 0.9])
    with c_bot1:
        st.markdown("<div class='card-section'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>🌐 Top Empirical Valuation Drivers</div>", unsafe_allow_html=True)
        
        # Check if SHAP importances available
        shap_csv = ROOT_DIR / "reports/tables/shap_feature_importance.csv"
        if shap_csv.exists():
            shap_df = pd.read_csv(shap_csv).head(8)
            f_col = "clean_feature" if "clean_feature" in shap_df.columns else "feature"
            fig_drv = px.bar(
                shap_df,
                x="mean_abs_shap",
                y=f_col,
                orientation="h",
                color="mean_abs_shap",
                color_continuous_scale="Blues",
                labels={"mean_abs_shap": "Mean |Impact| ($)", f_col: "Feature"}
            )
            fig_drv.update_layout(yaxis=dict(autorange="reversed"), height=260, margin=dict(l=10, r=10, t=10, b=10), coloraxis_showscale=False)
            st.plotly_chart(fig_drv, use_container_width=True)
        else:
            st.info("Run 10. Model Explainability to view game-theoretic SHAP drivers.")
        st.markdown("</div>", unsafe_allow_html=True)

    with c_bot2:
        st.markdown("<div class='card-section'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>🚦 End-to-End Pipeline Health & Readiness</div>", unsafe_allow_html=True)
        p_status = st.session_state.get("pipeline_status", {})
        
        cols = st.columns(3)
        stages = list(p_status.items())
        for idx, (stg, status_str) in enumerate(stages):
            col_target = cols[idx % 3]
            is_ready = "✓" in status_str
            color = "#16A34A" if is_ready else "#D97706"
            col_target.markdown(
                f'<div style="background: #F8FAFC; border: 1px solid #E2E8F0; padding: 10px; border-radius: 6px; margin-bottom: 8px; text-align: center;">'
                f'<div style="font-size: 0.72rem; color: #64748B; font-weight: 700;">{stg}</div>'
                f'<div style="font-size: 0.85rem; font-weight: 700; color: {color}; margin-top: 2px;">{status_str}</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        st.markdown("</div>", unsafe_allow_html=True)
