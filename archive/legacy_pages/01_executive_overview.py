"""Page 1 — Executive Overview."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from app.components.header import render_header
from app.components.kpi_cards import render_kpi_card
from src.analytics.descriptive import compute_market_kpis
from src.analytics.insights import generate_business_insights

def render_page(df: pd.DataFrame, metadata: dict, model_comp_df: pd.DataFrame):
    render_header(
        title="Executive Intelligence Overview",
        subtitle="Market-level KPIs, predictive performance health, top price drivers, and strategic risk assessment.",
        badge="EXECUTIVE DASHBOARD"
    )

    kpis = compute_market_kpis(df)
    metrics = metadata.get("metrics", {}).get("holdout_test", {})

    # Top KPI Row
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        render_kpi_card("Total Properties", f"{kpis['total_properties']:,}", "Verified assessor records")
    with col2:
        render_kpi_card("Median Price", f"${kpis['median_sale_price']:,.0f}", f"Mean: ${kpis['average_sale_price']:,.0f}")
    with col3:
        render_kpi_card("Median Living Area", f"{kpis['median_living_area']:,.0f} sq ft", f"Unit: ${kpis['median_price_per_sqft']:.0f}/sq ft")
    with col4:
        render_kpi_card("Champion Model", metadata.get("model_name", "XGBoost"), f"v{metadata.get('model_version', '1.0.0')}")
    with col5:
        render_kpi_card("Holdout Test RMSE", f"${metrics.get('RMSE', 22972):,.0f}", f"MAE: ${metrics.get('MAE', 13835):,.0f}")
    with col6:
        render_kpi_card("Test R² (Variance)", f"{metrics.get('R2', 0.9342):.1%}", "Generalization accuracy")

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Section A & B: Market Snapshot & Price Trend
    row1_col1, row1_col2 = st.columns([1.1, 0.9])

    with row1_col1:
        st.markdown("<div class='card-section'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Market Snapshot — Sale Price Distribution</div>", unsafe_allow_html=True)
        fig_dist = px.histogram(
            df,
            x="SalePrice",
            nbins=40,
            color_discrete_sequence=["#1E3A8A"],
            opacity=0.85,
            labels={"SalePrice": "Sale Price ($)"}
        )
        fig_dist.add_vline(x=kpis["median_sale_price"], line_dash="dash", line_color="#DC2626", annotation_text=f"Median (${kpis['median_sale_price']:,.0f})")
        fig_dist.add_vline(x=kpis["average_sale_price"], line_dash="dot", line_color="#D97706", annotation_text=f"Mean (${kpis['average_sale_price']:,.0f})")
        fig_dist.update_layout(margin=dict(l=20, r=20, t=30, b=20), height=320, plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_dist, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with row1_col2:
        st.markdown("<div class='card-section'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Historical Price Dynamics (2006–2010)</div>", unsafe_allow_html=True)
        if "YrSold" in df.columns:
            yearly = df.groupby("YrSold")["SalePrice"].agg(["median", "count"]).reset_index()
            fig_trend = px.line(
                yearly,
                x="YrSold",
                y="median",
                markers=True,
                line_shape="spline",
                color_discrete_sequence=["#2563EB"],
                labels={"YrSold": "Year Sold", "median": "Median Sale Price ($)"}
            )
            fig_trend.update_layout(margin=dict(l=20, r=20, t=30, b=20), height=320, plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("Temporal dimension not present.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Section C & D: Top Price Drivers & Model Health
    row2_col1, row2_col2 = st.columns([1, 1])

    with row2_col1:
        st.markdown("<div class='card-section'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Top Price Drivers (SHAP Global Attribution)</div>", unsafe_allow_html=True)
        # Load SHAP summary table if available
        try:
            shap_df = pd.read_csv("reports/tables/shap_feature_importance.csv").head(8)
            fig_shap = px.bar(
                shap_df,
                x="mean_abs_shap",
                y="clean_feature",
                orientation="h",
                color="mean_abs_shap",
                color_continuous_scale="Blues",
                labels={"mean_abs_shap": "Mean |SHAP Value| (Impact on Log Price)", "clean_feature": "Property Feature"}
            )
            fig_shap.update_layout(yaxis=dict(autorange="reversed"), margin=dict(l=20, r=20, t=30, b=20), height=300, coloraxis_showscale=False)
            st.plotly_chart(fig_shap, use_container_width=True)
        except Exception:
            st.info("SHAP feature importance table is being generated.")
        st.markdown("</div>", unsafe_allow_html=True)

    with row2_col2:
        st.markdown("<div class='card-section'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Model Health & Benchmark Leaderboard</div>", unsafe_allow_html=True)
        if not model_comp_df.empty:
            display_comp = model_comp_df.head(6)[["Model", "Test_RMSE", "Test_MAE", "Test_R2", "Overfit_Gap_RMSE"]].copy()
            display_comp.columns = ["Model", "Test RMSE ($)", "Test MAE ($)", "Test R²", "Overfit Gap ($)"]
            st.dataframe(
                display_comp.style.format({
                    "Test RMSE ($)": "${:,.0f}",
                    "Test MAE ($)": "${:,.0f}",
                    "Test R²": "{:.3f}",
                    "Overfit Gap ($)": "${:,.0f}"
                }),
                use_container_width=True,
                height=260
            )
        st.markdown("</div>", unsafe_allow_html=True)

    # Section E, F, G: Insights, Risks & Strategic Actions
    st.markdown("<div class='card-section'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>Executive Decision Support & Risk Briefing</div>", unsafe_allow_html=True)
    
    col_ins, col_risk = st.columns([1.2, 0.8])
    with col_ins:
        st.markdown("#### Primary Evidence-Backed Insights")
        insights = generate_business_insights(df, {
            "best_model_name": metadata.get("model_name", "XGBoost"),
            "test_rmse": metrics.get("RMSE", 22972),
            "test_mae": metrics.get("MAE", 13835),
            "test_r2": metrics.get("R2", 0.9342)
        })
        for ins in insights[:3]:
            st.markdown(
                f"""
                <div class="framework-card">
                    <span class="framework-pill pill-fact">Fact</span> <b>{ins['title']}</b><br/>
                    <div style="font-size: 0.88rem; color: #334155; margin-top: 4px;">{ins['fact']}</div>
                    <div style="margin-top: 8px;">
                        <span class="framework-pill pill-action">Recommended Action</span>
                        <div style="font-size: 0.85rem; color: #1E293B;">{ins['action']}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    with col_risk:
        st.markdown("#### Risk & Governance Watchlist")
        st.warning(
            "**Luxury Tier Heteroscedasticity**: Error variance expands significantly on properties exceeding $320,000. Large luxury custom estates carry higher absolute residual risk."
        )
        st.info(
            "**Geographic Scope Restriction**: Trained strictly on historical Ames, Iowa municipal data (2006–2010). Model must not be applied to non-Midwestern or modern post-inflation markets without retraining."
        )
        st.error(
            "**Non-Causal Interpretability**: SHAP attributions represent predictive associations within historical data, not causal property guarantees."
        )
    st.markdown("</div>", unsafe_allow_html=True)
