"""Page 6 — Risk & Error Analysis."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from app.components.header import render_header
from app.components.kpi_cards import render_kpi_card

def render_page():
    render_header(
        title="Model Risk & Error Diagnostics",
        subtitle="Forensic examination of residuals, segment-specific failure modes, worst-case prediction errors, and model boundaries.",
        badge="ERROR ANALYSIS"
    )

    # Load holdout test data and worst prediction forensics
    try:
        worst_under = pd.read_csv("reports/tables/worst_underpredictions.csv")
        worst_over = pd.read_csv("reports/tables/worst_overpredictions.csv")
    except Exception:
        worst_under = pd.DataFrame()
        worst_over = pd.DataFrame()

    # KPI Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_kpi_card("Holdout Test MAE", "$13,835", "Average absolute error")
    with col2:
        render_kpi_card("Holdout Test RMSE", "$22,973", "Penalizes large variances")
    with col3:
        render_kpi_card("Holdout Median AE", "$9,071", "Typical non-outlier error")
    with col4:
        render_kpi_card("Holdout Test MAPE", "7.31%", "Average percentage deviation")

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Residual Diagnostics & Heteroscedasticity", "Worst-Case Prediction Forensics", "Risk Mitigation Framework"])

    with tab1:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("<div class='card-section'>", unsafe_allow_html=True)
            st.markdown("<div class='card-title'>Actual vs Predicted (Holdout Test Set)</div>", unsafe_allow_html=True)
            st.image("reports/figures/actual_vs_predicted.png", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown("<div class='card-section'>", unsafe_allow_html=True)
            st.markdown("<div class='card-title'>Residual Distribution (Actual - Predicted)</div>", unsafe_allow_html=True)
            st.image("reports/figures/residual_distribution.png", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        st.markdown("<div class='card-section'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Top 10 Worst Underpredictions (Actual >> Predicted)</div>", unsafe_allow_html=True)
        st.markdown("Properties where the model severely undervalued the property. Often driven by unmodeled luxury finishes, partial sales, or unique architectural custom craftsmanship.")
        if not worst_under.empty:
            show_cols = [c for c in ["Neighborhood", "OverallQual", "GrLivArea", "YearBuilt", "actual", "predicted", "residual", "percentage_error"] if c in worst_under.columns]
            st.dataframe(
                worst_under[show_cols].style.format({
                    "actual": "${:,.0f}",
                    "predicted": "${:,.0f}",
                    "residual": "+${:,.0f}",
                    "percentage_error": "{:.1f}%",
                    "GrLivArea": "{:,.0f} sq ft"
                }),
                use_container_width=True
            )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card-section'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Top 10 Worst Overpredictions (Actual << Predicted)</div>", unsafe_allow_html=True)
        st.markdown("Properties where the model overvalued the property. Often associated with abnormal sale conditions (e.g. foreclosure, family transfer, physical structural defects not captured in assessor tabular columns).")
        if not worst_over.empty:
            show_cols = [c for c in ["Neighborhood", "OverallQual", "GrLivArea", "YearBuilt", "actual", "predicted", "residual", "percentage_error"] if c in worst_over.columns]
            st.dataframe(
                worst_over[show_cols].style.format({
                    "actual": "${:,.0f}",
                    "predicted": "${:,.0f}",
                    "residual": "${:,.0f}",
                    "percentage_error": "{:.1f}%",
                    "GrLivArea": "{:,.0f} sq ft"
                }),
                use_container_width=True
            )
        st.markdown("</div>", unsafe_allow_html=True)

    with tab3:
        st.markdown("<div class='card-section'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Strategic Model Risk & Operational Guidance</div>", unsafe_allow_html=True)
        
        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown(
                """
                <div class="framework-card">
                    <span class="framework-pill pill-fact">Risk 1: Luxury Tier Heteroscedasticity</span>
                    <p style="font-size: 0.85rem; color: #334155; margin-top: 6px;">
                        <b>Evidence</b>: Properties >$350k have a higher median absolute error ($28k) than mid-range properties ($9k).<br/>
                        <b>Impact</b>: Overconfidence on high-end custom properties.<br/>
                        <b>Recommendation</b>: Mandate a professional appraiser review whenever an automated valuation exceeds $350,000.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
        with r2:
            st.markdown(
                """
                <div class="framework-card">
                    <span class="framework-pill pill-fact">Risk 2: Low-Volume Neighborhoods</span>
                    <p style="font-size: 0.85rem; color: #334155; margin-top: 6px;">
                        <b>Evidence</b>: Neighborhoods with fewer than 15 historical sales (e.g. Green Hills, Landmark) carry wider prediction variance.<br/>
                        <b>Impact</b>: Neighborhood dummy coefficients lack statistical power.<br/>
                        <b>Recommendation</b>: Group low-density areas into broader municipal zoning tiers.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
        with r3:
            st.markdown(
                """
                <div class="framework-card">
                    <span class="framework-pill pill-fact">Risk 3: Temporal & Economic Drift</span>
                    <p style="font-size: 0.85rem; color: #334155; margin-top: 6px;">
                        <b>Evidence</b>: Dataset spans 2006 to 2010 during the mortgage crisis.<br/>
                        <b>Impact</b>: Does not reflect modern interest rate regimes or post-2020 inflation.<br/>
                        <b>Recommendation</b>: Re-calibrate base intercept with local CPI / Case-Shiller index before deploying to current market.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
        st.markdown("</div>", unsafe_allow_html=True)
