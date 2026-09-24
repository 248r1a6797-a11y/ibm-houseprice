"""Page 7 — Insights & Actions Engine."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from app.components.header import render_header
from src.analytics.insights import generate_business_insights
from src.analytics.prescriptive import PrescriptiveEngine
from src.analytics.advisor import GroundedAdvisor

def render_page(df: pd.DataFrame, metadata: dict):
    render_header(
        title="Business Insights & Strategic Action Engine",
        subtitle="Bridging raw data and machine learning into evidence-based decisions following the BI hierarchy.",
        badge="DECISION SUPPORT"
    )

    tab1, tab2, tab3 = st.tabs([
        "📋 Strategic Action Cards (BI Hierarchy)",
        "🎯 Prescriptive Portfolio Allocator (Capital Optimization)",
        "🤖 Grounded Real Estate Advisor (Zero-Hallucination Q&A)"
    ])

    # ==========================================================
    # TAB 1: STRATEGIC ACTION CARDS
    # ==========================================================
    with tab1:
        st.markdown(
            """
            <div style="background: #F8FAFC; border: 1px solid #E2E8F0; padding: 14px 18px; border-radius: 6px; margin-bottom: 20px; font-size: 0.88rem; color: #475569;">
                <b>The Decision Intelligence Framework:</b> Every strategic recommendation below is deterministically synthesized from validated empirical metrics and holdout evaluation residuals.
                $$\\text{DATA} \\longrightarrow \\text{INFORMATION} \\longrightarrow \\text{INSIGHT} \\longrightarrow \\text{DECISION} \\longrightarrow \\text{ACTION}$$
            </div>
            """,
            unsafe_allow_html=True
        )

        metrics = metadata.get("metrics", {}).get("holdout_test", {})
        insights = generate_business_insights(df, {
            "best_model_name": metadata.get("model_name", "XGBoost"),
            "test_rmse": metrics.get("RMSE", 22972),
            "test_mae": metrics.get("MAE", 13835),
            "test_r2": metrics.get("R2", 0.9342)
        })

        for idx, ins in enumerate(insights, 1):
            st.markdown(
                f"""
                <div class="card-section">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                        <span style="font-size: 1.1rem; font-weight: 700; color: #0F172A;">{idx}. {ins['title']}</span>
                        <span style="font-size: 0.75rem; background: #F1F5F9; color: #475569; padding: 3px 8px; border-radius: 4px; font-family: monospace;">{ins['id']}</span>
                    </div>
                    
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 12px;">
                        <div style="background: #F0F9FF; border: 1px solid #BAE6FD; padding: 12px 14px; border-radius: 6px;">
                            <span class="framework-pill pill-fact">1. The Verified Fact</span>
                            <div style="font-size: 0.86rem; color: #0369A1; line-height: 1.4;">{ins['fact']}</div>
                        </div>
                        <div style="background: #FEFCE8; border: 1px solid #FEF08A; padding: 12px 14px; border-radius: 6px;">
                            <span class="framework-pill pill-insight">2. The Analytical Insight</span>
                            <div style="font-size: 0.86rem; color: #A16207; line-height: 1.4;">{ins['insight']}</div>
                        </div>
                    </div>

                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 12px;">
                        <div style="background: #F0FDF4; border: 1px solid #BBF7D0; padding: 12px 14px; border-radius: 6px;">
                            <span class="framework-pill pill-opp">3. Business Opportunity</span>
                            <div style="font-size: 0.86rem; color: #15803D; line-height: 1.4;">{ins['opportunity']}</div>
                        </div>
                        <div style="background: #FAF5FF; border: 1px solid #E9D5FF; padding: 12px 14px; border-radius: 6px;">
                            <span class="framework-pill pill-action">4. Concrete Action</span>
                            <div style="font-size: 0.86rem; color: #7E22CE; line-height: 1.4;">{ins['action']}</div>
                        </div>
                    </div>

                    <div style="font-size: 0.78rem; color: #64748B; background: #F8FAFC; border: 1px solid #E2E8F0; padding: 8px 12px; border-radius: 4px;">
                        <b>Underlying Quantitative Evidence</b>: {' | '.join(ins['evidence'])}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # ==========================================================
    # TAB 2: PRESCRIPTIVE PORTFOLIO ALLOCATOR
    # ==========================================================
    with tab2:
        st.markdown("<div class='card-section'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>🎯 Resource-Constrained Property Acquisition Optimizer</div>", unsafe_allow_html=True)
        st.markdown(
            """
            Institutional real estate investors, municipal redevelopment authorities, and REIT asset managers operate under capital constraints.
            This prescriptive optimization engine calculates a multi-criteria **Property Prioritization Index (PPI)** across the holdout validation portfolio:
            $$\\text{PPI} = w_1 \\cdot \\text{Valuation Upside} + w_2 \\cdot \\text{Remodel Potential} + w_3 \\cdot \\text{Neighborhood Momentum} + w_4 \\cdot \\text{Liquidity Efficiency}$$
            """
        )

        try:
            prescriptive_engine = PrescriptiveEngine()
            
            p_col1, p_col2 = st.columns([1, 1])
            with p_col1:
                budget_m = st.slider("Total Capital Acquisition Budget ($ Millions):", 0.5, 15.0, 3.0, 0.5)
                budget_usd = budget_m * 1000000.0
                
                selected_tiers = st.multiselect(
                    "Eligible Action Tiers to Include:",
                    [
                        "Tier 1: Deep Value Opportunity",
                        "Tier 2: Fair Market Value",
                        "Tier 3: Overvaluation Risk",
                        "Tier 4: Speculative / High Volatility"
                    ],
                    default=["Tier 1: Deep Value Opportunity", "Tier 2: Fair Market Value"]
                )

            with p_col2:
                st.caption("Customize Prioritization Weights (Automatically Normalized):")
                w_up = st.slider("Valuation Upside Spread Weight:", 0.0, 1.0, 0.40, 0.05)
                w_rem = st.slider("Condition & Remodel Potential Weight:", 0.0, 1.0, 0.25, 0.05)
                w_neigh = st.slider("Neighborhood Momentum Weight:", 0.0, 1.0, 0.20, 0.05)
                w_liq = st.slider("Liquidity & Size Efficiency Weight:", 0.0, 1.0, 0.15, 0.05)

            alloc_result = prescriptive_engine.allocate_capital_budget(
                capital_budget_usd=budget_usd,
                min_tier_filter=selected_tiers if selected_tiers else None,
                weight_upside=w_up,
                weight_remodel=w_rem,
                weight_neighborhood=w_neigh,
                weight_liquidity=w_liq
            )

            metrics_alloc = alloc_result["metrics"]
            portfolio_df = alloc_result["portfolio"]

            st.markdown("---")
            kpi_c1, kpi_c2, kpi_c3, kpi_c4, kpi_c5 = st.columns(5)
            kpi_c1.metric("Properties Acquired", f"{metrics_alloc.get('properties_acquired', 0)}")
            kpi_c2.metric("Capital Deployed", f"${metrics_alloc.get('capital_spent_usd', 0)/1e6:.2f}M")
            kpi_c3.metric("Capital Utilization", f"{metrics_alloc.get('capital_utilization_pct', 0):.1f}%")
            kpi_c4.metric("Net Equity Spread", f"${metrics_alloc.get('net_equity_spread_usd', 0):,.0f}")
            kpi_c5.metric("Expected Portfolio ROI", f"{metrics_alloc.get('expected_portfolio_roi_pct', 0):.1f}%")

            if not portfolio_df.empty:
                st.markdown("##### 🏆 Optimal Acquisition Cohort (Ranked by Composite PPI Score)")
                display_cols = [
                    "PID", "Neighborhood", "OverallQual", "YearBuilt", "GrLivArea",
                    "SalePrice", "Predicted_Price", "Valuation_Spread_Pct", "PPI_Score", "Action_Tier"
                ]
                avail_disp = [c for c in display_cols if c in portfolio_df.columns]
                st.dataframe(portfolio_df[avail_disp], use_container_width=True)

                port_csv = portfolio_df[avail_disp].to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Export Optimized Acquisition Portfolio CSV",
                    data=port_csv,
                    file_name="optimized_acquisition_portfolio.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No properties matched the selected tier filter within the specified capital budget.")

        except Exception as e:
            st.error(f"Prescriptive Engine Error: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

    # ==========================================================
    # TAB 3: GROUNDED AI REAL ESTATE ADVISOR
    # ==========================================================
    with tab3:
        st.markdown("<div class='card-section'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>🤖 Zero-Hallucination Real Estate Advisor</div>", unsafe_allow_html=True)
        st.markdown(
            """
            Built on the **Zero-Hallucination Architecture** popularized by enterprise AI pipelines.
            Every generated response is strictly grounded in verified Ames Housing assessor data, SQL query results,
            and holdout model evaluation metrics. Facts are separated cleanly from advisory interpretations.
            """
        )

        try:
            advisor = GroundedAdvisor()
            
            st.markdown("##### 💡 Suggested Questions:")
            q_pills = [
                "What is the model's accuracy, RMSE, and prediction error margin?",
                "What are the top valuation drivers influencing property prices?",
                "Which neighborhoods have the highest price per square foot?",
                "What is the valuation premium for remodeled homes vs original condition?",
                "Why did XGBoost beat the Deep Neural Network (MLP) in the benchmark?"
            ]
            
            selected_preset = st.radio("Choose a quick inquiry or type below:", ["(Custom Question)"] + q_pills)
            
            custom_q = st.text_input(
                "Ask a Question to the Advisor:",
                value="" if selected_preset == "(Custom Question)" else selected_preset
            )

            if custom_q:
                ans = advisor.ask(custom_q)
                
                st.markdown(
                    f"""
                    <div style="background: #F0F9FF; border-left: 4px solid #0284C7; padding: 14px 18px; border-radius: 4px; margin-top: 15px; margin-bottom: 12px;">
                        <div style="font-weight: 700; color: #0369A1; font-size: 0.85rem; margin-bottom: 6px;">VERIFIED STATISTICAL FACT</div>
                        <div style="color: #0F172A; font-size: 0.92rem; line-height: 1.5;">{ans['fact']}</div>
                    </div>
                    <div style="background: #FAF5FF; border-left: 4px solid #9333EA; padding: 14px 18px; border-radius: 4px; margin-bottom: 15px;">
                        <div style="font-weight: 700; color: #7E22CE; font-size: 0.85rem; margin-bottom: 6px;">STRATEGIC INTERPRETATION & ACTION</div>
                        <div style="color: #0F172A; font-size: 0.92rem; line-height: 1.5;">{ans['interpretation']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
        except Exception as e:
            st.error(f"Advisor Engine Error: {e}")

        st.markdown("</div>", unsafe_allow_html=True)
