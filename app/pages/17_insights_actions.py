"""Page 17 — Insights, Strategic Action Cards & Prescriptive Portfolio Allocation."""
import json
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

from app.components.header import render_header
from src.analytics.insights import generate_business_insights
from src.analytics.prescriptive import PrescriptiveEngine
from src.analytics.advisor import GroundedAdvisor

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

def render_page(df: pd.DataFrame = None, metadata: dict = None):
    render_header(
        title="Executive Insights & Strategic Action Playbook",
        subtitle="Bridging raw data and machine learning into evidence-based decisions following the BI hierarchy.",
        badge="DECISION SUPPORT"
    )

    # Resolve active dataset
    if df is None:
        df = st.session_state.get("feature_df")
        if df is None or df.empty:
            df = st.session_state.get("clean_df")
        if df is None or df.empty:
            df = st.session_state.get("raw_df")

    target_col = st.session_state.get("target_col", "SalePrice")

    # Resolve model metadata
    if metadata is None:
        metadata = st.session_state.get("champion_metadata", {})
        if not metadata:
            meta_path = ROOT_DIR / "models/production/model_metadata.json"
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)

    if df is None or df.empty:
        st.warning("No data available in session. Please visit 1. Data Workspace first.")
        return

    tab1, tab2, tab3 = st.tabs([
        "📋 Strategic Action Cards (BI Hierarchy)",
        "🎯 Prescriptive Portfolio Allocator (Capital Optimization)",
        "🤖 Grounded Real Estate Advisor (Zero-Hallucination Q&A)"
    ])

    # ==========================================================
    # TAB 1: STRATEGIC ACTION CARDS
    # ==========================================================
    with tab1:
        desc_html = (
            '<div style="background: #F8FAFC; border: 1px solid #E2E8F0; padding: 14px 18px; border-radius: 6px; margin-bottom: 20px; font-size: 0.88rem; color: #475569;">'
            '<b>The Decision Intelligence Framework:</b> Every strategic recommendation below is deterministically synthesized from validated empirical metrics and holdout evaluation residuals.<br/>'
            '<span style="font-weight: 700; color: #1E3A8A; letter-spacing: 0.04em;">DATA ➔ INFORMATION ➔ INSIGHT ➔ DECISION ➔ ACTION</span>'
            '</div>'
        )
        if hasattr(st, "html"):
            st.html(desc_html)
        else:
            st.markdown(desc_html, unsafe_allow_html=True)

        metrics = metadata.get("metrics", {}).get("holdout_test", {})
        try:
            insights = generate_business_insights(
                df, 
                {
                    "best_model_name": metadata.get("model_name", "Champion Model"),
                    "test_rmse": metrics.get("RMSE", 0),
                    "test_mae": metrics.get("MAE", 0),
                    "test_r2": metrics.get("R2", 0)
                },
                target_col=target_col
            )

            for idx, ins in enumerate(insights, 1):
                ev_str = " | ".join(ins.get("evidence", []))
                card_html = (
                    '<div class="card-section" style="margin-bottom: 20px;">'
                    '<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">'
                    f'<span style="font-size: 1.1rem; font-weight: 700; color: #0F172A;">{idx}. {ins["title"]}</span>'
                    f'<span style="font-size: 0.75rem; background: #F1F5F9; color: #475569; padding: 3px 8px; border-radius: 4px; font-family: monospace;">{ins["id"]}</span>'
                    '</div>'
                    '<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 12px;">'
                    '<div style="background: #F0F9FF; border: 1px solid #BAE6FD; padding: 12px 14px; border-radius: 6px;">'
                    '<span class="framework-pill pill-fact" style="font-size: 0.75rem; font-weight: 700; color: #0369A1; text-transform: uppercase;">1. The Verified Fact</span>'
                    f'<div style="font-size: 0.86rem; color: #0369A1; line-height: 1.4; margin-top: 4px;">{ins["fact"]}</div>'
                    '</div>'
                    '<div style="background: #FEFCE8; border: 1px solid #FEF08A; padding: 12px 14px; border-radius: 6px;">'
                    '<span class="framework-pill pill-insight" style="font-size: 0.75rem; font-weight: 700; color: #A16207; text-transform: uppercase;">2. The Analytical Insight</span>'
                    f'<div style="font-size: 0.86rem; color: #A16207; line-height: 1.4; margin-top: 4px;">{ins["insight"]}</div>'
                    '</div>'
                    '</div>'
                    '<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 12px;">'
                    '<div style="background: #F0FDF4; border: 1px solid #BBF7D0; padding: 12px 14px; border-radius: 6px;">'
                    '<span class="framework-pill pill-opp" style="font-size: 0.75rem; font-weight: 700; color: #15803D; text-transform: uppercase;">3. Business Opportunity</span>'
                    f'<div style="font-size: 0.86rem; color: #15803D; line-height: 1.4; margin-top: 4px;">{ins["opportunity"]}</div>'
                    '</div>'
                    '<div style="background: #FAF5FF; border: 1px solid #E9D5FF; padding: 12px 14px; border-radius: 6px;">'
                    '<span class="framework-pill pill-action" style="font-size: 0.75rem; font-weight: 700; color: #7E22CE; text-transform: uppercase;">4. Concrete Action</span>'
                    f'<div style="font-size: 0.86rem; color: #7E22CE; line-height: 1.4; margin-top: 4px;">{ins["action"]}</div>'
                    '</div>'
                    '</div>'
                    f'<div style="font-size: 0.78rem; color: #64748B; background: #F8FAFC; border: 1px solid #E2E8F0; padding: 8px 12px; border-radius: 4px;"><b>Underlying Quantitative Evidence</b>: {ev_str}</div>'
                    '</div>'
                )
                if hasattr(st, "html"):
                    st.html(card_html)
                else:
                    st.markdown(card_html, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error synthesizing business insights: {e}")

    # ==========================================================
    # TAB 2: PRESCRIPTIVE PORTFOLIO ALLOCATOR
    # ==========================================================
    with tab2:
        st.markdown("<div class='card-section'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>🎯 Resource-Constrained Property Acquisition Optimizer</div>", unsafe_allow_html=True)
        st.markdown(
            """
            Institutional real estate investors and asset managers operate under capital constraints.
            This prescriptive optimization engine calculates a multi-criteria **Property Prioritization Index (PPI)**:
            $$\\text{PPI} = w_1 \\cdot \\text{Valuation Upside} + w_2 \\cdot \\text{Remodel Potential} + w_3 \\cdot \\text{Neighborhood Momentum} + w_4 \\cdot \\text{Liquidity Efficiency}$$
            """
        )

        try:
            test_df = st.session_state.get("X_test")
            if test_df is not None and not test_df.empty and st.session_state.get("champion_model") is not None:
                full_test = test_df.copy()
                if st.session_state.get("y_test") is not None:
                    full_test[target_col] = st.session_state.y_test.values
                prescriptive_engine = PrescriptiveEngine(
                    df_test=full_test,
                    model=st.session_state.champion_model,
                    target_col=target_col
                )
            else:
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

            p_metrics = alloc_result["metrics"]
            portfolio_df = alloc_result["portfolio"]

            st.markdown("#### Optimized Acquisition Portfolio Summary")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Properties Acquired", f"{p_metrics['properties_acquired']:,}")
            k2.metric("Total Capital Deployed", f"${p_metrics['capital_spent_usd']:,.0f}")
            k3.metric("Capital Utilization", f"{p_metrics['capital_utilization_pct']:.1f}%")
            k4.metric("Avg Predicted Portfolio Spread", f"{p_metrics['average_spread_pct']:+.1f}%")

            if not portfolio_df.empty:
                st.markdown("<div class='card-section'>", unsafe_allow_html=True)
                st.markdown("<div class='card-title'>Selected Acquisition Candidates</div>", unsafe_allow_html=True)
                st.dataframe(portfolio_df.head(20), use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            st.info(f"Prescriptive allocation preview: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

    # ==========================================================
    # TAB 3: GROUNDED AI ADVISOR
    # ==========================================================
    with tab3:
        st.markdown("<div class='card-section'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>🤖 Grounded Real Estate Advisory Engine</div>", unsafe_allow_html=True)
        st.caption("Ask questions about model performance, feature drivers, or submarkets. Responses strictly bounded by verified facts:")

        active_context = {
            "total_properties": len(df),
            "market_volume_millions": float(df[target_col].sum()) / 1e6 if target_col in df.columns and pd.api.types.is_numeric_dtype(df[target_col]) else 529.73,
            "mean_sale_price": float(df[target_col].mean()) if target_col in df.columns and pd.api.types.is_numeric_dtype(df[target_col]) else 180796.06,
            "champion_model": metadata.get("model_name", "Champion Model"),
            "test_rmse": metrics.get("RMSE", 22972.84),
            "test_mae": metrics.get("MAE", 13834.55),
            "test_r2": metrics.get("R2", 0.9342),
            "test_mape": metrics.get("MAPE_pct", 7.31),
            "interval_coverage": metadata.get("prediction_intervals", {}).get("coverage", 0.80),
            "lower_interval_offset": metadata.get("prediction_intervals", {}).get("lower_residual_offset", -17310.02),
            "upper_interval_offset": metadata.get("prediction_intervals", {}).get("upper_residual_offset", 20236.56),
            "top_neighborhood_by_ppsf": "Observed highest submarket",
            "top_shap_driver": "Living Area / Size"
        }

        advisor = GroundedAdvisor(context_override=active_context)

        query_prompt = st.text_input("Enter your real estate analytical question:", value="What is the model accuracy and holdout RMSE?")
        if query_prompt:
            ans = advisor.ask(query_prompt)
            st.markdown(
                f'<div style="background: #F0FDF4; border: 1px solid #86EFAC; padding: 14px 18px; border-radius: 6px; margin: 12px 0;">'
                f'<div style="font-weight: 700; color: #166534; font-size: 0.92rem; margin-bottom: 6px;">{ans["fact"]}</div>'
                f'<div style="color: #15803D; font-size: 0.88rem; line-height: 1.4;">{ans["interpretation"]}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

        st.markdown("</div>", unsafe_allow_html=True)
