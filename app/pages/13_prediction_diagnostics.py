"""Page 13 — Prediction Diagnostics & Model Error Forensics."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

from app.components.header import render_header

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

def render_page():
    render_header(
        title="Prediction Diagnostics & Error Analysis",
        subtitle="Forensic analysis of holdout test residuals: evaluate heteroscedasticity, error distributions, and boundary failure modes.",
        badge="DIAGNOSTIC LAYER"
    )

    X_test = st.session_state.get("X_test")
    y_test = st.session_state.get("y_test")
    active_model = st.session_state.get("champion_model")

    if active_model is None or X_test is None or y_test is None or len(X_test) == 0:
        st.info(f"ℹ️ Holdout evaluation set and champion model not yet initialized for `{st.session_state.get('dataset_name', 'current dataset')}`.")
        from app.state import quick_train_baseline_model
        if st.button("🚀 Auto-Train & Evaluate Baseline Model Now", type="primary"):
            with st.spinner("Training baseline model and partitioning holdout test set..."):
                if quick_train_baseline_model(st):
                    st.success("Model trained and evaluated! Loading diagnostics...")
                    st.rerun()
                else:
                    st.error("Could not auto-train model. Please check target variable in 1. Data Workspace.")
        return

    # Generate predictions on test partition
    try:
        preds = active_model.predict(X_test)
        actuals = y_test.values
        residuals = actuals - preds
        pct_errors = np.abs(residuals / np.maximum(1.0, actuals)) * 100.0

        eval_df = pd.DataFrame({
            "Actual_Price": actuals,
            "Predicted_Price": preds,
            "Residual_USD": residuals,
            "Abs_Error_USD": np.abs(residuals),
            "Pct_Error": pct_errors
        })

        # Summary KPIs
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Holdout Observations", f"{len(eval_df):,}")
        k2.metric("Mean Absolute Error (MAE)", f"${eval_df['Abs_Error_USD'].mean():,.0f}")
        k3.metric("Root Mean Squared Error", f"${np.sqrt(np.mean(residuals**2)):,.0f}")
        k4.metric("Mean Absolute % Error", f"{eval_df['Pct_Error'].mean():.2f}%")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<div class='card-section'>", unsafe_allow_html=True)
            st.markdown("<div class='card-title'>Actual vs Predicted Valuation Scatter</div>", unsafe_allow_html=True)
            fig_act_pred = px.scatter(
                eval_df,
                x="Predicted_Price",
                y="Actual_Price",
                opacity=0.7,
                labels={"Predicted_Price": "Model Prediction ($)", "Actual_Price": "Observed Actual Value ($)"},
                title="Predicted vs Observed Valuation (Holdout Set)"
            )
            # Add 45-degree reference line
            max_v = max(eval_df["Actual_Price"].max(), eval_df["Predicted_Price"].max())
            min_v = min(eval_df["Actual_Price"].min(), eval_df["Predicted_Price"].min())
            fig_act_pred.add_shape(
                type="line", line=dict(dash="dash", color="red", width=2),
                x0=min_v, y0=min_v, x1=max_v, y1=max_v
            )
            fig_act_pred.update_layout(height=360, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_act_pred, use_container_width=True)
            st.caption("Red dashed line represents perfect prediction ($y = \\hat{y}$). Symmetrical scatter confirms unbiased calibration.")
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='card-section'>", unsafe_allow_html=True)
            st.markdown("<div class='card-title'>Residual Distribution Histogram</div>", unsafe_allow_html=True)
            fig_res_dist = px.histogram(
                eval_df,
                x="Residual_USD",
                nbins=40,
                color_discrete_sequence=["#1E3A8A"],
                labels={"Residual_USD": "Residual Error ($) [Actual - Predicted]"},
                title="Holdout Residual Frequency Distribution"
            )
            fig_res_dist.update_layout(height=360, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_res_dist, use_container_width=True)
            skew_val = eval_df["Residual_USD"].skew()
            st.caption(f"Residual skewness: **{skew_val:+.3f}**. Near-zero skew indicates well-behaved normal error distribution.")
            st.markdown("</div>", unsafe_allow_html=True)

        # Residual vs Fitted Plot (Heteroscedasticity)
        st.markdown("<div class='card-section'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>📐 Residuals vs Fitted (Heteroscedasticity Diagnostics)</div>", unsafe_allow_html=True)
        fig_res_fit = px.scatter(
            eval_df,
            x="Predicted_Price",
            y="Residual_USD",
            opacity=0.65,
            labels={"Predicted_Price": "Fitted / Predicted Value ($)", "Residual_USD": "Residual ($)"}
        )
        fig_res_fit.add_hline(y=0, line_dash="dash", line_color="black")
        fig_res_fit.update_layout(height=320, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_res_fit, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Worst Under and Over Predictions Table
        st.markdown("<div class='card-section'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>🔍 Worst Prediction Forensics (Boundary Analysis)</div>", unsafe_allow_html=True)
        
        tab_under, tab_over = st.tabs(["Worst Underpredictions (Actual >> Predicted)", "Worst Overpredictions (Actual << Predicted)"])
        with tab_under:
            under_df = eval_df.sort_values(by="Residual_USD", ascending=False).head(5)
            st.dataframe(under_df.style.format({
                "Actual_Price": "${:,.0f}",
                "Predicted_Price": "${:,.0f}",
                "Residual_USD": "+${:,.0f}",
                "Abs_Error_USD": "${:,.0f}",
                "Pct_Error": "{:.1f}%"
            }), use_container_width=True)

        with tab_over:
            over_df = eval_df.sort_values(by="Residual_USD", ascending=True).head(5)
            st.dataframe(over_df.style.format({
                "Actual_Price": "${:,.0f}",
                "Predicted_Price": "${:,.0f}",
                "Residual_USD": "${:,.0f}",
                "Abs_Error_USD": "${:,.0f}",
                "Pct_Error": "{:.1f}%"
            }), use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Diagnostics error: {e}")
