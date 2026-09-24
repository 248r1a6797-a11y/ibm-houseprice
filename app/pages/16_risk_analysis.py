"""Page 16 — Sliced Risk & Uncertainty Analysis."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

from app.components.header import render_header
from app.components.kpi_cards import render_kpi_card

def render_page():
    render_header(
        title="Model Risk & Sliced Uncertainty Analysis",
        subtitle="Forensic examination of sliced subgroup performance, heteroscedasticity across price tiers, and boundary failure modes.",
        badge="RISK GOVERNANCE"
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
                    st.success("Model trained and evaluated! Loading sliced risk analysis...")
                    st.rerun()
                else:
                    st.error("Could not auto-train model. Please check target variable in 1. Data Workspace.")
        return

    # Compute holdout predictions and error metrics
    try:
        preds = active_model.predict(X_test)
        actuals = y_test.values
        residuals = actuals - preds
        pct_errors = np.abs(residuals / np.maximum(actuals, 1.0)) * 100.0

        eval_df = X_test.copy()
        eval_df["Actual_Price"] = actuals
        eval_df["Predicted_Price"] = preds
        eval_df["Residual_USD"] = residuals
        eval_df["Abs_Error_USD"] = np.abs(residuals)
        eval_df["Pct_Error"] = pct_errors

        # Global KPIs
        global_mae = float(np.mean(eval_df["Abs_Error_USD"]))
        global_rmse = float(np.sqrt(np.mean(residuals ** 2)))
        global_mape = float(np.mean(eval_df["Pct_Error"]))
        global_r2 = float(1.0 - (np.sum(residuals**2) / np.sum((actuals - np.mean(actuals))**2)))

        st.markdown("#### 🎯 Global Generalization Benchmarks")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Global Holdout MAE", f"${global_mae:,.0f}")
        k2.metric("Global Holdout RMSE", f"${global_rmse:,.0f}")
        k3.metric("Global Holdout MAPE", f"{global_mape:.2f}%")
        k4.metric("Holdout Explained Variance", f"{global_r2:.3f}")

        tab1, tab2 = st.tabs(["💰 Slicing by Price Tier (Quartiles)", "🏷️ Dynamic Slicing by Cohort / Attribute"])

        # =====================================================================
        # TAB 1: PRICE TIER SLICING (Empirical Quartiles)
        # =====================================================================
        with tab1:
            st.markdown("<div class='card-section'>", unsafe_allow_html=True)
            st.markdown("<div class='card-title'>Sliced Performance Across Empirical Price Tiers</div>", unsafe_allow_html=True)

            q25 = float(np.percentile(actuals, 25))
            q50 = float(np.percentile(actuals, 50))
            q75 = float(np.percentile(actuals, 75))

            def assign_quartile_tier(price):
                if price <= q25:
                    return f"1. Budget (≤ ${q25:,.0f})"
                elif price <= q50:
                    return f"2. Mid-Range (${q25:,.0f}–${q50:,.0f})"
                elif price <= q75:
                    return f"3. Upper-Mid (${q50:,.0f}–${q75:,.0f})"
                else:
                    return f"4. Luxury (> ${q75:,.0f})"

            eval_df["Price_Tier"] = eval_df["Actual_Price"].apply(assign_quartile_tier)

            tier_metrics = eval_df.groupby("Price_Tier", observed=False).agg(
                Count=("Actual_Price", "count"),
                Median_Actual=("Actual_Price", "median"),
                Median_Pred=("Predicted_Price", "median"),
                MAE=("Abs_Error_USD", "mean"),
                RMSE=("Residual_USD", lambda r: np.sqrt(np.mean(r**2))),
                MAPE=("Pct_Error", "mean"),
                Mean_Bias=("Residual_USD", "mean")
            ).reset_index()

            tier_metrics["Uncertainty_Flag"] = tier_metrics["MAPE"].apply(
                lambda m: "⚠️ Elevated Error (>10%)" if m > 10.0 else "✅ Calibrated (<10%)"
            )

            t_col1, t_col2 = st.columns([1.1, 0.9])
            with t_col1:
                st.dataframe(
                    tier_metrics.style.format({
                        "Median_Actual": "${:,.0f}",
                        "Median_Pred": "${:,.0f}",
                        "MAE": "${:,.0f}",
                        "RMSE": "${:,.0f}",
                        "MAPE": "{:.2f}%",
                        "Mean_Bias": "${:+,.0f}",
                        "Count": "{:,}"
                    }),
                    use_container_width=True
                )
            with t_col2:
                fig_tier = px.bar(
                    tier_metrics,
                    x="Price_Tier",
                    y="MAPE",
                    color="MAPE",
                    color_continuous_scale="Reds",
                    text=tier_metrics["MAPE"].apply(lambda v: f"{v:.1f}%"),
                    labels={"MAPE": "Mean Absolute % Error (MAPE)", "Price_Tier": "Price Tier"}
                )
                fig_tier.update_layout(margin=dict(l=10, r=10, t=20, b=20), height=280, coloraxis_showscale=False)
                st.plotly_chart(fig_tier, use_container_width=True)

            st.markdown("</div>", unsafe_allow_html=True)

        # =====================================================================
        # TAB 2: DYNAMIC COHORT SLICING
        # =====================================================================
        with tab2:
            st.markdown("<div class='card-section'>", unsafe_allow_html=True)
            st.markdown("<div class='card-title'>Sliced Performance by Feature Cohort</div>", unsafe_allow_html=True)

            candidate_slice_cols = [c for c in X_test.columns if X_test[c].nunique() <= 30]
            if not candidate_slice_cols:
                candidate_slice_cols = list(X_test.columns)

            slice_col = st.selectbox("Select Feature to Slice Test Performance by:", candidate_slice_cols)

            if slice_col:
                cohort_metrics = eval_df.groupby(slice_col, observed=False).agg(
                    Count=("Actual_Price", "count"),
                    Median_Actual=("Actual_Price", "median"),
                    MAE=("Abs_Error_USD", "mean"),
                    RMSE=("Residual_USD", lambda r: np.sqrt(np.mean(r**2))),
                    MAPE=("Pct_Error", "mean"),
                    Mean_Bias=("Residual_USD", "mean")
                ).reset_index().sort_values(by="MAE", ascending=False)

                c_col1, c_col2 = st.columns([1.1, 0.9])
                with c_col1:
                    st.dataframe(
                        cohort_metrics.head(15).style.format({
                            "Median_Actual": "${:,.0f}",
                            "MAE": "${:,.0f}",
                            "RMSE": "${:,.0f}",
                            "MAPE": "{:.2f}%",
                            "Mean_Bias": "${:+,.0f}",
                            "Count": "{:,}"
                        }),
                        use_container_width=True
                    )
                with c_col2:
                    fig_cohort = px.bar(
                        cohort_metrics.head(10),
                        x=slice_col,
                        y="MAE",
                        color="MAE",
                        color_continuous_scale="Blues",
                        text=cohort_metrics.head(10)["MAE"].apply(lambda v: f"${v:,.0f}"),
                        labels={"MAE": "Mean Absolute Error ($)", slice_col: slice_col}
                    )
                    fig_cohort.update_layout(margin=dict(l=10, r=10, t=20, b=20), height=280, coloraxis_showscale=False)
                    st.plotly_chart(fig_cohort, use_container_width=True)

            st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error in risk analysis: {e}")
