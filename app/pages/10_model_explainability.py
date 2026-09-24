"""Page 10 — Model Explainability & SHAP Game-Theoretic Attribution."""
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from app.components.header import render_header
from src.inference.predictor import load_production_model
from src.explainability.shap_analysis import compute_global_shap, explain_single_prediction

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

def render_page():
    render_header(
        title="Explainable AI (XAI) & SHAP Game-Theoretic Attribution",
        subtitle="Unmask model decisions: global feature importance rankings and single-property financial waterfall drivers.",
        badge="EXPLAINABILITY LAYER"
    )

    active_model = st.session_state.get("champion_model")
    if active_model is None:
        is_ames = "AmesHousing" in st.session_state.get("dataset_name", "")
        if is_ames:
            try:
                active_model, _ = load_production_model()
                st.session_state.champion_model = active_model
            except Exception:
                pass

    if active_model is None:
        st.info(f"ℹ️ No active model trained for `{st.session_state.get('dataset_name', 'current dataset')}`.")
        from app.state import quick_train_baseline_model
        if st.button("🚀 Auto-Train Baseline Model Now", type="primary"):
            with st.spinner("Training baseline model pipeline..."):
                if quick_train_baseline_model(st):
                    st.success("Model trained! Generating feature attributions...")
                    st.rerun()
                else:
                    st.error("Could not auto-train model. Please check target variable in 1. Data Workspace.")
        return

    X_test = st.session_state.get("X_test")
    if X_test is None or X_test.empty:
        # Fallback to feature_df -> clean_df -> raw_df
        df_src = st.session_state.feature_df if not st.session_state.feature_df.empty else (
            st.session_state.clean_df if not st.session_state.clean_df.empty else st.session_state.raw_df
        )
        target_col = st.session_state.target_col
        if not df_src.empty:
            sel = st.session_state.get("selected_features", [c for c in df_src.columns if c != target_col])
            X_test = df_src[[c for c in sel if c in df_src.columns]].copy()
        else:
            st.warning("No test data available for SHAP evaluation.")
            return

    glob_tab, indiv_tab = st.tabs(["1. Global Feature Attribution", "2. Single-Property Waterfall Explanation"])

    # =========================================================================
    # TAB 1: GLOBAL SHAP
    # =========================================================================
    with glob_tab:
        st.markdown("<div class='card-section'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>🌐 Global Feature Importance (TreeSHAP / Attribution)</div>", unsafe_allow_html=True)
        st.caption("Quantifies the mean absolute game-theoretic marginal contribution across evaluated properties:")

        with st.spinner("Computing global feature attributions..."):
            shap_result = compute_global_shap(active_model, X_test, sample_size=100)

        shap_df = shap_result.get("importance_df", pd.DataFrame())

        # Fallback to offline CSV if compute_global_shap returns empty
        if shap_df.empty:
            shap_csv = ROOT_DIR / "reports/tables/shap_feature_importance.csv"
            if shap_csv.exists():
                shap_df = pd.read_csv(shap_csv)

        if not shap_df.empty:
            top_shap = shap_df.head(15).copy()
            feat_col = "clean_feature" if "clean_feature" in top_shap.columns else "feature"
            
            fig_bar = px.bar(
                top_shap,
                x="mean_abs_shap",
                y=feat_col,
                orientation="h",
                color="mean_abs_shap",
                color_continuous_scale="Blues",
                labels={"mean_abs_shap": "Mean |Attribution Value| (Marginal Impact)", feat_col: "Property Feature"}
            )
            fig_bar.update_layout(yaxis=dict(autorange="reversed"), height=420, margin=dict(l=20, r=20, t=20, b=20), coloraxis_showscale=False)
            st.plotly_chart(fig_bar, use_container_width=True)

            st.dataframe(top_shap[[feat_col, "mean_abs_shap"]], use_container_width=True)
        else:
            st.info("Feature importance is currently calculating. Please execute candidate training in 7. Model Laboratory.")

        st.markdown("</div>", unsafe_allow_html=True)

    # =========================================================================
    # TAB 2: INDIVIDUAL WATERFALL
    # =========================================================================
    with indiv_tab:
        st.markdown("<div class='card-section'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>🔍 Individual Property Waterfall Breakdown</div>", unsafe_allow_html=True)
        st.caption("Select any property from the holdout set to explain why the model arrived at its predicted valuation:")

        df_eval = X_test
        if not df_eval.empty:
            prop_indices = list(df_eval.index[:30])
            selected_idx = st.selectbox(
                "Select Property Observation (Row Index):",
                prop_indices,
                format_func=lambda x: f"Property Observation #{x}"
            )

            single_row = df_eval.loc[[selected_idx]]
            
            try:
                # Raw prediction
                raw_pred = active_model.predict(single_row)
                pred_price = float(raw_pred[0])
                
                # Check actual price if available
                y_test = st.session_state.get("y_test")
                actual_price = None
                if y_test is not None and selected_idx in y_test.index:
                    actual_price = float(y_test.loc[selected_idx])

                w1, w2, w3 = st.columns(3)
                w1.metric("Predicted Valuation", f"${pred_price:,.0f}")
                if actual_price is not None:
                    w2.metric("Actual Target Value", f"${actual_price:,.0f}")
                    w3.metric("Valuation Residual", f"${actual_price - pred_price:+,.0f}")
                else:
                    w2.metric("Dataset Median Target", f"${st.session_state.raw_df[st.session_state.target_col].median():,.0f}")
                    w3.metric("Model Architecture", st.session_state.get("champion_name", "Champion"))

                # Compute single explanation
                expl = explain_single_prediction(active_model, single_row, top_k=5)
                
                pos = expl.get("positive_drivers", [])
                neg = expl.get("negative_drivers", [])

                c_pos, c_neg = st.columns(2)
                with c_pos:
                    st.markdown("##### 🟢 Top Positive Value Contributors")
                    if pos:
                        for p in pos:
                            f_name = p.get("clean_feature", p.get("feature", "Feature"))
                            c_val = p.get("dollar_contribution", p.get("contribution", 0.0))
                            st.markdown(f"- **`{f_name}`**: <span style='color: #16A34A; font-weight: 700;'>+${abs(c_val):,.0f}</span>", unsafe_allow_html=True)
                    else:
                        st.caption("No strong positive contributors.")

                with c_neg:
                    st.markdown("##### 🔴 Top Downward Price Penalties")
                    if neg:
                        for n in neg:
                            f_name = n.get("clean_feature", n.get("feature", "Feature"))
                            c_val = n.get("dollar_contribution", n.get("contribution", 0.0))
                            st.markdown(f"- **`{f_name}`**: <span style='color: #DC2626; font-weight: 700;'>-${abs(c_val):,.0f}</span>", unsafe_allow_html=True)
                    else:
                        st.caption("No strong downward penalties.")

            except Exception as e:
                st.error(f"Error computing local SHAP waterfall: {e}")

        st.markdown("</div>", unsafe_allow_html=True)
