"""Page 9 — Machine Learning Experiment Tracker & History Comparison."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from app.components.header import render_header

def render_page():
    render_header(
        title="Machine Learning Experiment Tracker & Audit Trail",
        subtitle="Versioned run-by-run audit trail of model iterations, feature sets, hyperparameters, and cross-validation metrics.",
        badge="AUDIT LAYER"
    )

    history = st.session_state.get("experiment_history", [])

    if not history:
        st.info("No experiment runs recorded in this session. Visit 7. Model Laboratory or 8. Hyperparameter Tuning to execute runs.")
        return

    hist_df = pd.DataFrame(history)

    st.markdown("<div class='card-section'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>📜 Session Experiment Run Log</div>", unsafe_allow_html=True)

    def fmt_num(v):
        if pd.isna(v) or v is None: return "N/A"
        try:
            val = float(v)
            return f"${val:,.0f}" if val >= 1000 else f"{val:,.3f}"
        except Exception:
            return str(v)

    formatters = {}
    if "cv_rmse" in hist_df.columns: formatters["cv_rmse"] = fmt_num
    if "test_rmse" in hist_df.columns: formatters["test_rmse"] = fmt_num
    if "cv_r2" in hist_df.columns: formatters["cv_r2"] = "{:.3f}"
    if "test_r2" in hist_df.columns: formatters["test_r2"] = "{:.3f}"

    st.dataframe(
        hist_df.style.format(formatters),
        use_container_width=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Visual Iteration Progress
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='card-section'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Holdout Test RMSE ($) Across Experiments</div>", unsafe_allow_html=True)
        fig_prog = px.line(
            hist_df,
            x="experiment_id",
            y="test_rmse",
            markers=True,
            color="model",
            labels={"experiment_id": "Experiment Run", "test_rmse": "Test RMSE ($) [Lower is Better]"}
        )
        fig_prog.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_prog, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card-section'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Explained Variance (Test R²) Across Experiments</div>", unsafe_allow_html=True)
        fig_r2 = px.bar(
            hist_df,
            x="experiment_id",
            y="test_r2",
            color="test_r2",
            color_continuous_scale="Blues",
            labels={"experiment_id": "Experiment Run", "test_r2": "Test R² [Higher is Better]"}
        )
        fig_r2.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20), coloraxis_showscale=False)
        st.plotly_chart(fig_r2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Side-by-Side Comparator
    st.markdown("<div class='card-section'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>⚖️ Side-by-Side Experiment Comparator</div>", unsafe_allow_html=True)
    
    exp_ids = hist_df["experiment_id"].tolist()
    if len(exp_ids) >= 2:
        cmp1, cmp2 = st.columns(2)
        with cmp1:
            sel_exp1 = st.selectbox("Select Baseline Experiment:", exp_ids, index=0)
        with cmp2:
            sel_exp2 = st.selectbox("Select Comparison Experiment:", exp_ids, index=len(exp_ids)-1)

        row1 = hist_df[hist_df["experiment_id"] == sel_exp1].iloc[0]
        row2 = hist_df[hist_df["experiment_id"] == sel_exp2].iloc[0]

        delta_rmse = row2["test_rmse"] - row1["test_rmse"]
        delta_r2 = row2["test_r2"] - row1["test_r2"]

        d1, d2, d3 = st.columns(3)
        d1.metric(f"RMSE Delta ({sel_exp2} vs {sel_exp1})", f"${row2['test_rmse']:,.0f}", delta=f"{delta_rmse:+,.0f} USD", delta_color="inverse")
        d2.metric("R² Delta", f"{row2['test_r2']:.3f}", delta=f"{delta_r2:+.3f}")
        d3.metric("Model Architecture", f"{row2['model']}", f"vs {row1['model']}")
    else:
        st.info("Execute at least 2 experiment runs to enable the comparative differential view.")

    st.markdown("</div>", unsafe_allow_html=True)
