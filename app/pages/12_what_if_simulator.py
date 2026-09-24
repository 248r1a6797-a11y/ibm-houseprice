"""Page 12 — Dynamic What-If Scenario Simulator & Sensitivity Laboratory."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from app.components.header import render_header
from src.inference.predictor import load_production_model
from src.data.schema import compute_feature_distributions

def render_page():
    render_header(
        title="Interactive What-If Scenario Simulator",
        subtitle="Simulate capital renovations, floorplan expansions, and material upgrades side-by-side against baseline property state.",
        badge="SCENARIO LAB"
    )

    active_model = st.session_state.get("champion_model")
    metadata = st.session_state.get("champion_metadata", {})

    if active_model is None:
        is_ames = "AmesHousing" in st.session_state.get("dataset_name", "")
        if is_ames:
            try:
                active_model, default_meta = load_production_model()
                metadata = metadata or default_meta
                st.session_state.champion_model = active_model
                st.session_state.champion_metadata = metadata
            except Exception:
                pass

    if active_model is None:
        st.info(f"ℹ️ No active model trained for `{st.session_state.get('dataset_name', 'current dataset')}`.")
        from app.state import quick_train_baseline_model
        if st.button("🚀 Auto-Train Baseline Model Now", type="primary"):
            with st.spinner("Training baseline model pipeline..."):
                if quick_train_baseline_model(st):
                    st.success("Baseline model trained! Launching scenario simulator...")
                    st.rerun()
                else:
                    st.error("Could not auto-train model. Please check target variable in 1. Data Workspace.")
        return

    # Determine Active Features
    active_features = None
    if hasattr(active_model, "feature_names_in_"):
        active_features = list(active_model.feature_names_in_)
    elif hasattr(active_model, "named_steps") and hasattr(active_model.named_steps.get("preprocessor"), "feature_names_in_"):
        active_features = list(active_model.named_steps["preprocessor"].feature_names_in_)
    elif hasattr(active_model, "regressor_") and hasattr(active_model.regressor_, "named_steps") and hasattr(active_model.regressor_.named_steps.get("preprocessor"), "feature_names_in_"):
        active_features = list(active_model.regressor_.named_steps["preprocessor"].feature_names_in_)

    if not active_features:
        active_features = st.session_state.get("selected_features", [])

    if not active_features:
        st.warning("No features configured for simulation.")
        return

    # Distributions
    ctx = st.session_state.get("dataset_context")
    df_src = st.session_state.feature_df if not st.session_state.feature_df.empty else (
        st.session_state.clean_df if not st.session_state.clean_df.empty else st.session_state.raw_df
    )
    distributions = {}
    if ctx and ctx.feature_distributions:
        distributions.update(ctx.feature_distributions)
    if not df_src.empty:
        missing_feats = [f for f in active_features if f not in distributions]
        if missing_feats:
            distributions.update(compute_feature_distributions(df_src, missing_feats))
    if not distributions and not df_src.empty:
        distributions = compute_feature_distributions(df_src, active_features)

    # Pick up to 4 primary numerical features for side-by-side sliders
    num_features = [f for f in active_features if distributions.get(f, {}).get("type") == "numeric"]
    if not num_features:
        st.info("No numerical features available for continuous slider simulation.")
        return

    tune_features = num_features[:4]

    st.markdown("<div class='card-section'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>⚖️ Side-by-Side Property Feature Customization</div>", unsafe_allow_html=True)
    st.caption("Adjust target features simultaneously to observe marginal valuation impacts in real time:")

    base_col, scen_col = st.columns(2)

    base_vals = {}
    scen_vals = {}

    with base_col:
        st.markdown("#### 🏛️ Base Property (Current State)")
        for fn in tune_features:
            dist = distributions[fn]
            c_min = dist.get("min", 0.0)
            c_max = dist.get("max", 1000.0)
            if c_min >= c_max:
                c_max = c_min + 10.0
            c_def = float(np.clip(dist.get("default", (c_min + c_max) / 2.0), c_min, c_max))
            c_step = max(0.01, dist.get("step", 1.0))
            is_int = dist.get("is_integer", False)

            if is_int:
                base_vals[fn] = st.slider(f"Base {fn}:", int(c_min), int(c_max), int(c_def), int(max(1, c_step)), key=f"b_{fn}")
            else:
                base_vals[fn] = st.slider(f"Base {fn}:", float(c_min), float(c_max), float(c_def), float(c_step), key=f"b_{fn}")

    with scen_col:
        st.markdown("#### 🚀 Scenario Property (Proposed Upgrades)")
        for fn in tune_features:
            dist = distributions[fn]
            c_min = dist.get("min", 0.0)
            c_max = dist.get("max", 1000.0)
            if c_min >= c_max:
                c_max = c_min + 10.0
            c_def = float(np.clip(dist.get("default", (c_min + c_max) / 2.0), c_min, c_max))
            c_step = max(0.01, dist.get("step", 1.0))
            is_int = dist.get("is_integer", False)

            if is_int:
                scen_vals[fn] = st.slider(f"Scenario {fn}:", int(c_min), int(c_max), int(c_def), int(max(1, c_step)), key=f"s_{fn}")
            else:
                scen_vals[fn] = st.slider(f"Scenario {fn}:", float(c_min), float(c_max), float(c_def), float(c_step), key=f"s_{fn}")

    st.markdown("---")

    # Construct complete records for model
    def build_full_record(tuned_dict):
        rec = {}
        for fn in active_features:
            if fn in tuned_dict:
                rec[fn] = tuned_dict[fn]
            else:
                dist = distributions.get(fn, {})
                rec[fn] = dist.get("default", 0.0 if dist.get("type") == "numeric" else "Standard")
        return pd.DataFrame([rec])

    base_df = build_full_record(base_vals)
    scen_df = build_full_record(scen_vals)

    try:
        base_pred = float(active_model.predict(base_df)[0])
        scen_pred = float(active_model.predict(scen_df)[0])
        dollar_diff = scen_pred - base_pred
        pct_diff = (dollar_diff / max(1.0, base_pred)) * 100.0

        # KPI Outcome Row
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Base Property Valuation", f"${base_pred:,.0f}")
        k2.metric("Scenario Property Valuation", f"${scen_pred:,.0f}")
        k3.metric("Valuation Difference (Δ)", f"{dollar_diff:+,.0f}", delta=f"{pct_diff:+.1f}%")
        roi_eval = "Positive Equity Expansion" if dollar_diff > 0 else "Valuation Neutral / Discount"
        k4.metric("Scenario Assessment", roi_eval)

        # Bar chart comparison
        fig_bar = go.Figure(data=[
            go.Bar(name="Base Valuation", x=["Valuation Comparison"], y=[base_pred], marker_color="#64748B"),
            go.Bar(name="Scenario Valuation", x=["Valuation Comparison"], y=[scen_pred], marker_color="#1E3A8A")
        ])
        fig_bar.update_layout(barmode="group", height=320, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_bar, use_container_width=True)

    except Exception as e:
        st.error(f"Scenario prediction error: {e}")

    st.markdown("</div>", unsafe_allow_html=True)
