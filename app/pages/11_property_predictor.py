"""Page 11 — Dynamic Property Predictor & Valuation Studio."""
import streamlit as st
import pandas as pd
import numpy as np

from app.components.header import render_header
from src.inference.predictor import load_production_model, predict_single_property, predict_batch_dataframe
from src.data.schema import compute_feature_distributions

def render_page():
    render_header(
        title="Dynamic Property Valuation Predictor",
        subtitle="Generate calibrated single-property valuations, prediction intervals, and local value drivers dynamically synthesized from the active model schema.",
        badge="INFERENCE LAYER"
    )

    # 1. Resolve Active Model
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
                    st.success("Baseline model trained! Launching property predictor...")
                    st.rerun()
                else:
                    st.error("Could not auto-train model. Please check target variable in 1. Data Workspace.")
        return

    # 2. Determine Feature Schema from Model or Session
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
        st.warning("No features detected for active model. Please configure feature selection in 4. Feature Engineering.")
        return

    # 3. Retrieve or compute feature distributions
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

    tab_single, tab_batch = st.tabs(["🏡 Single Property Valuation", "📁 Batch File Scoring"])

    # =========================================================================
    # TAB 1: DYNAMIC SINGLE PROPERTY VALUATION
    # =========================================================================
    with tab_single:
        st.markdown("<div class='card-section'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>🏡 Property Specification Input Form</div>", unsafe_allow_html=True)
        st.caption(f"Dynamically generated input controls for all **{len(active_features)}** active model features:")

        user_inputs = {}
        cols_per_row = 4

        with st.form("dynamic_property_valuation_form"):
            for i in range(0, len(active_features), cols_per_row):
                row_feats = active_features[i:i+cols_per_row]
                ui_cols = st.columns(cols_per_row)

                for j, f_name in enumerate(row_feats):
                    with ui_cols[j]:
                        dist = distributions.get(f_name, {})
                        f_type = dist.get("type", "numeric")

                        if f_type == "numeric":
                            c_min = dist.get("min", 0.0)
                            c_max = dist.get("max", 100000.0)
                            c_def = dist.get("default", (c_min + c_max) / 2.0)
                            c_step = dist.get("step", 1.0)
                            is_int = dist.get("is_integer", False)

                            # Handle edge case where min == max
                            if c_min == c_max:
                                c_max = c_min + 100.0

                            if is_int:
                                val = st.number_input(
                                    f"{f_name}:",
                                    min_value=int(c_min),
                                    max_value=int(c_max),
                                    value=int(c_def),
                                    step=int(max(1, c_step)),
                                    key=f"input_{f_name}"
                                )
                            else:
                                val = st.number_input(
                                    f"{f_name}:",
                                    min_value=float(c_min),
                                    max_value=float(c_max),
                                    value=float(c_def),
                                    step=float(c_step),
                                    key=f"input_{f_name}"
                                )
                            user_inputs[f_name] = val
                        else:
                            # Categorical
                            cats = dist.get("categories", ["Standard", "Other"])
                            default_cat = dist.get("default", cats[0])
                            def_idx = cats.index(default_cat) if default_cat in cats else 0
                            val = st.selectbox(
                                f"{f_name}:",
                                cats,
                                index=def_idx,
                                key=f"input_{f_name}"
                            )
                            user_inputs[f_name] = val

            predict_btn = st.form_submit_button("💰 GENERATE VALUATION ESTIMATE", type="primary")

        if predict_btn:
            # Build medians for relative comparison
            dataset_medians = {}
            for fn in active_features:
                if fn in distributions and distributions[fn].get("type") == "numeric":
                    dataset_medians[fn] = distributions[fn].get("median", 0.0)

            try:
                pred_result = predict_single_property(
                    input_data=user_inputs,
                    dataset_medians=dataset_medians,
                    model=active_model,
                    metadata=metadata,
                    expected_features=active_features
                )

                st.markdown("---")
                
                # Valuation Headline Card
                p_val = pred_result["predicted_price"]
                p_lower = pred_result["lower_bound"]
                p_upper = pred_result["upper_bound"]

                result_html = (
                    '<div style="background: #1E3A8A; color: white; padding: 24px; border-radius: 8px; margin: 15px 0 25px 0;">'
                    '<div style="font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.08em; opacity: 0.85;">ESTIMATED MARKET VALUATION</div>'
                    f'<div style="font-size: 3rem; font-weight: 800; margin: 4px 0;">${p_val:,.0f}</div>'
                    f'<div style="font-size: 1.05rem; opacity: 0.95;">Calibrated 80% Prediction Interval: <b>${p_lower:,.0f} – ${p_upper:,.0f}</b></div>'
                    f'<div style="font-size: 0.8rem; color: #BFDBFE; margin-top: 8px;">Active Model: <b>{pred_result.get("model_name", "Champion")}</b> | Evaluated Features: <b>{len(active_features)}</b></div>'
                    '</div>'
                )
                if hasattr(st, "html"):
                    st.html(result_html)
                else:
                    st.markdown(result_html, unsafe_allow_html=True)

                # Local Value Drivers
                pos = pred_result.get("positive_drivers", [])
                neg = pred_result.get("negative_drivers", [])
                comps = pred_result.get("comparisons", [])

                c_drv1, c_drv2 = st.columns(2)
                with c_drv1:
                    st.markdown("##### 🟢 Top Positive Value Contributors")
                    if pos:
                        for p in pos:
                            f_name = p.get("clean_feature", p.get("feature", "Feature"))
                            c_val = p.get("dollar_contribution", p.get("contribution", 0.0))
                            st.markdown(f"- **`{f_name}`**: <span style='color: #16A34A; font-weight: 700;'>+${abs(c_val):,.0f}</span>", unsafe_allow_html=True)
                    else:
                        st.caption("No strong positive contributors.")

                with c_drv2:
                    st.markdown("##### 🔴 Top Downward Price Penalties")
                    if neg:
                        for n in neg:
                            f_name = n.get("clean_feature", n.get("feature", "Feature"))
                            c_val = n.get("dollar_contribution", n.get("contribution", 0.0))
                            st.markdown(f"- **`{f_name}`**: <span style='color: #DC2626; font-weight: 700;'>-${abs(c_val):,.0f}</span>", unsafe_allow_html=True)
                    else:
                        st.caption("No strong downward penalties.")

                if comps:
                    st.markdown("##### 📐 Profile vs Market Benchmark")
                    for c_text in comps:
                        st.markdown(f"- {c_text}")

            except Exception as e:
                st.error(f"Prediction failed: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

    # =========================================================================
    # TAB 2: BATCH FILE SCORING
    # =========================================================================
    with tab_batch:
        st.markdown("<div class='card-section'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>📁 Batch Property File Scoring & Export</div>", unsafe_allow_html=True)
        st.caption("Upload a CSV file containing multiple properties to generate batch valuation estimates and prediction bounds:")

        batch_file = st.file_uploader("Upload Batch CSV for Scoring:", type=["csv"], key="batch_uploader")
        if batch_file is not None:
            try:
                b_df = pd.read_csv(batch_file)
                st.write(f"Loaded batch file with **{len(b_df):,}** properties.")
                
                if st.button("🚀 Score Entire Batch", type="primary"):
                    with st.spinner("Scoring batch properties..."):
                        scored_df = predict_batch_dataframe(b_df, model=active_model, metadata=metadata)
                    
                    st.success(f"Successfully scored {len(scored_df):,} properties!")
                    st.dataframe(scored_df.head(15), use_container_width=True)
                    
                    csv_export = scored_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download Scored CSV with Prediction Bounds",
                        data=csv_export,
                        file_name="scored_properties_with_bounds.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            except Exception as e:
                st.error(f"Batch scoring error: {e}")

        st.markdown("</div>", unsafe_allow_html=True)
