"""Page 5 — Model Explainability."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from app.components.header import render_header

def render_page(df: pd.DataFrame):
    render_header(
        title="Explainable AI & Feature Attribution",
        subtitle="Global and feature-level explanations powered by Shapley Additive Explanations (SHAP) and permutation importance.",
        badge="EXPLAINABLE AI"
    )

    tab1, tab2 = st.tabs(["Global Feature Importance", "Interactive Feature Response Simulator"])

    with tab1:
        st.markdown("<div class='card-section'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Global SHAP Feature Attribution (Mean |SHAP Value|)</div>", unsafe_allow_html=True)
        try:
            shap_df = pd.read_csv("reports/tables/shap_feature_importance.csv")
            
            top_15 = shap_df.head(15)
            fig_shap = px.bar(
                top_15,
                x="mean_abs_shap",
                y="clean_feature",
                orientation="h",
                color="mean_abs_shap",
                color_continuous_scale="Blues",
                labels={"mean_abs_shap": "Mean |SHAP Value| (Impact on Model Output)", "clean_feature": "Feature Name"},
                text_auto=".3f"
            )
            fig_shap.update_layout(yaxis=dict(autorange="reversed"), height=480, margin=dict(l=20, r=20, t=20, b=20), coloraxis_showscale=False)
            st.plotly_chart(fig_shap, use_container_width=True)

            with st.expander("View Full Feature Importance Table"):
                st.dataframe(shap_df, use_container_width=True)

        except Exception as e:
            st.info("SHAP importance values are currently being computed.")
        st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        st.markdown("<div class='card-section'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Model Feature Response Simulator</div>", unsafe_allow_html=True)
        st.markdown("Select a core numerical or categorical driver to inspect the empirical relationship observed in historical transactions:")

        sim_feature = st.selectbox(
            "Select Property Characteristic to Analyze:",
            ["OverallQual", "GrLivArea", "TotalSF", "HouseAgeAtSale", "YearBuilt", "GarageCars", "TotalBath"]
        )

        if sim_feature in df.columns:
            if df[sim_feature].nunique() <= 12:
                # Grouped bar / boxplot for discrete ratings
                fig_sim = px.box(
                    df,
                    x=sim_feature,
                    y="SalePrice",
                    color=sim_feature,
                    labels={sim_feature: sim_feature, "SalePrice": "Historical Sale Price ($)"}
                )
                fig_sim.update_layout(height=420, margin=dict(l=20, r=20, t=20, b=20), showlegend=False)
                st.plotly_chart(fig_sim, use_container_width=True)
            else:
                # Continuous scatter with trendline
                fig_sim = px.scatter(
                    df,
                    x=sim_feature,
                    y="SalePrice",
                    opacity=0.4,
                    trendline="lowess",
                    color_discrete_sequence=["#1E3A8A"],
                    labels={sim_feature: sim_feature, "SalePrice": "Historical Sale Price ($)"}
                )
                fig_sim.update_layout(height=420, margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig_sim, use_container_width=True)

        st.warning(
            "⚠️ **Methodological Boundary**: Feature importance and SHAP attributions represent predictive associations captured by tree ensembles within historical data. They do NOT establish legal or structural causality."
        )
        st.markdown("</div>", unsafe_allow_html=True)
