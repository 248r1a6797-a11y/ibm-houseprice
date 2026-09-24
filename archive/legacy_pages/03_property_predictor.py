"""Page 3 — Property Price Predictor & Batch Valuation."""
import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from app.components.header import render_header
from src.inference.predictor import predict_single_property, predict_batch_dataframe

# Pre-configured demo property presets based on authentic dataset rows
PRESETS = {
    "Select a preset or customize below...": None,
    "1. Typical Suburban Home (CollgCr, 1650 sq ft, Qual 7)": {
        "GrLivArea": 1650, "LotArea": 9500, "OverallQual": 7, "OverallCond": 5,
        "YearBuilt": 2003, "YearRemodAdd": 2004, "TotalBsmtSF": 900, "1stFlrSF": 900,
        "2ndFlrSF": 750, "FullBath": 2, "HalfBath": 1, "BsmtFullBath": 1, "BsmtHalfBath": 0,
        "BedroomAbvGr": 3, "KitchenAbvGr": 1, "TotRmsAbvGrd": 7, "Fireplaces": 1,
        "GarageCars": 2, "GarageArea": 550, "WoodDeckSF": 150, "OpenPorchSF": 60,
        "EnclosedPorch": 0, "3SsnPorch": 0, "ScreenPorch": 0, "PoolArea": 0,
        "Neighborhood": "CollgCr", "BldgType": "1Fam", "HouseStyle": "2Story",
        "MSZoning": "RL", "ExterQual": "Gd", "KitchenQual": "Gd", "Foundation": "PConc",
        "CentralAir": "Y", "YrSold": 2009, "MoSold": 6
    },
    "2. High-Value Custom Home (NridgHt, 2800 sq ft, Qual 9)": {
        "GrLivArea": 2800, "LotArea": 14000, "OverallQual": 9, "OverallCond": 5,
        "YearBuilt": 2006, "YearRemodAdd": 2007, "TotalBsmtSF": 1800, "1stFlrSF": 1800,
        "2ndFlrSF": 1000, "FullBath": 3, "HalfBath": 1, "BsmtFullBath": 1, "BsmtHalfBath": 0,
        "BedroomAbvGr": 4, "KitchenAbvGr": 1, "TotRmsAbvGrd": 10, "Fireplaces": 2,
        "GarageCars": 3, "GarageArea": 850, "WoodDeckSF": 250, "OpenPorchSF": 120,
        "EnclosedPorch": 0, "3SsnPorch": 0, "ScreenPorch": 0, "PoolArea": 0,
        "Neighborhood": "NridgHt", "BldgType": "1Fam", "HouseStyle": "2Story",
        "MSZoning": "RL", "ExterQual": "Ex", "KitchenQual": "Ex", "Foundation": "PConc",
        "CentralAir": "Y", "YrSold": 2009, "MoSold": 7
    },
    "3. Starter Home / Fixer-Upper (OldTown, 1050 sq ft, Qual 5)": {
        "GrLivArea": 1050, "LotArea": 6000, "OverallQual": 5, "OverallCond": 6,
        "YearBuilt": 1945, "YearRemodAdd": 1950, "TotalBsmtSF": 750, "1stFlrSF": 1050,
        "2ndFlrSF": 0, "FullBath": 1, "HalfBath": 0, "BsmtFullBath": 0, "BsmtHalfBath": 0,
        "BedroomAbvGr": 2, "KitchenAbvGr": 1, "TotRmsAbvGrd": 5, "Fireplaces": 0,
        "GarageCars": 1, "GarageArea": 280, "WoodDeckSF": 0, "OpenPorchSF": 20,
        "EnclosedPorch": 50, "3SsnPorch": 0, "ScreenPorch": 0, "PoolArea": 0,
        "Neighborhood": "OldTown", "BldgType": "1Fam", "HouseStyle": "1Story",
        "MSZoning": "RM", "ExterQual": "TA", "KitchenQual": "TA", "Foundation": "CBlock",
        "CentralAir": "N", "YrSold": 2008, "MoSold": 5
    }
}

def render_page(df: pd.DataFrame, metadata: dict):
    render_header(
        title="Interactive Property Valuation Predictor",
        subtitle="Generate estimated market prices with 80% calibrated prediction intervals and local SHAP factor attributions.",
        badge="INFERENCE ENGINE"
    )

    pred_tab, batch_tab = st.tabs(["Single Property Valuation", "Bulk CSV Batch Scoring"])

    with pred_tab:
        # Preset selector
        selected_preset = st.selectbox("Quick Load a Realistic Benchmark Property:", list(PRESETS.keys()))
        preset_vals = PRESETS[selected_preset] if PRESETS[selected_preset] else PRESETS["1. Typical Suburban Home (CollgCr, 1650 sq ft, Qual 7)"]

        with st.form("property_input_form"):
            st.markdown("#### 1. Property Basics & Living Space")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                gr_liv = st.number_input("Above-Grade Living Area (sq ft):", min_value=300, max_value=8000, value=int(preset_vals["GrLivArea"]), step=50)
            with col2:
                lot_area = st.number_input("Lot Size (sq ft):", min_value=1000, max_value=100000, value=int(preset_vals["LotArea"]), step=500)
            with col3:
                bedrooms = st.number_input("Bedrooms (Above Grade):", min_value=1, max_value=8, value=int(preset_vals["BedroomAbvGr"]))
            with col4:
                full_bath = st.number_input("Full Bathrooms:", min_value=1, max_value=5, value=int(preset_vals["FullBath"]))

            st.markdown("#### 2. Construction Quality & Condition")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                overall_qual = st.slider("Overall Quality (1=Poor, 10=Very Excellent):", 1, 10, int(preset_vals["OverallQual"]))
            with col2:
                overall_cond = st.slider("Overall Condition (1=Poor, 9=Excellent):", 1, 9, int(preset_vals["OverallCond"]))
            with col3:
                kitchen_qual = st.selectbox("Kitchen Quality:", ["Ex", "Gd", "TA", "Fa", "Po"], index=["Ex", "Gd", "TA", "Fa", "Po"].index(preset_vals["KitchenQual"]))
            with col4:
                exter_qual = st.selectbox("Exterior Quality:", ["Ex", "Gd", "TA", "Fa", "Po"], index=["Ex", "Gd", "TA", "Fa", "Po"].index(preset_vals["ExterQual"]))

            st.markdown("#### 3. Property Age & Renovation")
            col1, col2, col3 = st.columns(3)
            with col1:
                year_built = st.number_input("Year Built:", min_value=1870, max_value=2010, value=int(preset_vals["YearBuilt"]))
            with col2:
                year_remod = st.number_input("Year Remodeled / Addition:", min_value=1950, max_value=2010, value=int(preset_vals["YearRemodAdd"]))
            with col3:
                st.info("Tip: If the property was never remodeled, set Year Remodeled equal to Year Built.")

            st.markdown("#### 4. Basement, Garage & Amenities")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                total_bsmt = st.number_input("Total Basement Area (sq ft):", min_value=0, max_value=4000, value=int(preset_vals["TotalBsmtSF"]), step=50)
            with col2:
                garage_cars = st.number_input("Garage Capacity (Cars):", min_value=0, max_value=5, value=int(preset_vals["GarageCars"]))
            with col3:
                garage_area = st.number_input("Garage Area (sq ft):", min_value=0, max_value=1500, value=int(preset_vals["GarageArea"]), step=50)
            with col4:
                fireplaces = st.number_input("Fireplaces:", min_value=0, max_value=4, value=int(preset_vals["Fireplaces"]))

            st.markdown("#### 5. Location & Property Classification")
            col1, col2, col3, col4 = st.columns(4)
            all_neighs = sorted(df["Neighborhood"].unique().tolist()) if "Neighborhood" in df.columns else ["CollgCr", "NAmes", "NridgHt", "OldTown"]
            with col1:
                neighborhood = st.selectbox("Neighborhood:", all_neighs, index=all_neighs.index(preset_vals["Neighborhood"]) if preset_vals["Neighborhood"] in all_neighs else 0)
            with col2:
                bldg_type = st.selectbox("Dwelling Type:", ["1Fam", "2FmCon", "Duplex", "TwnhsE", "Twnhs"], index=0)
            with col3:
                house_style = st.selectbox("House Style:", ["1Story", "2Story", "1.5Fin", "SLvl", "SFoyer"], index=1)
            with col4:
                central_air = st.selectbox("Central Air Conditioning:", ["Y", "N"], index=0)

            submitted = st.form_submit_button("PREDICT ESTIMATED HOUSE PRICE", use_container_width=True)

        if submitted:
            # Construct input dictionary
            prop_data = {
                "GrLivArea": gr_liv, "LotArea": lot_area, "OverallQual": overall_qual, "OverallCond": overall_cond,
                "YearBuilt": year_built, "YearRemodAdd": year_remod, "TotalBsmtSF": total_bsmt,
                "1stFlrSF": gr_liv * 0.6, "2ndFlrSF": gr_liv * 0.4, "FullBath": full_bath, "HalfBath": 1,
                "BsmtFullBath": 1 if total_bsmt > 0 else 0, "BsmtHalfBath": 0, "BedroomAbvGr": bedrooms,
                "KitchenAbvGr": 1, "TotRmsAbvGrd": bedrooms + 3, "Fireplaces": fireplaces,
                "GarageCars": garage_cars, "GarageArea": garage_area, "WoodDeckSF": 100, "OpenPorchSF": 40,
                "EnclosedPorch": 0, "3SsnPorch": 0, "ScreenPorch": 0, "PoolArea": 0,
                "Neighborhood": neighborhood, "BldgType": bldg_type, "HouseStyle": house_style,
                "MSZoning": "RL", "ExterQual": exter_qual, "KitchenQual": kitchen_qual, "Foundation": "PConc",
                "CentralAir": central_air, "YrSold": 2009, "MoSold": 6
            }

            try:
                with st.spinner("Executing model valuation and SHAP attribution pipeline..."):
                    res = predict_single_property(prop_data, dataset_medians={"median_living_area": 1442.0})

                st.markdown(
                    f"""
                    <div class="prediction-box">
                        <div class="prediction-title">Estimated Market Valuation</div>
                        <div class="prediction-val">{res['formatted_price']}</div>
                        <div class="prediction-interval">80% Calibrated Prediction Interval: <b>{res['formatted_interval']}</b></div>
                        <div style="font-size: 0.8rem; margin-top: 8px; opacity: 0.85;">
                            Production Model: <b>{res['model_name']}</b> | Test MAE: <b>${res['test_mae']:,.0f}</b> | Valuation Scale: USD ($)
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Visualizing Drivers
                col_d1, col_d2 = st.columns([1.1, 0.9])
                with col_d1:
                    st.markdown("<div class='card-section'>", unsafe_allow_html=True)
                    st.markdown("<div class='card-title'>Top Valuation Drivers (SHAP Local Impact)</div>", unsafe_allow_html=True)
                    
                    # Combine positive and negative drivers into a single waterfall/bar
                    drivers = res["positive_drivers"] + res["negative_drivers"]
                    if drivers:
                        d_df = pd.DataFrame(drivers)
                        d_df["impact"] = d_df["contribution"]
                        d_df["color"] = np.where(d_df["impact"] > 0, "#16A34A", "#DC2626")
                        d_df = d_df.sort_values(by="impact", ascending=True)

                        fig_d = go.Figure(go.Bar(
                            x=d_df["impact"],
                            y=d_df["feature"],
                            orientation="h",
                            marker=dict(color=d_df["color"].tolist())
                        ))
                        fig_d.update_layout(
                            xaxis_title="SHAP Attribution (Relative Log Impact)",
                            yaxis_title="Feature",
                            margin=dict(l=20, r=20, t=20, b=20),
                            height=280
                        )
                        st.plotly_chart(fig_d, use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                with col_d2:
                    st.markdown("<div class='card-section'>", unsafe_allow_html=True)
                    st.markdown("<div class='card-title'>Market Baseline Profile Comparison</div>", unsafe_allow_html=True)
                    for comp in res["comparisons"]:
                        st.markdown(f"• {comp}")
                    st.markdown(
                        f"""
                        <div style="background: #F8FAFC; border: 1px solid #E2E8F0; padding: 12px; border-radius: 6px; margin-top: 14px; font-size: 0.85rem; color: #475569;">
                            <b>Interpretability Notice</b>: This prediction represents an estimated market value derived from 2,930 historical transactions in Ames, Iowa. Physical inspections and appraisal reviews remain essential.
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    st.markdown("</div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Prediction failed: {e}")

    with batch_tab:
        st.markdown("<div class='card-section'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Bulk Portfolio Scoring (CSV Upload)</div>", unsafe_allow_html=True)
        st.markdown("Upload a CSV file containing property characteristics. The production pipeline will validate schema, engineer features, generate estimated prices, calculate 80% confidence bands, and export the enriched dataset.")

        # Download Template CSV
        sample_template = df.head(5).drop(columns=["SalePrice"], errors="ignore")
        csv_buffer = io.StringIO()
        sample_template.to_csv(csv_buffer, index=False)
        st.download_button(
            label="Download Sample Batch Template CSV",
            data=csv_buffer.getvalue(),
            file_name="housewise_batch_template.csv",
            mime="text/csv"
        )

        uploaded_file = st.file_uploader("Upload CSV for Batch Prediction:", type=["csv"])
        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                st.info(f"Loaded batch file with {len(batch_df)} rows and {len(batch_df.columns)} columns.")
                if st.button("EXECUTE BATCH VALUATIONS", use_container_width=True):
                    with st.spinner("Scoring portfolio..."):
                        scored_df = predict_batch_dataframe(batch_df)
                    st.success(f"Successfully scored {len(scored_df)} properties!")
                    st.dataframe(scored_df[["predicted_price", "lower_bound", "upper_bound"] + [c for c in scored_df.columns if c not in ["predicted_price", "lower_bound", "upper_bound"]][:5]].head(10))

                    # Download scored CSV
                    scored_buffer = io.StringIO()
                    scored_df.to_csv(scored_buffer, index=False)
                    st.download_button(
                        label="Download Scored Predictions CSV",
                        data=scored_buffer.getvalue(),
                        file_name="housewise_batch_predictions.csv",
                        mime="text/csv"
                    )
            except Exception as e:
                st.error(f"Batch processing error: {e}")
        st.markdown("</div>", unsafe_allow_html=True)
