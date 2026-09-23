"""Page 2 — Market Analytics."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time
from app.components.header import render_header
from src.analytics.segmentation import segment_by_neighborhood, segment_by_price_tier
from src.analytics.sql_engine import SQLEngine

def render_page(df: pd.DataFrame):
    render_header(
        title="Market Analytics & Exploratory Intelligence",
        subtitle="Univariate, bivariate, correlation, and geographic segmentations across 2,930 historical transactions.",
        badge="EXPLORATORY DATA ANALYSIS"
    )

    # Interactive Sidebar Filters
    st.sidebar.markdown("### Market Filters")
    all_neighs = sorted(df["Neighborhood"].unique().tolist()) if "Neighborhood" in df.columns else []
    selected_neighs = st.sidebar.multiselect("Select Neighborhood(s):", all_neighs, default=all_neighs[:6])

    min_p, max_p = int(df["SalePrice"].min()), int(df["SalePrice"].max())
    price_range = st.sidebar.slider("Sale Price Range ($):", min_p, max_p, (min_p, max_p))

    qual_range = st.sidebar.slider("Overall Quality (1-10):", 1, 10, (1, 10))

    # Filtered dataframe
    filtered_df = df.copy()
    if selected_neighs:
        filtered_df = filtered_df[filtered_df["Neighborhood"].isin(selected_neighs)]
    filtered_df = filtered_df[
        (filtered_df["SalePrice"] >= price_range[0]) & 
        (filtered_df["SalePrice"] <= price_range[1]) &
        (filtered_df["OverallQual"] >= qual_range[0]) &
        (filtered_df["OverallQual"] <= qual_range[1])
    ]

    st.markdown(f"**Filtered Cohort**: Showing **{len(filtered_df):,}** properties matching selected criteria.")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Geographic & Quality Segments",
        "Living Area & Rooms",
        "Price per Sq Ft & Age",
        "Correlation Matrix",
        "⚡ Enterprise SQL Analytics Studio"
    ])

    with tab1:
        c1, c2 = st.columns([1.1, 0.9])
        with c1:
            st.markdown("<div class='card-section'>", unsafe_allow_html=True)
            st.markdown("<div class='card-title'>Median Sale Price by Neighborhood</div>", unsafe_allow_html=True)
            neigh_stats = segment_by_neighborhood(filtered_df)
            if not neigh_stats.empty:
                fig_neigh = px.bar(
                    neigh_stats.head(15),
                    x="median_price",
                    y="Neighborhood",
                    orientation="h",
                    color="median_price",
                    color_continuous_scale="Blues",
                    labels={"median_price": "Median Price ($)", "Neighborhood": "Neighborhood"},
                    text_auto=".2s"
                )
                fig_neigh.update_layout(yaxis=dict(autorange="reversed"), height=380, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_neigh, use_container_width=True)
            else:
                st.info("No records match the current filter selection.")
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown("<div class='card-section'>", unsafe_allow_html=True)
            st.markdown("<div class='card-title'>Price by Overall Material & Finish Quality (1–10)</div>", unsafe_allow_html=True)
            fig_qual = px.box(
                filtered_df,
                x="OverallQual",
                y="SalePrice",
                color="OverallQual",
                labels={"OverallQual": "Overall Quality (1=Very Poor, 10=Very Excellent)", "SalePrice": "Sale Price ($)"}
            )
            fig_qual.update_layout(height=380, margin=dict(l=20, r=20, t=30, b=20), showlegend=False)
            st.plotly_chart(fig_qual, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("<div class='card-section'>", unsafe_allow_html=True)
            st.markdown("<div class='card-title'>Price vs Above-Grade Living Area (Sq Ft)</div>", unsafe_allow_html=True)
            fig_area = px.scatter(
                filtered_df,
                x="GrLivArea",
                y="SalePrice",
                color="OverallQual",
                labels={"GrLivArea": "Above-Ground Living Area (Sq Ft)", "SalePrice": "Sale Price ($)", "OverallQual": "Quality"}
            )
            fig_area.update_layout(height=380, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_area, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown("<div class='card-section'>", unsafe_allow_html=True)
            st.markdown("<div class='card-title'>Price Distribution by Bedroom Count</div>", unsafe_allow_html=True)
            if "BedroomAbvGr" in filtered_df.columns:
                fig_bed = px.box(
                    filtered_df[filtered_df["BedroomAbvGr"] <= 6],
                    x="BedroomAbvGr",
                    y="SalePrice",
                    color="BedroomAbvGr",
                    labels={"BedroomAbvGr": "Bedrooms Above Grade", "SalePrice": "Sale Price ($)"}
                )
                fig_bed.update_layout(height=380, margin=dict(l=20, r=20, t=30, b=20), showlegend=False)
                st.plotly_chart(fig_bed, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with tab3:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("<div class='card-section'>", unsafe_allow_html=True)
            st.markdown("<div class='card-title'>Price per Square Foot by Price Tier</div>", unsafe_allow_html=True)
            tier_df = segment_by_price_tier(filtered_df)
            if not tier_df.empty:
                fig_tier = px.bar(
                    tier_df,
                    x="PriceTier",
                    y="mean_price",
                    color="PriceTier",
                    labels={"PriceTier": "Price Tier", "mean_price": "Mean Sale Price ($)"},
                    text_auto=".2s"
                )
                fig_tier.update_layout(height=380, margin=dict(l=20, r=20, t=30, b=20), showlegend=False)
                st.plotly_chart(fig_tier, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown("<div class='card-section'>", unsafe_allow_html=True)
            st.markdown("<div class='card-title'>House Construction Year vs Sale Price</div>", unsafe_allow_html=True)
            if "YearBuilt" in filtered_df.columns:
                fig_yb = px.scatter(
                    filtered_df,
                    x="YearBuilt",
                    y="SalePrice",
                    color="OverallQual",
                    opacity=0.6,
                    labels={"YearBuilt": "Year Built", "SalePrice": "Sale Price ($)", "OverallQual": "Quality"}
                )
                fig_yb.update_layout(height=380, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_yb, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with tab4:
        st.markdown("<div class='card-section'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Numerical Feature Correlation Heatmap</div>", unsafe_allow_html=True)
        key_corr_cols = [
            "SalePrice", "GrLivArea", "OverallQual", "TotalBsmtSF", "1stFlrSF",
            "GarageCars", "GarageArea", "FullBath", "YearBuilt", "YearRemodAdd"
        ]
        available_corr = [c for c in key_corr_cols if c in filtered_df.columns]
        corr_matrix = filtered_df[available_corr].corr().round(2)
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Blues",
            labels=dict(color="Correlation")
        )
        fig_corr.update_layout(height=450, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_corr, use_container_width=True)
        st.info("⚠️ Note: Statistical correlation reflects empirical linear association in historical data and must not be conflated with causation.")
        st.markdown("</div>", unsafe_allow_html=True)

    with tab5:
        st.markdown("<div class='card-section'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>⚡ Enterprise Relational SQL Analytics Layer (SQLite 3)</div>", unsafe_allow_html=True)
        st.markdown(
            """
            This module provides direct SQL querying over the relational SQLite database (`data/db/housewise.db`).
            Demonstrates enterprise BI competencies: Common Table Expressions (CTEs), Window Functions (`DENSE_RANK()`, `AVG() OVER()`, `NTILE(4)`), cohort aggregations, and query optimization.
            """
        )

        try:
            sql_engine = SQLEngine()
            query_files = sql_engine.list_available_queries()
            
            sub_col1, sub_col2 = st.columns([1, 1])
            with sub_col1:
                selected_query = st.selectbox(
                    "Select Pre-Built Production SQL Analytical Script:",
                    query_files,
                    format_func=lambda x: f"{x} — {x.split('_', 1)[1].replace('.sql', '').replace('_', ' ').title()}"
                )
            
            with sub_col2:
                show_schema = st.checkbox("Show Database Schema & Indexes (PRAGMA)", value=False)

            if show_schema:
                schema_df = sql_engine.get_table_schema("properties")
                st.caption(f"Table: `properties` ({len(schema_df)} columns indexed in SQLite)")
                st.dataframe(schema_df[["cid", "name", "type", "notnull", "pk"]], height=200, use_container_width=True)

            if selected_query:
                query_sql = sql_engine.get_query_sql(selected_query)
                with st.expander(f"Inspect SQL Code: `{selected_query}`", expanded=False):
                    st.code(query_sql, language="sql")

                t0 = time.perf_counter()
                query_results = sql_engine.execute_script_query(selected_query)
                exec_time_ms = (time.perf_counter() - t0) * 1000

                st.markdown(
                    f"**Execution Status**: Returned **{len(query_results):,} rows** × **{len(query_results.columns)} columns** "
                    f"in **{exec_time_ms:.2f} ms** | Engine: `SQLite 3.x`"
                )
                st.dataframe(query_results, use_container_width=True)

                csv_data = query_results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label=f"📥 Export Results as CSV ({selected_query.replace('.sql', '.csv')})",
                    data=csv_data,
                    file_name=selected_query.replace(".sql", ".csv"),
                    mime="text/csv"
                )

            st.markdown("---")
            st.markdown("##### 🔍 Interactive Ad-Hoc SQL Query Console")
            st.caption("Execute read-only SELECT or WITH statements against `properties` table in `data/db/housewise.db`:")
            
            custom_query_example = "SELECT Neighborhood, COUNT(*) AS count, ROUND(AVG(SalePrice), 0) AS avg_price FROM properties GROUP BY Neighborhood ORDER BY avg_price DESC LIMIT 10;"
            custom_sql = st.text_area("SQL Statement:", value=custom_query_example, height=90)
            
            if st.button("Execute Custom SQL", type="primary"):
                t_custom_0 = time.perf_counter()
                custom_df, err = sql_engine.execute_custom_query(custom_sql)
                t_custom_ms = (time.perf_counter() - t_custom_0) * 1000
                
                if err:
                    st.error(f"SQL Error: {err}")
                elif custom_df is not None:
                    st.success(f"Executed successfully in {t_custom_ms:.2f} ms ({len(custom_df)} rows returned)")
                    st.dataframe(custom_df, use_container_width=True)

        except Exception as e:
            st.error(f"SQL Engine connection error: {e}")

        st.markdown("</div>", unsafe_allow_html=True)
