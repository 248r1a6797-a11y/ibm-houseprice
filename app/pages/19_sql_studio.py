"""Page 19 — Enterprise Relational SQL Studio & Schema Inspector."""
import time
from pathlib import Path
import streamlit as st
import pandas as pd
import sqlite3

from app.components.header import render_header
from src.analytics.sql_engine import SQLEngine

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

PRESET_QUERIES = {
    "02_market_overview.sql": {
        "title": "Macro Market Overview & Price Distribution",
        "description": "Calculates municipality-wide valuation benchmarks, volume, and distribution spread."
    },
    "03_neighborhood_price_per_sqft.sql": {
        "title": "Submarket Price per Square Foot Dispersion",
        "description": "Compares unit economics ($/sqft) across submarkets, ranked by median price."
    },
    "04_quality_condition_matrix.sql": {
        "title": "Quality vs Condition Cross-Tabulation",
        "description": "Analyzes how construction quality interacts with maintenance condition."
    },
    "05_property_age_depreciation.sql": {
        "title": "Vintage Depreciation & Age Cohorts",
        "description": "Evaluates depreciation schedules and price retention across historic construction decades."
    },
    "06_remodel_roi_impact.sql": {
        "title": "Remodeling Return on Investment Premiums",
        "description": "Quantifies the empirical price differential between remodeled homes and original-condition properties."
    },
    "01_schema.sql": {
        "title": "Relational Table DDL & Index Definitions",
        "description": "Full SQLite table structure, data types, constraints, and column indexing statements."
    }
}

def render_page():
    render_header(
        title="Enterprise Relational SQL Studio",
        subtitle="High-performance SQLite engine executing analytical queries, schema inspection, and ad-hoc SQL querying against active data.",
        badge="SQL ANALYTICS"
    )

    engine = SQLEngine()

    tab1, tab2, tab3 = st.tabs([
        "📜 Analytical Script Suite",
        "💻 Ad-Hoc SQL Query Console",
        "🗂️ Relational Schema Inspector (properties)"
    ])

    # ==========================================================
    # TAB 1: PRE-BUILT ANALYTICAL QUERIES
    # ==========================================================
    with tab1:
        st.markdown("<div class='card-section'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Production SQL Analytics Suite</div>", unsafe_allow_html=True)

        query_keys = list(PRESET_QUERIES.keys())
        query_labels = [f"{k} — {PRESET_QUERIES[k]['title']}" for k in query_keys]
        
        selected_idx = st.selectbox(
            "Select an Analytical Script to Inspect and Execute:",
            range(len(query_keys)),
            format_func=lambda i: query_labels[i]
        )
        selected_script = query_keys[selected_idx]
        script_info = PRESET_QUERIES[selected_script]

        st.caption(f"**Objective**: {script_info['description']}")

        # Read SQL query content
        try:
            sql_text = engine.get_query_sql(selected_script)
            with st.expander("🔍 View Raw SQL Query", expanded=True):
                st.code(sql_text, language="sql")

            if st.button(f"▶️ Execute {selected_script}", type="primary", use_container_width=True):
                t0 = time.perf_counter()
                result_df = engine.execute_script_query(selected_script)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

                st.success(f"Query executed successfully in **{elapsed_ms:.2f} ms** | Returned **{len(result_df):,} rows**.")
                st.dataframe(result_df, use_container_width=True)

                csv_bytes = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label=f"📥 Download Query Results ({selected_script}.csv)",
                    data=csv_bytes,
                    file_name=f"{selected_script}.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.warning(f"Notice: This benchmark query references specific Ames column schemas. Use the Ad-Hoc Console in Tab 2 for arbitrary custom tables. Details: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

    # ==========================================================
    # TAB 2: AD-HOC SQL CONSOLE
    # ==========================================================
    with tab2:
        st.markdown("<div class='card-section'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Ad-Hoc SQL Query Console</div>", unsafe_allow_html=True)
        st.markdown("Execute read-only SQL queries (`SELECT` / `WITH`) against the relational `properties` table.")

        # Adapt default query to active schema
        target_col = st.session_state.get("target_col", "SalePrice")
        ctx = st.session_state.get("dataset_context")
        loc_col = ctx.location_col if ctx and ctx.location_col else ("Neighborhood" if "Neighborhood" in st.session_state.raw_df.columns else (st.session_state.raw_df.columns[0] if not st.session_state.raw_df.empty else "id"))
        
        default_sql = f"""SELECT 
    {loc_col},
    COUNT(*) AS property_count,
    ROUND(AVG({target_col}), 0) AS avg_{target_col}
FROM properties
GROUP BY {loc_col}
ORDER BY avg_{target_col} DESC
LIMIT 10;"""

        custom_query = st.text_area("SQL Statement:", value=default_sql, height=160)

        c1, c2 = st.columns([1, 4])
        with c1:
            run_btn = st.button("🚀 Run SQL Query", type="primary")

        if run_btn and custom_query:
            t0 = time.perf_counter()
            df_custom, err = engine.execute_custom_query(custom_query)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            if err:
                st.error(err)
            else:
                st.success(f"Executed in **{elapsed_ms:.2f} ms** | Returned **{len(df_custom):,} rows**.")
                st.dataframe(df_custom, use_container_width=True)

                csv_custom = df_custom.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Export Query Result CSV",
                    data=csv_custom,
                    file_name="custom_sql_query_result.csv",
                    mime="text/csv"
                )

        st.markdown("</div>", unsafe_allow_html=True)

    # ==========================================================
    # TAB 3: SCHEMA INSPECTOR
    # ==========================================================
    with tab3:
        st.markdown("<div class='card-section'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Relational Table Schema: `properties`</div>", unsafe_allow_html=True)
        try:
            db_path = ROOT_DIR / "database/housewise.db"
            if db_path.exists():
                with sqlite3.connect(db_path) as conn:
                    schema_df = pd.read_sql("PRAGMA table_info(properties);", conn)
                st.dataframe(schema_df[["cid", "name", "type", "notnull", "dflt_value", "pk"]], use_container_width=True)
            else:
                st.info("Database file not yet created. Load a dataset in 1. Data Workspace to initialize.")
        except Exception as e:
            st.error(f"Error inspecting schema: {e}")
        st.markdown("</div>", unsafe_allow_html=True)
