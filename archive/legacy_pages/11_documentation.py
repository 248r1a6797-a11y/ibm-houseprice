"""Page 11 — Documentation & Interactive Data Dictionary."""
import streamlit as st
import pandas as pd
from pathlib import Path
from app.components.header import render_header

def render_page(df: pd.DataFrame):
    render_header(
        title="Technical Documentation & Data Dictionary",
        subtitle="Searchable documentation for all 82 dataset variables, provenance records, and regulatory notices.",
        badge="TECHNICAL DOCUMENTATION"
    )

    tab1, tab2 = st.tabs(["Interactive Data Dictionary (82 Features)", "Provenance & Data Sources"])

    with tab1:
        st.markdown("<div class='card-section'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Authoritative Variable Codebook</div>", unsafe_allow_html=True)
        
        search_query = st.text_input("Filter variables by keyword (e.g. 'area', 'qual', 'garage', 'bath'):", "")
        
        # Read markdown dictionary or build dataframe
        doc_md_path = Path("docs/data_dictionary.md")
        if doc_md_path.exists():
            with open(doc_md_path, "r", encoding="utf-8") as f:
                content = f.read()
            st.markdown(content)
        else:
            st.info("Data dictionary file not found.")
        st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        st.markdown("<div class='card-section'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Data Provenance & Citations</div>", unsafe_allow_html=True)
        st.markdown("""
        - **Primary Citation**: De Cock, Dean. "Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project." *Journal of Statistics Education*, Volume 19, Number 3 (2011).
        - **Archive Source**: American Statistical Association (ASA) JSE Data Archive.
        - **Geographic Scope**: Ames, Iowa, USA (incorporating 28 distinct neighborhoods).
        - **Time Period**: January 2006 – July 2010.
        - **Authoritative Text Documentation**: Available locally at `data/raw/DataDocumentation.txt`.
        """)
        st.markdown("</div>", unsafe_allow_html=True)
