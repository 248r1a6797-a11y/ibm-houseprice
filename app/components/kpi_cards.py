"""Reusable KPI cards component."""
import streamlit as st

def render_kpi_card(title: str, value: str, subtext: str = ""):
    """Render a clean, modern KPI card."""
    st.markdown(
        f"""
        <div class="kpi-container">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-subtext">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
