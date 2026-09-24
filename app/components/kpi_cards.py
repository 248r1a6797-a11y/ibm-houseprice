"""Reusable KPI cards component."""
import streamlit as st

def render_kpi_card(title: str, value: str, subtext: str = ""):
    """Render a clean, modern KPI card."""
    kpi_html = (
        '<div class="kpi-container">'
        f'<div class="kpi-title">{title}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-subtext">{subtext}</div>'
        '</div>'
    )
    if hasattr(st, "html"):
        st.html(kpi_html)
    else:
        st.markdown(kpi_html, unsafe_allow_html=True)
