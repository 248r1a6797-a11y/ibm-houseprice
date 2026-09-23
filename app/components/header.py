"""Reusable UI header component."""
import streamlit as st

def render_header(title: str, subtitle: str, badge: str = "PRODUCTION ML"):
    """Render corporate header with title, subtitle, and badge."""
    st.markdown(
        f"""
        <div style="margin-bottom: 24px; padding-bottom: 12px; border-bottom: 1px solid #E2E8F0;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h1 style="font-size: 1.85rem; font-weight: 700; color: #0F172A; margin: 0;">{title}</h1>
                <span style="background: #EFF6FF; color: #1D4ED8; font-size: 0.75rem; font-weight: 600; padding: 4px 10px; border-radius: 9999px; border: 1px solid #BFDBFE;">{badge}</span>
            </div>
            <p style="font-size: 0.95rem; color: #64748B; margin-top: 6px; margin-bottom: 0;">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
