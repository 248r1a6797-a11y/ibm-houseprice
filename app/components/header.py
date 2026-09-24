"""Reusable UI header component."""
import streamlit as st

def render_header(title: str, subtitle: str, badge: str = "PRODUCTION ML"):
    """Render modern SaaS header with title, subtitle, breadcrumb and badge."""
    header_html = (
        '<div style="margin-bottom: 22px; padding-bottom: 14px; border-bottom: 1px solid #E2E8F0;">'
        '<div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">'
        f'<h1 style="font-size: 1.75rem; font-weight: 800; color: #0F172A; margin: 0; letter-spacing: -0.02em;">{title}</h1>'
        f'<span style="background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%); color: #1D4ED8; font-size: 0.72rem; font-weight: 700; padding: 4px 12px; border-radius: 9999px; border: 1px solid #BFDBFE; text-transform: uppercase; letter-spacing: 0.06em;">{badge}</span>'
        '</div>'
        f'<p style="font-size: 0.92rem; color: #64748B; margin-top: 6px; margin-bottom: 0; line-height: 1.5;">{subtitle}</p>'
        '</div>'
    )
    if hasattr(st, "html"):
        st.html(header_html)
    else:
        st.markdown(header_html, unsafe_allow_html=True)
