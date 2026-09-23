"""Page 10 — Proposed Model Monitoring & Future Roadmap."""
import streamlit as st
from app.components.header import render_header

def render_page():
    render_header(
        title="Model Governance & Monitoring Framework",
        subtitle="Proposed production drift monitoring, retraining telemetry, and future feature expansion roadmap.",
        badge="MLOPS & MONITORING"
    )

    st.markdown(
        """
        <div style="background: #F8FAFC; border: 1px solid #E2E8F0; padding: 14px 18px; border-radius: 6px; margin-bottom: 20px; font-size: 0.88rem; color: #475569;">
            <b>Operational Context</b>: In live production deployments, real estate models degrade over time due to inflation, interest rate shifts, and neighborhood demographic changes. The architecture below specifies our automated monitoring and retraining protocols.
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='card-section'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Production Telemetry & Drift Detection</div>", unsafe_allow_html=True)
        st.markdown("""
        - **Feature Distribution Drift (Covariate Shift)**:
          - Monitor continuous distributions (e.g., `GrLivArea`, `LotArea`) using the **Kolmogorov-Smirnov (KS) test** ($p < 0.05$ trigger).
          - Track categorical distributions (e.g., `Neighborhood`, `Zoning`) via **Population Stability Index (PSI > 0.2)**.
        - **Target & Residual Drift**:
          - Compute rolling 30-day Mean Absolute Error (MAE). If rolling MAE degrades by $>15\%$ relative to baseline, flag automated review.
        - **Missingness & Pipeline Anomalies**:
          - Monitor percentage of unexpected category encounters in `OneHotEncoder(handle_unknown='ignore')`.
        """)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card-section'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Future Capabilities & Feature Roadmap</div>", unsafe_allow_html=True)
        st.markdown("""
        - **Macroeconomic Enrichments**:
          - Integrate federal 30-year fixed mortgage interest rate indices.
          - Incorporate regional CPI and statewide unemployment rates.
        - **Geospatial & Neighborhood Amenities**:
          - Spatial proximity metrics: distance to Iowa State University, major highways, retail centers.
          - School district ratings and transit accessibility scores.
        - **Multi-Modal Vision Modeling**:
          - Aerial satellite imagery and interior photograph feature embeddings via Vision Transformers (ViT) to augment tabular quality ratings.
        """)
        st.markdown("</div>", unsafe_allow_html=True)
