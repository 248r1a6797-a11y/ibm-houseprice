"""Pipeline lifecycle status banner component for HouseWise AI."""
import streamlit as st

def render_pipeline_banner():
    """Renders a sleek, modern enterprise lifecycle stepper."""
    status = st.session_state.get("pipeline_status", {})
    needs_retrain = st.session_state.get("needs_retraining", False)

    steps = [
        ("Data Ingestion", status.get("DATA", "✓ Loaded")),
        ("Quality Profiler", status.get("QUALITY", "✓ Validated")),
        ("Cleaning Studio", status.get("CLEANING", "✓ Completed")),
        ("Feature Studio", status.get("FEATURES", "✓ Selected")),
        ("EDA Lab", status.get("EDA", "✓ Ready")),
        ("Model Lab", status.get("TRAINING", "✓ Trained")),
        ("Diagnostics", status.get("EVALUATION", "✓ Scored")),
        ("Deployment", status.get("PRODUCTION", "✓ Champion"))
    ]

    pills_html = []
    for step_title, step_val in steps:
        is_pending = "Pending" in step_val or "⏳" in step_val
        is_ready = "✓" in step_val or "Ready" in step_val or "Loaded" in step_val or "Champion" in step_val
        
        if is_ready:
            badge_bg = "#ECFDF5"
            badge_border = "#A7F3D0"
            badge_text = "#065F46"
            dot_color = "#10B981"
            short_status = "Ready" if "Ready" in step_val or "Champion" in step_val else ("Loaded" if "Loaded" in step_val else "Done")
        elif is_pending:
            badge_bg = "#FFFBEB"
            badge_border = "#FDE68A"
            badge_text = "#92400E"
            dot_color = "#F59E0B"
            short_status = "Pending"
        else:
            badge_bg = "#F1F5F9"
            badge_border = "#E2E8F0"
            badge_text = "#475569"
            dot_color = "#94A3B8"
            short_status = step_val[:10]

        pills_html.append(
            f'<div style="display: inline-flex; align-items: center; gap: 6px; background: {badge_bg}; border: 1px solid {badge_border}; border-radius: 9999px; padding: 3px 10px; font-size: 0.74rem; font-weight: 600; color: {badge_text}; white-space: nowrap;">'
            f'<span style="width: 6px; height: 6px; border-radius: 50%; background: {dot_color}; display: inline-block;"></span>'
            f'<span>{step_title}</span>'
            f'<span style="opacity: 0.65; font-weight: 500;">· {short_status}</span>'
            f'</div>'
        )

    retrain_pill = ""
    if needs_retrain:
        retrain_pill = (
            '<div style="margin-top: 8px; display: flex; align-items: center; gap: 8px; background: #FFFBEB; border: 1px solid #FDE68A; border-radius: 8px; padding: 6px 12px; font-size: 0.78rem; color: #92400E;">'
            '<span>⚡</span>'
            '<span><b>Pipeline Sync Notice:</b> Upstream dataset or feature parameters have changed. Run <b>07. Model Benchmark Lab</b> to synchronize models.</span>'
            '</div>'
        )

    banner_html = (
        '<div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 8px 14px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.03);">'
        f'<div style="display: flex; gap: 6px; overflow-x: auto; padding-bottom: 2px; align-items: center;">{"".join(pills_html)}</div>'
        f'{retrain_pill}'
        '</div>'
    )

    if hasattr(st, "html"):
        st.html(banner_html)
    else:
        st.markdown(banner_html, unsafe_allow_html=True)
