"""Page 5 — Interactive Exploratory Data Analysis (EDA) Laboratory."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from app.components.header import render_header

def render_page():
    render_header(
        title="Interactive Exploratory Data Analysis (EDA) Laboratory",
        subtitle="Dynamic multidimensional visualization workspace: build custom charts, segment cohorts, and test hypotheses.",
        badge="EXPLORATION LAYER"
    )

    df = st.session_state.clean_df if not st.session_state.clean_df.empty else st.session_state.raw_df
    target_col = st.session_state.target_col

    if df.empty:
        st.warning("No data available in session. Please visit 1. Data Workspace.")
        return

    st.markdown("<div class='card-section'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>🎨 Interactive Multi-Variable Chart Builder</div>", unsafe_allow_html=True)

    all_cols = list(df.columns)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        default_x = "GrLivArea" if "GrLivArea" in all_cols else all_cols[0]
        x_var = st.selectbox("X Variable:", all_cols, index=all_cols.index(default_x))

    with col2:
        default_y = target_col if target_col in all_cols else (all_cols[1] if len(all_cols) > 1 else all_cols[0])
        y_var = st.selectbox("Y Variable (Optional for Hist):", ["(None)"] + all_cols, index=1 if default_y in all_cols else 0)

    with col3:
        chart_types = ["Scatter Plot", "Box Plot", "Violin Plot", "Histogram / Distribution", "Bar Chart", "2D Density Contour"]
        chart_type = st.selectbox("Chart Type:", chart_types)

    with col4:
        hue_var = st.selectbox("Group By / Color (Hue):", ["(None)"] + all_cols, index=0)

    hue_param = None if hue_var == "(None)" else hue_var
    y_param = None if y_var == "(None)" else y_var

    # Render Dynamic Chart
    try:
        fig = None
        if chart_type == "Scatter Plot":
            if y_param is None:
                st.info("Please select a Y Variable for the Scatter Plot.")
            else:
                fig = px.scatter(
                    df,
                    x=x_var,
                    y=y_param,
                    color=hue_param,
                    opacity=0.75,
                    labels={x_var: x_var, y_param: y_param}
                )

        elif chart_type == "Box Plot":
            fig = px.box(
                df,
                x=x_var,
                y=y_param,
                color=hue_param,
                labels={x_var: x_var, y_param: y_param if y_param else ""}
            )

        elif chart_type == "Violin Plot":
            fig = px.violin(
                df,
                x=x_var,
                y=y_param,
                color=hue_param,
                box=True,
                points="all",
                labels={x_var: x_var, y_param: y_param if y_param else ""}
            )

        elif chart_type == "Histogram / Distribution":
            fig = px.histogram(
                df,
                x=x_var,
                y=y_param,
                color=hue_param,
                marginal="box",
                nbins=40,
                labels={x_var: x_var}
            )

        elif chart_type == "Bar Chart":
            agg_df = df.groupby(x_var)[y_param].median().reset_index() if y_param else df[x_var].value_counts().reset_index()
            fig = px.bar(
                agg_df.head(25),
                x=agg_df.columns[0],
                y=agg_df.columns[1],
                color=agg_df.columns[1],
                labels={agg_df.columns[0]: x_var, agg_df.columns[1]: f"Median {y_param}" if y_param else "Count"}
            )

        elif chart_type == "2D Density Contour":
            if y_param is None:
                st.info("Please select a Y Variable for 2D Density.")
            else:
                fig = px.density_contour(
                    df,
                    x=x_var,
                    y=y_param,
                    color=hue_param
                )

        if fig:
            fig.update_layout(height=480, margin=dict(l=20, r=20, t=30, b=20), plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

            # Automated Insight Generation based on actual calculation
            st.markdown("##### 💡 Analytical Observation")
            if x_var in num_cols and y_param in num_cols and y_param:
                r_val = df[x_var].corr(df[y_param])
                direction = "strong positive" if r_val > 0.6 else ("moderate positive" if r_val > 0.3 else ("negative" if r_val < -0.3 else "weak / neutral"))
                st.markdown(
                    f"**Statistical Evidence**: Pearson correlation $r = {r_val:.4f}$. There is a **{direction}** association "
                    f"between `{x_var}` and `{y_param}`. Changes in `{x_var}` correspond with predictable monotonic shifts in `{y_param}`."
                )
            elif x_var in cat_cols and y_param in num_cols and y_param:
                top_group = df.groupby(x_var)[y_param].median().idxmax()
                top_val = df.groupby(x_var)[y_param].median().max()
                low_group = df.groupby(x_var)[y_param].median().idxmin()
                low_val = df.groupby(x_var)[y_param].median().min()
                st.markdown(
                    f"**Categorical Variation**: Subgroup `{top_group}` captures the highest median `{y_param}` (${top_val:,.0f}), "
                    f"while `{low_group}` captures the lowest (${low_val:,.0f}), demonstrating a spread of ${top_val - low_val:,.0f} across categories."
                )

    except Exception as e:
        st.error(f"Error rendering chart: {e}")

    st.markdown("</div>", unsafe_allow_html=True)
