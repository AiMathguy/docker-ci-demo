import streamlit as st
import plotly.express as px


def load_css():
    st.markdown(
        '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
    <style>
        /* Paste your full CSS here – same as in previous utils.py */
        /* (Keep all the CSS you already have – it's the same) */
    </style>
    """,
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str):
    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-title">{title}</div>
            <div class="hero-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, foot: str = ""):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-foot">{foot}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def panel_open(title: str, subtitle: str = ""):
    st.markdown(
        f"""
        <div class="panel-card">
            <div class="panel-title">{title}</div>
            <div class="panel-subtitle">{subtitle}</div>
        """,
        unsafe_allow_html=True,
    )


def panel_close():
    st.markdown("</div>", unsafe_allow_html=True)


def polish_plotly(fig, height=380, line_color="#3b82f6", bar_color="#8b5cf6"):
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=8, r=8, t=18, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248,250,252,0.9)",
        font=dict(family="Inter, sans-serif", color="#0f172a"),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            linecolor="#e2e8f0",
            tickfont=dict(color="#475569"),
        ),
        yaxis=dict(gridcolor="#e2e8f0", zeroline=False, tickfont=dict(color="#475569")),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#475569"),
        ),
    )
    for trace in fig.data:
        if hasattr(trace, "line") and trace.line is not None:
            trace.line.color = line_color
        if hasattr(trace, "marker") and trace.marker is not None:
            if not isinstance(trace.marker.color, (list, tuple)):
                trace.marker.color = bar_color
    return fig
