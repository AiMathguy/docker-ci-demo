import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from ml_model import load_model

BASE_DIR = Path(__file__).resolve().parent
LOGO_PATH = BASE_DIR / "assets" / "linkfields_logo 1 1.svg"


def load_css():
    st.markdown(
        '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <style>
        :root {
            --bg: #050712;
            --surface: #0b0d18;
            --surface-2: #121522;
            --surface-3: #181b2b;

            --border: rgba(255,255,255,0.08);
            --border-dark: rgba(167,139,250,0.45);

            --text: #f8fafc;
            --text-muted: #8b90a3;

            --blue: #8b7cff;
            --purple: #a78bfa;
            --yellow: #c4b5fd;
        }

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg);
            color: var(--text);
        }

        #MainMenu, footer, header {
            visibility: hidden;
        }

        .stApp {
            background:
                radial-gradient(circle at top right, rgba(139,124,255,0.18), transparent 28%),
                radial-gradient(circle at bottom left, rgba(167,139,250,0.10), transparent 30%),
                linear-gradient(135deg, #050712 0%, #090b16 45%, #101322 100%);
        }

        .main .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 94%;
        }

        section[data-testid="stSidebar"],
        [data-testid="collapsedControl"] {
            display: none !important;
        }

        .top-nav {
            background: rgba(11,13,24,0.88);
            border: 1px solid var(--border);
            border-radius: 28px;
            padding: 0.9rem;
            margin-bottom: 1.5rem;
            box-shadow:
                0 16px 40px rgba(0,0,0,0.45),
                inset 0 1px 0 rgba(255,255,255,0.04);
            backdrop-filter: blur(18px);
        }

        .hero-card {
            background:
                radial-gradient(circle at top right, rgba(139,124,255,0.22), transparent 35%),
                linear-gradient(135deg, rgba(18,21,34,0.96), rgba(7,9,18,0.98));
            border: 1px solid var(--border);
            border-radius: 32px;
            padding: 1.8rem 2rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 18px 50px rgba(0,0,0,0.45);
        }

        .hero-title {
            font-size: 2.2rem;
            font-weight: 800;
            background: linear-gradient(
                135deg,
                #ffffff 0%,
                #c4b5fd 35%,
                #8b7cff 70%,
                #a78bfa 100%
            );
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            letter-spacing: -0.03em;
        }

        .hero-subtitle {
            margin-top: 0.4rem;
            color: var(--text-muted);
        }

        .kpi-card {
            background:
                linear-gradient(180deg, rgba(18,21,34,0.96), rgba(9,11,21,0.98));
            border: 1px solid var(--border);
            border-radius: 28px;
            padding: 1.25rem 1rem;
            box-shadow: 0 14px 36px rgba(0,0,0,0.38);
            transition: all 0.2s ease;
        }

        .kpi-card:hover {
            border-color: var(--border-dark);
            transform: translateY(-3px);
            box-shadow: 0 20px 55px rgba(139,124,255,0.18);
        }

        .kpi-label {
            font-size: 0.7rem;
            font-weight: 700;
            color: #a78bfa;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .kpi-value {
            font-size: 1.75rem;
            font-weight: 800;
            color: #ffffff;
        }

        .kpi-foot {
            font-size: 0.72rem;
            color: var(--text-muted);
        }

        .panel-card {
            background:
                radial-gradient(circle at top right, rgba(139,124,255,0.08), transparent 30%),
                linear-gradient(180deg, rgba(18,21,34,0.96), rgba(7,9,18,0.98));
            border: 1px solid var(--border);
            border-radius: 30px;
            padding: 1rem 1rem 0.75rem 1rem;
            margin-bottom: 1.2rem;
            box-shadow: 0 16px 45px rgba(0,0,0,0.42);
        }

        .panel-card:hover {
            border-color: rgba(167,139,250,0.36);
        }

        .panel-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: #ffffff;
        }

        .panel-subtitle {
            font-size: 0.8rem;
            color: var(--text-muted);
        }

        .stButton > button,
        .stDownloadButton > button {
            border-radius: 16px;
            border: 1px solid rgba(255,255,255,0.10);
            background: linear-gradient(135deg, #c4b5fd, #8b7cff);
            color: #080a14 !important;
            font-weight: 700;
            padding: 0.55rem 1.2rem;
            box-shadow: 0 8px 24px rgba(139,124,255,0.32);
            transition: all 0.2s ease;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 14px 34px rgba(139,124,255,0.45);
        }

        div[data-testid="stDataFrame"] {
            background:
                linear-gradient(180deg, rgba(18,21,34,0.96), rgba(7,9,18,0.98)) !important;
            border: 1px solid var(--border) !important;
            border-radius: 24px !important;
            overflow: hidden;
            box-shadow: 0 14px 36px rgba(0,0,0,0.35);
            backdrop-filter: blur(12px);
        }

        [data-testid="stDataFrame"] div {
            color: #f8fafc !important;
        }
        /* ===== FORCE DARK DATAFRAME ===== */

        [data-testid="stDataFrame"] * {
            color: #f8fafc !important;
        }

        /* dataframe canvas */
        .glideDataEditor {
            background: rgba(11,13,24,0.98) !important;
        }

        /* header row */
        .glideDataEditor .gdg-header,
        .glideDataEditor .gdg-header-row,
        .glideDataEditor .gdg-column-header {
            background: rgba(139,124,255,0.12) !important;
            color: #c4b5fd !important;
        }

        /* body cells */
        .glideDataEditor .gdg-cell {
            background: rgba(11,13,24,0.96) !important;
            color: #f8fafc !important;
        }

        /* alternating rows */
        .glideDataEditor .gdg-row:nth-child(even) .gdg-cell {
            background: rgba(15,18,30,0.96) !important;
        }

        /* selected cell */
        .glideDataEditor .gdg-selected {
            background: rgba(139,124,255,0.18) !important;
        }

        /* row hover */
        .glideDataEditor .gdg-row:hover .gdg-cell {
            background: rgba(139,124,255,0.08) !important;
        }

        /* scrollbar */
        .glideDataEditor ::-webkit-scrollbar {
            width: 10px;
            height: 10px;
        }

        .glideDataEditor ::-webkit-scrollbar-thumb {
            background: rgba(139,124,255,0.35);
            border-radius: 20px;
        }
        .stPlotlyChart,
        .stPyplot {
            border-radius: 24px !important;
            overflow: hidden;
            background: rgba(11,13,24,0.9);
        }

        input, textarea, select {
            background: rgba(15,23,42,0.9) !important;
            color: white !important;
            border: 1px solid var(--border) !important;
        }
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


def polish_plotly(fig, height=380, line_color="#a78bfa", bar_color="#8b7cff"):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=8, r=8, t=18, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(11,13,24,0.95)",
        font=dict(family="Inter, sans-serif", color="#f8fafc"),
        xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(color="#8b90a3")),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.06)",
            zeroline=False,
            tickfont=dict(color="#8b90a3"),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#8b90a3"),
        ),
    )

    for trace in fig.data:
        if hasattr(trace, "line") and trace.line is not None:
            trace.line.color = line_color
        if hasattr(trace, "marker") and trace.marker is not None:
            if not isinstance(trace.marker.color, (list, tuple)):
                trace.marker.color = bar_color

    return fig


load_css()
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    st.error("DATABASE_URL is not set")
    st.stop()

engine = create_engine(DATABASE_URL, pool_pre_ping=True)


@st.cache_data
def load_table(name: str) -> pd.DataFrame:
    query = text(f"SELECT * FROM {name}")

    with engine.connect() as conn:
        rows = conn.execute(query).mappings().all()

    return pd.DataFrame(rows)


@st.cache_data
def load_all_data():
    users = load_table("users")
    activity = load_table("user_activity_log")
    features = load_table("customer_features")
    preds = load_table("customer_predictions")

    try:
        subs = load_table("subscriptions")
    except Exception:
        subs = pd.DataFrame()

    return users, activity, features, preds, subs


@st.cache_data
def convert_for_download(df: pd.DataFrame):
    return df.to_csv(index=False).encode("utf-8")


def render_top_nav(users: pd.DataFrame):
    st.markdown('<div class="top-nav">', unsafe_allow_html=True)

    nav1, nav2, nav3, nav4, nav5 = st.columns([1, 1, 1, 1, 1.3])

    if "page_nav" not in st.session_state:
        st.session_state["page_nav"] = "Dashboard"

    with nav1:
        if st.button("Dashboard", use_container_width=True):
            st.session_state["page_nav"] = "Dashboard"

    with nav2:
        if st.button("ML Predictions", use_container_width=True):
            st.session_state["page_nav"] = "ML Predictions"

    with nav3:
        if st.button("Explainability", use_container_width=True):
            st.session_state["page_nav"] = "Explainability"

    with nav4:
        if st.button("Data Export", use_container_width=True):
            st.session_state["page_nav"] = "Data Export"

    with nav5:
        st.download_button(
            label="Download CSV",
            data=convert_for_download(users),
            file_name="users.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    return st.session_state["page_nav"]


def render_sidebar():
    users, activity, features, preds, subs = load_all_data()

    if "model" not in st.session_state:
        st.session_state["model"] = load_model()

    users["id"] = users["id"].astype(str)

    if "created_at" in users.columns:
        users["created_at"] = pd.to_datetime(users["created_at"], errors="coerce")

    page = render_top_nav(users)

    df = users.copy()

    if not features.empty:
        features["user_id"] = features["user_id"].astype(str)
        df = df.merge(
            features,
            left_on="id",
            right_on="user_id",
            how="left",
            suffixes=("", "_feat"),
        )

    if not preds.empty:
        preds["user_id"] = preds["user_id"].astype(str)
        df = df.merge(
            preds,
            left_on="id",
            right_on="user_id",
            how="left",
            suffixes=("", "_pred"),
        )

    if not subs.empty:
        subs["user_id"] = subs["user_id"].astype(str)
        df = df.merge(
            subs,
            left_on="id",
            right_on="user_id",
            how="left",
            suffixes=("", "_sub"),
        )

    filtered_df = df.copy()

    st.session_state["filtered_df"] = filtered_df
    st.session_state["users"] = users
    st.session_state["activity"] = activity

    return filtered_df, page
