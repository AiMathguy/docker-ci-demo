import streamlit as st
from datetime import datetime, timedelta
from app.services.data_service import load_all_data, apply_filters


def render_sidebar():
    users, activity, features, preds, subs = load_all_data()

    users["id"] = users["id"].astype(str)
    users["created_at"] = pd.to_datetime(users["created_at"], errors="coerce")

    with st.sidebar:
        st.markdown(
            "### <i class='fas fa-compass'></i> Navigation", unsafe_allow_html=True
        )
        page = st.radio(
            "Go to",
            ["Dashboard", "ML Predictions", "Explainability"],
            index=0,
            label_visibility="collapsed",
            key="page_nav",
        )
        st.markdown("---")

        st.markdown(
            "### <i class='fas fa-sliders-h'></i> Filters", unsafe_allow_html=True
        )

        # Date range
        st.markdown(
            """
        <div class="filter-card">
            <div class="filter-title"><i class='fas fa-calendar-alt'></i> USER CREATION DATE</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
        min_date = (
            users["created_at"].min().date()
            if users["created_at"].notna().any()
            else datetime.today().date() - timedelta(days=30)
        )
        max_date = (
            users["created_at"].max().date()
            if users["created_at"].notna().any()
            else datetime.today().date()
        )
        date_range = st.date_input(
            "",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            label_visibility="collapsed",
        )

        # Role
        st.markdown(
            """
        <div class="filter-card">
            <div class="filter-title"><i class='fas fa-user-tag'></i> USER ROLE</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
        role_options = sorted(users["role"].dropna().unique().tolist())
        role_filter = st.multiselect(
            "", options=role_options, default=role_options, label_visibility="collapsed"
        )

        # Status
        st.markdown(
            """
        <div class="filter-card">
            <div class="filter-title"><i class='fas fa-circle'></i> USER STATUS</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
        status_options = sorted(users["status"].dropna().unique().tolist())
        status_filter = st.multiselect(
            "",
            options=status_options,
            default=status_options,
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.caption(
            "<i class='fas fa-chart-simple'></i> CEO Growth & Churn Dashboard · v2.0",
            unsafe_allow_html=True,
        )

    filtered_df = apply_filters(
        users, features, preds, subs, date_range, role_filter, status_filter
    )

    # Store in session state for other pages
    st.session_state.filtered_df = filtered_df
    st.session_state.users = users
    st.session_state.activity = activity

    return page, filtered_df, users, activity
