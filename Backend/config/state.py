import streamlit as st


def init_session_state():
    if "filtered_df" not in st.session_state:
        st.session_state.filtered_df = None
    if "users" not in st.session_state:
        st.session_state.users = None
    if "activity" not in st.session_state:
        st.session_state.activity = None
    if "model" not in st.session_state:
        st.session_state.model = None
    if "page" not in st.session_state:
        st.session_state.page = "Dashboard"
