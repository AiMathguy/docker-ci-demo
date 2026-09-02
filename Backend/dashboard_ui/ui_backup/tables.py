import streamlit as st


def display_top_risky_users(df, columns, n=15):
    top_risk = df.sort_values("churn_probability", ascending=False)[columns].head(n)
    st.dataframe(top_risk, use_container_width=True)
