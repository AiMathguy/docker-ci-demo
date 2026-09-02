import streamlit as st

st.set_page_config(
    page_title="CEO Growth & Churn Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import shap
from ml_model import prepare_features, load_model
from shap_utils import get_explainer, get_shap_values
from utils import (
    load_css,
    hero,
    kpi_card,
    panel_open,
    panel_close,
    polish_plotly,
    render_sidebar,
)

# st.set_page_config(
#     page_title="CEO Growth & Churn Dashboard",
#     layout="wide",
#     initial_sidebar_state="collapsed",
# )

load_css()

filtered_df, page = render_sidebar()

if page == "Dashboard":
    hero(
        "CEO Growth & Churn Dashboard",
        "Real-time growth, revenue, churn risk, and executive-level operational visibility.",
    )

    total_users = len(filtered_df)
    inactive_users = int((filtered_df["status"] == "inactive").sum())
    high_risk_users = (
        int((filtered_df["risk_level"] == "High").sum())
        if "risk_level" in filtered_df.columns
        else 0
    )
    total_revenue = (
        float(filtered_df["amount"].fillna(0).sum())
        if "amount" in filtered_df.columns
        else 0.0
    )

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi_card("Total Users", f"{total_users:,}", "Current filtered base")
    with k2:
        kpi_card("Inactive", f"{inactive_users:,}", "Low engagement")
    with k3:
        kpi_card("High Risk", f"{high_risk_users:,}", "Likely churn candidates")
    with k4:
        kpi_card("Revenue", f"${total_revenue:,.0f}", "Subscription revenue")
    st.markdown("<div style='margin-bottom: 2rem;'></div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        panel_open(" User Growth Over Time", "New users created by day")
        growth_df = (
            filtered_df.groupby(filtered_df["created_at"].dt.date)
            .size()
            .reset_index(name="new_users")
        )
        growth_df.columns = ["date", "new_users"]
        fig = px.line(
            growth_df, x="date", y="new_users", color_discrete_sequence=["#4f46e5"]
        )
        st.plotly_chart(polish_plotly(fig, height=380), use_container_width=True)
        panel_close()

    with c2:
        panel_open(" Daily Activity", "Events and usage trend over time")
        activity = st.session_state.get("activity", pd.DataFrame())
        if not activity.empty:
            filtered_activity = activity[
                activity["user_id"].astype(str).isin(filtered_df["id"])
            ]
            daily_df = (
                filtered_activity.groupby(filtered_activity["activity_time"].dt.date)
                .size()
                .reset_index(name="events")
            )
            daily_df.columns = ["date", "events"]
            fig = px.line(
                daily_df, x="date", y="events", color_discrete_sequence=["#10b981"]
            )
            st.plotly_chart(polish_plotly(fig, height=380), use_container_width=True)
        else:
            st.info("No activity data available.")
        panel_close()

    c3, c4 = st.columns(2)
    with c3:
        panel_open("👥 Users by Status", "Health distribution across the base")
        status_counts = filtered_df["status"].value_counts().reset_index()
        status_counts.columns = ["status", "count"]
        fig = px.bar(
            status_counts,
            x="status",
            y="count",
            color="status",
            color_discrete_sequence=["#4f46e5", "#f59e0b", "#ef4444"],
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(polish_plotly(fig, height=360), use_container_width=True)
        panel_close()

    with c4:
        panel_open("🎭 Users by Role", "Who makes up the customer base")
        role_counts = filtered_df["role"].value_counts().reset_index()
        role_counts.columns = ["role", "count"]
        fig = px.bar(
            role_counts,
            x="role",
            y="count",
            color="role",
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(polish_plotly(fig, height=360), use_container_width=True)
        panel_close()

    if "plan_name" in filtered_df.columns and "amount" in filtered_df.columns:
        c5, c6 = st.columns(2)
        with c5:
            panel_open("Revenue by Plan", "Revenue concentration by subscription plan")
            revenue_df = (
                filtered_df.groupby("plan_name", dropna=True)["amount"]
                .sum()
                .reset_index()
            )
            fig = px.bar(
                revenue_df,
                x="plan_name",
                y="amount",
                color="plan_name",
                color_discrete_sequence=["#4f46e5"],
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(polish_plotly(fig, height=360), use_container_width=True)
            panel_close()
        with c6:
            panel_open(
                " Average Churn by Plan", "Revenue segments with higher churn exposure"
            )
            if "churn_probability" in filtered_df.columns:
                churn_df = (
                    filtered_df.groupby("plan_name", dropna=True)["churn_probability"]
                    .mean()
                    .reset_index()
                )
                fig = px.bar(
                    churn_df,
                    x="plan_name",
                    y="churn_probability",
                    color="plan_name",
                    color_discrete_sequence=["#ef4444"],
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(
                    polish_plotly(fig, height=360), use_container_width=True
                )
            else:
                st.info("No churn prediction data available.")
            panel_close()

elif page == "ML Predictions":
    hero(
        "ML Predictions",
        "Churn probability, segment-level risk analysis, and executive triage of high-risk users.",
    )

    if "churn_probability" not in filtered_df.columns:
        st.warning("No prediction data found in customer_predictions.")
    else:
        # st.download_button(
        # label="Download CSV",
        # data=filtered_df.to_csv(index=False),
        # file_name="data.csv",
        # mime="text/csv"
        # )
        m1, m2 = st.columns(2)
        with m1:
            panel_open(" Churn Probability Distribution", "Overall model risk spread")
            fig = px.histogram(
                filtered_df,
                x="churn_probability",
                nbins=20,
                color_discrete_sequence=["#4f46e5"],
            )
            st.plotly_chart(polish_plotly(fig, height=380), use_container_width=True)
            panel_close()
        with m2:
            panel_open(
                " Average Churn by Status", "Which status groups carry the most risk"
            )
            churn_status = (
                filtered_df.groupby("status", dropna=True)["churn_probability"]
                .mean()
                .reset_index()
            )
            fig = px.bar(
                churn_status,
                x="status",
                y="churn_probability",
                color="status",
                color_discrete_sequence=["#10b981", "#f59e0b", "#ef4444"],
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(polish_plotly(fig, height=380), use_container_width=True)
            panel_close()

        m3, m4 = st.columns(2)
        with m3:
            panel_open("Average Churn by Role", "Risk concentration by user segment")
            churn_role = (
                filtered_df.groupby("role", dropna=True)["churn_probability"]
                .mean()
                .reset_index()
            )
            fig = px.bar(
                churn_role,
                x="role",
                y="churn_probability",
                color="role",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(polish_plotly(fig, height=380), use_container_width=True)
            panel_close()
        with m4:
            panel_open(" Feature Correlation", "How behavior variables move with churn")
            numeric_cols = [
                c
                for c in [
                    "login_count_7d",
                    "login_count_30d",
                    "failed_login_attempts",
                    "days_since_last_login",
                    "churn_probability",
                ]
                if c in filtered_df.columns
            ]
            if numeric_cols:
                corr = filtered_df[numeric_cols].corr()
                fig = px.imshow(
                    corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r"
                )
                fig.update_layout(coloraxis_colorbar=dict(title=""))
                st.plotly_chart(
                    polish_plotly(fig, height=430), use_container_width=True
                )
            else:
                st.info("Not enough numeric columns for correlation.")
            panel_close()

        panel_open(
            " Top Risky Users", "Users that should be reviewed or retained first"
        )
        display_cols = [
            c
            for c in [
                "full_name",
                "email",
                "role",
                "status",
                "churn_probability",
                "risk_level",
                "top_reason_1",
                "top_reason_2",
                "top_reason_3",
            ]
            if c in filtered_df.columns
        ]
        top_risk = filtered_df.sort_values("churn_probability", ascending=False)[
            display_cols
        ].head(15)
        st.dataframe(top_risk, use_container_width=True)
        panel_close()

else:  # Explainability
    hero(
        "Explainability",
        "Understand what drives the churn model globally and for individual users.",
    )

    model = st.session_state.get("model")

    if model is None:
        st.warning(
            "Model file not found. Train the model first so best_model.pkl exists."
        )
    elif filtered_df.empty:
        st.warning("customer_features is empty or no rows after filtering.")
    else:
        try:
            MAX_EXPLAIN_ROWS = 200
            X_full = prepare_features(filtered_df)
            X = X_full.head(MAX_EXPLAIN_ROWS).copy()
            explain_df = filtered_df.head(MAX_EXPLAIN_ROWS).copy()

            if len(X) == 0:
                st.warning("No rows available after filtering.")
            else:
                with st.spinner("Generating SHAP explainability..."):
                    explainer = get_explainer(model)
                    shap_values = get_shap_values(explainer, X)

                    if (
                        hasattr(shap_values, "values")
                        and len(shap_values.values.shape) == 3
                    ):
                        shap_values_to_plot = shap_values[:, :, 1]
                    else:
                        shap_values_to_plot = shap_values

                shap_vals = (
                    shap_values_to_plot.values
                    if hasattr(shap_values_to_plot, "values")
                    else shap_values_to_plot
                )

                st.info(
                    f"Showing explainability for the first {len(X)} rows to keep the page responsive."
                )

                e1, e2 = st.columns(2)
                with e1:
                    panel_open(
                        "Global Feature Importance",
                        "Overall model drivers across the filtered population",
                    )
                    mean_shap = pd.DataFrame(
                        {
                            "feature": X.columns,
                            "importance": abs(shap_vals).mean(axis=0),
                        }
                    ).sort_values("importance", ascending=True)

                    fig = px.bar(
                        mean_shap,
                        x="importance",
                        y="feature",
                        orientation="h",
                        color="importance",
                        color_continuous_scale="Blues",
                    )
                    fig.update_layout(coloraxis_showscale=False)
                    st.plotly_chart(
                        polish_plotly(fig, height=400), use_container_width=True
                    )
                    panel_close()

                with e2:
                    panel_open(
                        "SHAP Summary Plot", "Feature impact distribution across users"
                    )
                    shap_df = pd.DataFrame(shap_vals, columns=X.columns)
                    shap_long = shap_df.melt(
                        var_name="feature", value_name="shap_value"
                    )
                    shap_long["direction"] = shap_long["shap_value"].apply(
                        lambda v: "Increases churn" if v > 0 else "Decreases churn"
                    )
                    fig = px.strip(
                        shap_long,
                        x="shap_value",
                        y="feature",
                        color="direction",
                        color_discrete_map={
                            "Increases churn": "#ef4444",
                            "Decreases churn": "#10b981",
                        },
                    )
                    fig.update_traces(jitter=0.4, marker=dict(size=4, opacity=0.6))
                    fig.add_vline(x=0, line_width=1, line_color="#94a3b8")
                    st.plotly_chart(
                        polish_plotly(fig, height=400), use_container_width=True
                    )
                    panel_close()

                panel_open(
                    "Local Explanation",
                    "Inspect one user and see what pushed the prediction up or down",
                )
                user_index = st.slider("Select user index", 0, len(X) - 1, 0)

                show_cols = [
                    c
                    for c in [
                        "full_name",
                        "email",
                        "role",
                        "status",
                        "churn_probability",
                    ]
                    if c in explain_df.columns
                ]
                if show_cols:
                    st.markdown(
                        '<div class="small-note">Selected user context</div>',
                        unsafe_allow_html=True,
                    )
                    st.dataframe(
                        explain_df.iloc[[user_index]][show_cols],
                        use_container_width=True,
                    )

                user_shap = shap_vals[user_index]
                waterfall_df = pd.DataFrame(
                    {"feature": X.columns, "shap_value": user_shap}
                ).sort_values("shap_value")

                waterfall_df["color"] = waterfall_df["shap_value"].apply(
                    lambda v: "#ef4444" if v > 0 else "#10b981"
                )

                fig = px.bar(
                    waterfall_df,
                    x="shap_value",
                    y="feature",
                    orientation="h",
                    color="color",
                    color_discrete_map="identity",
                )
                fig.add_vline(x=0, line_width=1, line_color="#94a3b8")
                fig.update_layout(showlegend=False)
                st.plotly_chart(
                    polish_plotly(fig, height=400), use_container_width=True
                )
                panel_close()

        except Exception as e:
            st.error(f"Explainability failed: {e}")
