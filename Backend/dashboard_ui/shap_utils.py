import shap
import streamlit as st


@st.cache_resource
def get_explainer(_model):
    """Return a SHAP TreeExplainer for the model."""
    return shap.TreeExplainer(_model)


def get_shap_values(explainer, X):
    """Return SHAP values for the given data."""
    return explainer(X)
