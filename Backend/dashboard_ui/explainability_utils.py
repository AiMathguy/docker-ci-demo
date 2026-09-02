import shap
import streamlit as st


@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)


def get_shap_values(explainer, X):
    return explainer(X)
