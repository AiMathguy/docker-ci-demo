import os
import joblib
import shap
import streamlit as st

MODEL_PATH = "churn_model.pkl"


@st.cache_resource
def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def compute_shap_values(_model, X):
    explainer = shap.TreeExplainer(_model)
    return explainer(X)
