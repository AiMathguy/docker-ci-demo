import streamlit as st
import matplotlib.pyplot as plt
import shap


def plot_global_importance(shap_values, X, plot_type="bar"):
    fig = plt.figure(figsize=(10, 5))
    shap.summary_plot(shap_values, X, plot_type=plot_type, show=False)
    st.pyplot(fig)
    plt.clf()


def plot_summary(shap_values, X):
    fig = plt.figure(figsize=(10, 5))
    shap.summary_plot(shap_values, X, show=False)
    st.pyplot(fig)
    plt.clf()


def plot_waterfall(shap_values, index):
    fig = plt.figure(figsize=(10, 5))
    shap.plots.waterfall(shap_values[index], show=False)
    st.pyplot(fig)
    plt.clf()
