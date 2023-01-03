import streamlit as st
import pandas as pd
from joblib import load

import DataAnalysis
import LivePrediction
import ModelAnalysis


@st.cache(allow_output_mutation=True)
def load_models():
    et = load(r'D:\PythonProjects\streamlit\crimes\manualModels\m_et.joblib')
    gbr = load(r'D:\PythonProjects\streamlit\crimes\manualModels\m_gbr.joblib')
    huber = load(r'D:\PythonProjects\streamlit\crimes\manualModels\m_huber.joblib')
    lightgbm = load(r'D:\PythonProjects\streamlit\crimes\manualModels\m_lightgbm.joblib')
    rf = load(r'D:\PythonProjects\streamlit\crimes\manualModels\m_rf.joblib')
    stack = load(r'D:\PythonProjects\streamlit\crimes\manualModels\m_stack.joblib')
    return et, gbr, huber, lightgbm, rf, stack


def run():
    menu_sidebar = st.sidebar.selectbox(
        "Menu",
        ("Data Analysis", "Models analysis", "Live Prediction"),
        key=1
    )

    et, gbr, huber, lightgbm, rf, stack = load_models()

    if menu_sidebar == "Live Prediction":
        LivePrediction.live_prediction_page(et, gbr, huber, lightgbm, rf, stack)
    elif menu_sidebar == "Models analysis":
        ModelAnalysis.models_analysis_page(et, gbr, huber, lightgbm, rf, stack)
    else:
        DataAnalysis.data_analysis_page()


if __name__ == '__main__':
    st.set_option('deprecation.showPyplotGlobalUse', False)
    pd.options.display.float_format = "{:,.2f}".format
    run()
