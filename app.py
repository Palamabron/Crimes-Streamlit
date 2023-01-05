import streamlit as st
import pandas as pd
from joblib import load

import DataAnalysis
import LivePrediction
import ModelAnalysis


@st.cache(allow_output_mutation=True)
def load_models():
    customPath = r''
    et = load(customPath + 'm_et.joblib')
    gbr = load(customPath + 'm_gbr.joblib')
    huber = load(customPath + 'm_huber.joblib')
    lightgbm = load(customPath + 'm_lightgbm.joblib')
    rf = load(customPath + 'm_rf.joblib')
    stack = load(customPath + 'm_stack.joblib')
    return et, gbr, huber, lightgbm, rf, stack


def run():
    menu_sidebar = st.sidebar.selectbox(
        "Menu",
        ("Data Analysis", "Models analysis", "Live Prediction"),
        key=1
    )

    models = load_models()

    if menu_sidebar == "Live Prediction":
        LivePrediction.live_prediction_page(models)
    elif menu_sidebar == "Models analysis":
        ModelAnalysis.models_analysis_page(models)
    else:
        DataAnalysis.data_analysis_page()


if __name__ == '__main__':
    st.set_option('deprecation.showPyplotGlobalUse', False)
    pd.options.display.float_format = "{:,.2f}".format
    run()
