import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from joblib import load

import DataAnalysis
import LivePrediction
import ModelAnalysis
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.decomposition import PCA
import plotly.express as px
from sklearn.impute import SimpleImputer


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


# def prepare_data(df):
#     ignore = ['communityname', 'state', 'countyCode', 'communityCode', 'fold']
#     df = df.drop(columns=ignore, axis=1)
#     label_encoder = LabelEncoder()
#     df['LemasGangUnitDeploy'] = label_encoder.fit_transform(df['LemasGangUnitDeploy'])
#     imp = SimpleImputer(missing_values=np.nan)
#     idf = pd.DataFrame(imp.fit_transform(df))
#     idf.columns = df.columns
#     idf.index = df.index
#     return idf.copy()


if __name__ == '__main__':
    st.set_option('deprecation.showPyplotGlobalUse', False)
    pd.options.display.float_format = "{:,.2f}".format
    run()
