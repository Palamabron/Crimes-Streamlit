import numpy as np
from matplotlib import pyplot as plt
import streamlit as st
from joblib import load
from yellowbrick.regressor import ResidualsPlot
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, mean_squared_log_error, \
    mean_absolute_percentage_error, median_absolute_error
import pandas as pd
import shap
import seaborn as sns

import LivePrediction


def load_data():
    processed_data = pd.read_csv("https://raw.githubusercontent.com/Palamabron/PRO1D/main/TrainProc2.csv",
                                 on_bad_lines='skip', sep=";", index_col="Column1")
    processed_data_test = pd.read_csv("https://raw.githubusercontent.com/Palamabron/PRO1D/main/Test2.csv",
                                      on_bad_lines='skip', sep=";", index_col="Column1")
    X_train, y_train = processed_data.iloc[:, :-1], processed_data["ViolentCrimesPerPop"]
    X_test, y_test = processed_data_test.iloc[:, :-1], processed_data_test["ViolentCrimesPerPop"]
    return X_train, y_train, X_test, y_test


def show_residuals(model, X_train, y_train, X_test, y_test):
    try:
        visualizer = ResidualsPlot(model)
        visualizer.fit(X_train, y_train)
        visualizer.score(X_test, y_test)
        st.pyplot(visualizer.finalize())
    except:
        pass


def show_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    metrics = {
        "R2": r2_score(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "MSE": mean_squared_error(y_test, y_pred),
        # "MSLE": mean_squared_log_error(y_test, y_pred),
        "MAPE": mean_absolute_percentage_error(y_test, y_pred),
        "MedAE": median_absolute_error(y_test, y_pred),
    }
    metrics_pd = pd.DataFrame(metrics, index=[type(model).__name__])
    st.dataframe(metrics_pd.T)


def show_shap(model, X_train):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    st.header("Feature Importance")
    plt.title("Feature Importance based on SHAP Values")
    shap.summary_plot(shap_values, X_train, feature_names=X_train.columns)
    st.pyplot(bbox_inches='tight')
    st.write('---')
    plt.title('Feature importance based on SHAP values (Bar)')
    shap.summary_plot(shap_values, X_train, plot_type="bar", feature_names=X_train.columns)
    st.pyplot(bbox_inches='tight')


def analyze_model(model, X_train, y_train, X_test, y_test):
    show_residuals(model, X_train, y_train, X_test, y_test)
    st.write("---")
    show_metrics(model, X_test, y_test)
    st.write("---")


def analyze_stack_model(model, X_train, y_train, X_test, y_test):
    estimators = model.estimators
    predictions = []
    for estimator in estimators:
        predictions.append((estimator[0], estimator[1].predict(X_test)))
    pred_df = pd.DataFrame(data={prediction[0]: prediction[1] for prediction in predictions})
    pred_df["stack"] = model.predict(X_test)
    pred_df["target"] = y_test.values
    st.dataframe(pred_df)
    st.write("---")
    fig, ax = plt.subplots()
    # plt.xlim(-500, 2500)
    sns.kdeplot(pred_df)
    st.pyplot(fig)


def models_analysis_page(et, gbr, huber, lightgbm, rf, stack):
    model_sidebar = st.sidebar.selectbox(
        "Choose a model",
        ("Extra-trees Regressor", "Gradient Boosting Regressor", "Huber Regressor", "Light Gradient Boosting Machine",
         "Random Trees Regressor", "Stacked Regressor"),
        key=10
    )
    # stack, et, gbr, huber, lightgbm, rf = load_models()
    X_train, y_train, X_test, y_test = load_data()
    if model_sidebar == "Extra-trees Regressor":
        analyze_model(et, X_train, y_train, X_test, y_test)
        show_shap(et, X_train)
    elif model_sidebar == "Gradient Boosting Regressor":
        analyze_model(gbr, X_train, y_train, X_test, y_test)
        show_shap(gbr, X_train)
    elif model_sidebar == "Huber Regressor":
        analyze_model(huber, X_train, y_train, X_test, y_test)
    elif model_sidebar == "Light Gradient Boosting Machine":
        analyze_model(lightgbm, X_train, y_train, X_test, y_test)
        show_shap(lightgbm, X_train)
    elif model_sidebar == "Random Trees Regressor":
        analyze_model(rf, X_train, y_train, X_test, y_test)
        show_shap(rf, X_train)
    else:
        analyze_model(stack, X_train, y_train, X_test, y_test)
        analyze_stack_model(stack, X_train.values, y_train, X_test.values, y_test)

