import streamlit as st
import pandas as pd

from ModelAnalysis import load_data


def user_input_features(sliders):
    inputs = {
        "PctKids2Par": sliders[0],
        "PctKidsBornNeverMar": sliders[1],
        "racePctWhite": sliders[2],
        "NumKidsBornNeverMar": sliders[3],
        "racepctblack": sliders[4],
        "pctWInvInc": sliders[5],
        "PctTeen2Par": sliders[6]
    }
    features = pd.DataFrame(inputs, index=[0])
    return features


@st.cache
def get_most_imported_features(model, X_train, y_train):
    model.fit(X_train, y_train)
    feature_importance = []
    for i in range(len(model.feature_importances_)):
        feature_importance.append((model.feature_names_in_[i], model.feature_importances_[i]))
    feature_importance.sort(key=lambda x: x[1])
    result = [feature[0] for feature in feature_importance]
    return result[:8]


def get_columns_average(X_train):
    average_dict = {}
    for col in X_train.columns:
        average_dict[col] = X_train[col].mean()
    avg_pd = pd.DataFrame(average_dict, index=[0])
    return avg_pd


def get_predictions(models, X_train, sliders):
    input_df = user_input_features(sliders)
    result_pd = get_columns_average(X_train)
    for col in input_df.columns:
        result_pd[col] = input_df[col]

    st.dataframe(result_pd.T)

    st.write("# Prediction:")
    predictionDict = dict()

    etPredictions = models[0].predict(result_pd)
    predictionDict["Extra Trees"] = etPredictions

    gbrPredictions = models[1].predict(result_pd)
    predictionDict["Gradient Boosting"] = gbrPredictions

    huberPredictions = models[2].predict(result_pd)
    predictionDict["Huber"] = huberPredictions

    lightgbmPredictions = models[3].predict(result_pd)
    predictionDict["Gradient Boosting"] = lightgbmPredictions

    rfPredictions = models[4].predict(result_pd)
    predictionDict["Random Forest"] = rfPredictions

    stackPredictions = models[5].predict(result_pd)
    predictionDict["Stacking"] = stackPredictions

    prediction_df = pd.DataFrame(predictionDict)
    style = prediction_df.style.hide_index()
    st.write(style.to_html(), unsafe_allow_html=True)


def live_prediction_page(models):
    X_train, y_train, X_test, y_test = load_data()
    PctKids2Par = st.slider("PctKids2Par", float(X_train["PctKids2Par"].min()),
                            float(X_train["PctKids2Par"].max()), float(X_train["PctKids2Par"].mean()),
                            key="PctKids2Par", step=0.1)
    PctKidsBornNeverMar = st.slider("PctKidsBornNeverMar", float(X_train["PctKidsBornNeverMar"].min()),
                                    float(X_train["PctKidsBornNeverMar"].max()),
                                    float(X_train["PctKidsBornNeverMar"].mean()),
                                    key="PctKidsBornNeverMar", step=0.1)
    racePctWhite = st.slider("racePctWhite", float(X_train["racePctWhite"].min()),
                             float(X_train["racePctWhite"].max()), float(X_train["racePctWhite"].mean()),
                             key="racePctWhite", step=0.1)
    NumKidsBornNeverMar = st.slider("NumKidsBornNeverMar", float(X_train["NumKidsBornNeverMar"].min()),
                                    float(X_train["NumKidsBornNeverMar"].max()),
                                    float(X_train["NumKidsBornNeverMar"].mean()),
                                    key="NumKidsBornNeverMar", step=0.1)
    racepctblack = st.slider("racepctblack", float(X_train["racepctblack"].min()),
                             float(X_train["racepctblack"].max()), float(X_train["racepctblack"].mean()),
                             key="racepctblack", step=0.1)
    pctWInvInc = st.slider("pctWInvInc", float(X_train["pctWInvInc"].min()),
                           float(X_train["pctWInvInc"].max()), float(X_train["pctWInvInc"].mean()),
                           key="pctWInvInc", step=0.1)
    PctTeen2Par = st.slider("PctTeen2Par", float(X_train["PctTeen2Par"].min()),
                            float(X_train["PctTeen2Par"].max()), float(X_train["PctTeen2Par"].mean()),
                            key="PctTeen2Par", step=0.1)
    sliders = [PctKids2Par, PctKidsBornNeverMar, racePctWhite, NumKidsBornNeverMar, racepctblack, pctWInvInc, PctTeen2Par]
    get_predictions(models, X_train, sliders)
