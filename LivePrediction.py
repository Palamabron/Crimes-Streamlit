import streamlit as st
import pandas as pd
from joblib import load

from ModelAnalysis import load_data


def user_input_features(et, gbr, huber, lightgbm, rf, stack, X_train, y_train, features):
    PctKids2Par = st.slider("PctKids2Par", float(X_train["PctKids2Par"].min()),
                            float(X_train["PctKids2Par"].max()), float(X_train["PctKids2Par"].mean()),
                            on_change=get_predictions, key="PctKids2Par",
                            args=(et, gbr, huber, lightgbm, rf, stack, X_train, y_train, features), step=0.1)
    PctKidsBornNeverMar = st.slider("PctKidsBornNeverMar", float(X_train["PctKidsBornNeverMar"].min()),
                                    float(X_train["PctKidsBornNeverMar"].max()),
                                    float(X_train["PctKidsBornNeverMar"].mean()),
                                    on_change=get_predictions, key="PctKidsBornNeverMar",
                                    args=(et, gbr, huber, lightgbm, rf, stack, X_train, y_train, features), step=0.1)
    racePctWhite = st.slider("racePctWhite", float(X_train["racePctWhite"].min()),
                             float(X_train["racePctWhite"].max()), float(X_train["racePctWhite"].mean()),
                             on_change=get_predictions, key="racePctWhite",
                             args=(et, gbr, huber, lightgbm, rf, stack, X_train, y_train, features), step=0.1)
    NumKidsBornNeverMar = st.slider("NumKidsBornNeverMar", float(X_train["NumKidsBornNeverMar"].min()),
                                    float(X_train["NumKidsBornNeverMar"].max()),
                                    float(X_train["NumKidsBornNeverMar"].mean()),
                                    on_change=get_predictions, key="NumKidsBornNeverMar",
                                    args=(et, gbr, huber, lightgbm, rf, stack, X_train, y_train, features), step=0.1)
    racepctblack = st.slider("racepctblack", float(X_train["racepctblack"].min()),
                             float(X_train["racepctblack"].max()), float(X_train["racepctblack"].mean()),
                             on_change=get_predictions, key="racepctblack",
                             args=(et, gbr, huber, lightgbm, rf, stack, X_train, y_train, features), step=0.1)
    pctWInvInc = st.slider("pctWInvInc", float(X_train["pctWInvInc"].min()),
                           float(X_train["pctWInvInc"].max()), float(X_train["pctWInvInc"].mean()),
                           on_change=get_predictions, key="pctWInvInc",
                           args=(et, gbr, huber, lightgbm, rf, stack, X_train, y_train, features), step=0.1)
    PctTeen2Par = st.slider("PctTeen2Par", float(X_train["PctTeen2Par"].min()),
                            float(X_train["PctTeen2Par"].max()), float(X_train["PctTeen2Par"].mean()),
                            on_change=get_predictions, key="PctTeen2Par",
                            args=(et, gbr, huber, lightgbm, rf, stack, X_train, y_train, features), step=0.1)
    # for column in features:
    #     inputs[column] = st.slider(column, float(X_train[column].min()),
    #                                float(X_train[column].max()), float(X_train[column].mean()),
    #                                on_change=get_predictions, key=str(i),
    #                                args=(et, gbr, huber, lightgbm, rf, stack, X_train, y_train, features), step=0.1)
    #     i += 1
    inputs = {
        "PctKids2Par": PctKids2Par,
        "PctKidsBornNeverMar": PctKidsBornNeverMar,
        "racePctWhite": racePctWhite,
        "NumKidsBornNeverMar": NumKidsBornNeverMar,
        "racepctblack": racepctblack,
        "pctWInvInc": pctWInvInc,
        "PctTeen2Par": PctTeen2Par
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
    # st.write(f"{feature_importance}")
    return result[:8]


def get_columns_average(X_train):
    average_dict = {}
    for col in X_train.columns:
        average_dict[col] = X_train[col].mean()
    avg_pd = pd.DataFrame(average_dict, index=[0])
    return avg_pd


def get_predictions(et, gbr, huber, lightgbm, rf, stack, X_train, y_train, features):
    input_df = user_input_features(et, gbr, huber, lightgbm, rf, stack, X_train, y_train, features)

    result_pd = get_columns_average(X_train)
    for col in input_df.columns:
        result_pd[col] = input_df[col]

    st.dataframe(result_pd.T)

    st.write("# Prediction:")
    pred_dict = dict()

    et_pred = et.predict(result_pd)
    pred_dict["Extra Trees"] = et_pred

    gbr_pred = gbr.predict(result_pd)
    pred_dict["Gradient Boosting"] = gbr_pred

    huber_pred = huber.predict(result_pd)
    pred_dict["Huber"] = huber_pred

    lightgbm_pred = lightgbm.predict(result_pd)
    pred_dict["Gradient Boosting"] = lightgbm_pred

    rf_pred = rf.predict(result_pd)
    pred_dict["Random Forest"] = rf_pred

    stack_pred = stack.predict(result_pd)
    pred_dict["Stacking"] = stack_pred

    pred_df = pd.DataFrame(pred_dict)
    style = pred_df.style.hide_index()
    st.write(style.to_html(), unsafe_allow_html=True)
    # st.dataframe(pred_df)


def live_prediction_page(et, gbr, huber, lightgbm, rf, stack):
    X_train, y_train, X_test, y_test = load_data()
    most_imported_features = get_most_imported_features(et, X_train, y_train)
    # st.write(f"{most_imported_features}")
    get_predictions(et, gbr, huber, lightgbm, rf, stack, X_train, y_train, most_imported_features)
