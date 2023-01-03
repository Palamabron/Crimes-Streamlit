import re

import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.decomposition import PCA
import plotly.express as px
from sklearn.impute import SimpleImputer
import plotly.graph_objs as go
from plotly.offline import iplot
import json
from pandas.io.json import json_normalize


def highlight_cells(s: pd.Series, true_css: str, false_css: str = '') -> np.ndarray:
    return np.where(s > s.mean(), true_css, false_css)


def prepare_data(df):
    ignore = ['communityname', 'state', 'countyCode', 'communityCode', 'fold']
    df = df.drop(columns=ignore, axis=1)
    label_encoder = LabelEncoder()
    df['LemasGangUnitDeploy'] = label_encoder.fit_transform(df['LemasGangUnitDeploy'])
    imp = SimpleImputer(missing_values=np.nan)
    idf = pd.DataFrame(imp.fit_transform(df))
    idf.columns = df.columns
    idf.index = df.index
    return idf.copy()


@st.cache
def load_data():
    columns = [
        'communityname', 'state', 'countyCode', 'communityCode', 'fold', 'population', 'householdsize', 'racepctblack',
        'racePctWhite', 'racePctAsian', 'racePctHisp', 'agePct12t21', 'agePct12t29', 'agePct16t24', 'agePct65up',
        'numbUrban', 'pctUrban', 'medIncome', 'pctWWage', 'pctWFarmSelf', 'pctWInvInc', 'pctWSocSec', 'pctWPubAsst',
        'pctWRetire', 'medFamInc', 'perCapInc', 'whitePerCap', 'blackPerCap', 'indianPerCap', 'AsianPerCap',
        'OtherPerCap', 'HispPerCap', 'NumUnderPov', 'PctPopUnderPov', 'PctLess9thGrade', 'PctNotHSGrad', 'PctBSorMore',
        'PctUnemployed', 'PctEmploy', 'PctEmplManu', 'PctEmplProfServ', 'PctOccupManu', 'PctOccupMgmtProf',
        'MalePctDivorce', 'MalePctNevMarr', 'FemalePctDiv', 'TotalPctDiv', 'PersPerFam', 'PctFam2Par', 'PctKids2Par',
        'PctYoungKids2Par', 'PctTeen2Par', 'PctWorkMomYoungKids', 'PctWorkMom', 'NumKidsBornNeverMar',
        'PctKidsBornNeverMar', 'NumImmig', 'PctImmigRecent', 'PctImmigRec5', 'PctImmigRec8', 'PctImmigRec10',
        'PctRecentImmig', 'PctRecImmig5', 'PctRecImmig8', 'PctRecImmig10', 'PctSpeakEnglOnly', 'PctNotSpeakEnglWell',
        'PctLargHouseFam', 'PctLargHouseOccup', 'PersPerOccupHous', 'PersPerOwnOccHous', 'PersPerRentOccHous',
        'PctPersOwnOccup', 'PctPersDenseHous', 'PctHousLess3BR', 'MedNumBR', 'HousVacant', 'PctHousOccup',
        'PctHousOwnOcc', 'PctVacantBoarded', 'PctVacMore6Mos', 'MedYrHousBuilt', 'PctHousNoPhone', 'PctWOFullPlumb',
        'OwnOccLowQuart', 'OwnOccMedVal', 'OwnOccHiQuart', 'OwnOccQrange', 'RentLowQ', 'RentMedian', 'RentHighQ',
        'RentQrange', 'MedRent', 'MedRentPctHousInc', 'MedOwnCostPctInc', 'MedOwnCostPctIncNoMtg', 'NumInShelters',
        'NumStreet', 'PctForeignBorn', 'PctBornSameState', 'PctSameHouse85', 'PctSameCity85', 'PctSameState85',
        'LemasSwornFT', 'LemasSwFTPerPop', 'LemasSwFTFieldOps', 'LemasSwFTFieldPerPop', 'LemasTotalReq',
        'LemasTotReqPerPop', 'PolicReqPerOffic', 'PolicPerPop', 'RacialMatchCommPol', 'PctPolicWhite', 'PctPolicBlack',
        'PctPolicHisp', 'PctPolicAsian', 'PctPolicMinor', 'OfficAssgnDrugUnits', 'NumKindsDrugsSeiz',
        'PolicAveOTWorked', 'LandArea', 'PopDens', 'PctUsePubTrans', 'PolicCars', 'PolicOperBudg',
        'LemasPctPolicOnPatr', 'LemasGangUnitDeploy', 'LemasPctOfficDrugUn', 'PolicBudgPerPop', 'murders', 'murdPerPop',
        'rapes', 'rapesPerPop', 'robberies', 'robbbPerPop', 'assaults', 'assaultPerPop', 'burglaries', 'burglPerPop',
        'larcenies', 'larcPerPop', 'autoTheft', 'autoTheftPerPop', 'arsons', 'arsonsPerPop', 'ViolentCrimesPerPop',
        'nonViolPerPop'
    ]
    data = pd.read_csv('https://raw.githubusercontent.com/Palamabron/PRO1D/main/CommViolPredUnnormalizedData.txt',
                       names=columns, na_values=["?"])
    processed_data = pd.read_csv("https://raw.githubusercontent.com/Palamabron/PRO1D/main/TrainProc2.csv",
                                 on_bad_lines='skip', sep=";", index_col="Column1")
    rawData = pd.DataFrame(data)
    return rawData, processed_data


def data_analysis_page():
    data_sidebar = st.sidebar.selectbox(
        "Data Analysis",
        ("Preprocessed", "Processed")
    )
    target = [
        'murders', 'murdPerPop', 'rapes', 'rapesPerPop', 'robberies', 'robbbPerPop',
        'assaults', 'assaultPerPop', 'burglaries', 'burglPerPop', 'larcenies',
        'larcPerPop', 'autoTheft', 'autoTheftPerPop', 'arsons', 'arsonsPerPop',
        'ViolentCrimesPerPop', 'nonViolPerPop'
    ]
    rawData, processed_data = load_data()
    primarilyPreData = prepare_data(rawData)
    styledData = rawData.head(50).copy()
    numeric_columns = styledData.select_dtypes(include=[np.float]).columns.tolist()
    if data_sidebar == "Preprocessed":
        st.dataframe(styledData.style.apply(highlight_cells, true_css='color:green', false_css='color:red',
                                            subset=target).format(subset=numeric_columns, formatter="{:.2f}"))
        geoPlot(rawData)
        corr_plot(primarilyPreData)
        most_corr_df(primarilyPreData)
        get_vif_scores(primarilyPreData)
        pca_plot(primarilyPreData)
        tSNE_plot(primarilyPreData)
    else:
        st.write("After processing")
        numeric_columns = processed_data.select_dtypes(include=[np.float]).columns.tolist()
        styledData = processed_data.head(50).sort_index(ascending=True).copy()
        st.dataframe(
            styledData.style
            .apply(highlight_cells, true_css='color:green', false_css='color:red',
                   subset=['ViolentCrimesPerPop'])
            .format(subset=numeric_columns, formatter="{:.2f}")
        )
        corr_plot(processed_data)
        most_corr_df(processed_data)
        get_vif_scores(processed_data)
        pca_plot(processed_data)
        tSNE_plot(processed_data)


def corr_plot(df):
    fig = plt.figure(figsize=(12, 12))
    sns.heatmap(df.corr(), mask=np.triu(df.corr()))
    st.pyplot(fig)


def pca_plot(df):
    df = df.sample(frac=0.8)
    n_components = 3
    st.write("PCA PLOT")
    pca = PCA(n_components=n_components)
    features = df.loc[:, :'ViolentCrimesPerPop']
    components = pca.fit_transform(features)

    total_var = pca.explained_variance_ratio_.sum() * 100
    # labels = {str(i): f"PC {i + 1}" for i in range(n_components)}
    # labels['color'] = 'ViolentCrimesPerPop'
    fig = px.scatter_3d(
        components, x=0, y=1, z=2,
        color=df.ViolentCrimesPerPop,
        labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'},
        title=f'Total Explained Variance: {total_var:.2f}%',
        color_continuous_scale=px.colors.sequential.Sunset
    )
    # fig.update_traces(diagonal_visible=False)
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    n_components = 2
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(features)
    fig = px.scatter(
        components, x=0, y=1,
        color=df.ViolentCrimesPerPop,
        labels={'color': 'ViolentCrimesPerPop'},
        color_continuous_scale=px.colors.sequential.Sunset
    )
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)


def tSNE_plot(df):
    df = df.sample(frac=0.8)
    tsne = TSNE(n_components=3, random_state=0)
    projections = tsne.fit_transform(df)
    st.write("TSNE_plot")
    fig = px.scatter_3d(
        projections, x=0, y=1, z=2,
        color=df.ViolentCrimesPerPop,
        labels={'color': 'ViolentCrimesPerPop'},
        color_continuous_scale=px.colors.sequential.Sunset
    )
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    tsne = TSNE(n_components=2, random_state=0)
    projections = tsne.fit_transform(df)

    fig = px.scatter(
        projections, x=0, y=1,
        color=df.ViolentCrimesPerPop,
        labels={'color': 'ViolentCrimesPerPop'},
        color_continuous_scale=px.colors.sequential.Sunset
    )

    st.plotly_chart(fig, theme="streamlit", use_container_width=True)


def most_corr_df(df):
    corr_mat = pd.DataFrame(df.corr(method='pearson').abs())
    upper_corr_mat = corr_mat.where(
        np.triu(np.ones(corr_mat.shape), k=1).astype(np.bool)
    )
    unique_corr_pairs = upper_corr_mat.unstack().dropna()
    sorted_mat = unique_corr_pairs.sort_values(kind='quicksort', ascending=False).head(20)
    result = pd.DataFrame(sorted_mat)
    # print(result.columns.values)
    result.rename(columns={0: "correlation"}, inplace=True)
    st.dataframe(result)
    result = pd.DataFrame(unique_corr_pairs.sort_values(kind='quicksort', ascending=False).loc[lambda x: (x > 0.9)])
    st.write(f"number of columns correlated above 0.9: {len(result.index)}")


def get_vif_scores(df):
    X = add_constant(df)
    vif_data = pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
                         index=X.columns)
    st.dataframe(
        vif_data.rename("VIF score").sort_values(kind='quicksort', ascending=False).loc[lambda x: (x != np.inf)])
    st.write(f"number of columns with vif score above 50: {len(vif_data.loc[lambda x: (x > 50)].index)}")


def geoPlot(df):
    df_state = df.groupby('state').agg({'ViolentCrimesPerPop': 'mean', 'nonViolPerPop': 'mean'})[
        ['ViolentCrimesPerPop', 'nonViolPerPop']].reset_index()
    df_state = df_state.fillna(0)
    # print(df_state)
    # Aggregate view of Non-Violent Crimes by State
    data1 = dict(type='choropleth',
                 colorscale='Viridis',
                 autocolorscale=False,
                 locations=df_state['state'],
                 locationmode='USA-states',
                 z=df_state['nonViolPerPop'].astype(float),
                 colorbar={'title': 'non-Violent Crimes(Per-100K-Pop)'}
                 )
    layout1 = dict(
        title='Aggregate view of non-Violent Crimes Per 100K Population',
        geo=dict(
            scope='usa',
            projection=dict(type='albers usa'),
            showlakes=True,
            lakecolor='rgb(85,173,240)'),
    )

    fig1 = go.Figure(data=[data1], layout=layout1)
    st.plotly_chart(fig1, theme="streamlit", use_container_width=True)

    # Aggregate view of Violent Crimes by State
    data2 = dict(type='choropleth',
                 autocolorscale=False,
                 colorscale="Earth",
                 locations=df_state['state'],
                 locationmode='USA-states',
                 z=df_state['ViolentCrimesPerPop'].astype('float'),
                 colorbar={'title': 'Violent Crimes(Per-100K-Pop)'}
                 )
    layout2 = dict(
        title='Aggregate view of Violent Crimes Per 100K Population across US',
        geo=dict(
            scope='usa',
            projection=dict(type='albers usa'),
            showlakes=True,
            lakecolor='rgb(85,173,240)'),
    )

    fig2 = go.Figure(data=[data2], layout=layout2)
    st.plotly_chart(fig2, theme="streamlit", use_container_width=True)
