# Example origin:
# https://habr.com/ru/post/664076/

# Source code:
# https://github.com/sozykin/streamlit_demo_app

# to run use ".\" before filename: 
# streamlit run .\gpu_prediction_app.py

import streamlit as st
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import curve_fit

# ---------------------------------------------------------------------
def plot_GPU_by_year(dataset):
    fig = plt.figure(figsize=(16,8))
    sns.set_style("whitegrid")
    sns.countplot(x="Release_Year", data=dataset);
    plt.title('Розподіл графічних процесорів за роком випуску', fontsize=20, fontweight='bold', y=1.05,)
    plt.ylabel('Кількість графічних процесорів', fontsize=15)
    plt.xlabel('Рік випуску', fontsize=15)

    st.write('Розподіл графічних процесорів за роком випуску')
    st.pyplot(fig)

def plot_GPUmemory_by_year(dataset):
    fig = plt.figure(figsize=(16,10))
    sns.set_style("whitegrid")
    plt.title('Розподіл обсягу памяті процесорів за роком випуску', fontsize=20, fontweight='bold', y=1.05,)
    plt.xlabel('Рік випуску', fontsize=15)
    plt.ylabel('Обсяг памяті', fontsize=15)

    years = dataset["Release"].values
    memory = dataset["Memory"].values.astype(int)

    plt.scatter(years, memory, edgecolors='black')
    st.write('Розподіл обсягу памяті процесорів за роком випуску')
    st.pyplot(fig)


def show_plots(dataset):   
    plot_GPU_by_year(dataset) 
    plot_GPUmemory_by_year(dataset)

# ---------------------------------------------------------------------
def load_data():
    uploaded_file = st.file_uploader(
        label='Виберіть CSV-файл з даними')
    if uploaded_file is not None:
        dataset = pd.read_csv(uploaded_file)

        dataset['Release_Date'] = dataset['Release_Date'].str[1:-1]
        dataset = dataset[dataset['Release_Date'].str.len()==11]
        dataset['Release_Date'] = pd.to_datetime(dataset['Release_Date'], format='%d-%b-%Y')
        dataset['Release_Year'] = dataset['Release_Date'].dt.year
        dataset['Release_Month'] = dataset['Release_Date'].dt.month
        dataset['Release'] = dataset['Release_Year'] + dataset['Release_Month']/12

        dataset['Memory'] = dataset['Memory'].str[:-3].fillna(0).astype(int)
        
        st.write('**Вхідні дані про характеристики графічних процесорів:**')
        st.write(dataset) 
        
        st.metric("Кількість параметрів графічних процесорів (стопців): ", dataset.shape[1])
        st.metric('Кількість графічних процесорів (рядків): ', dataset.shape[0])
        
        return dataset
    else:
        return None

# ---------------------------------------------------------------------

def calculateMooresValue(x, y_trans):
    return memory_arr_median[0] * 2**((x-y_trans)/2)


def exponentialCurve(x, a, b, c):
    return a*2**((x-c)*b)

# ---------------------------------------------------------------------

st.title("Моделювання змінення характеристик графічних процесорів")
gpu_df = load_data();

if gpu_df is not None:
    show_plots(gpu_df)

    ##### Models #######
    dataset = gpu_df
    # Numpy array that holds unique release year values
    year_arr = dataset.sort_values("Release_Year")['Release_Year'].unique()
    # Numpy array that holds mean values of GPUs memory for each year
    memory_arr_mean = dataset.groupby('Release_Year')['Memory'].mean().values
    # Numpy array that holds median values of GPUs memory for each year
    memory_arr_median = dataset.groupby('Release_Year')['Memory'].median().values


    # Fitting Polynomial Regression to the dataset
    poly_reg_2 = PolynomialFeatures(degree = 2, include_bias=False)
    poly_reg_3 = PolynomialFeatures(degree = 3, include_bias=False)

    X_poly_2 = poly_reg_2.fit_transform(year_arr.reshape(-1, 1))
    X_poly_3 = poly_reg_3.fit_transform(year_arr.reshape(-1, 1))

    lin_reg_2 = LinearRegression()
    lin_reg_3 = LinearRegression()

    lin_reg_2.fit(X_poly_2, memory_arr_mean)
    lin_reg_3.fit(X_poly_3, memory_arr_mean)

    popt, pcov = curve_fit(exponentialCurve, \
        year_arr, memory_arr_mean,  p0=(2, 0.5, 1998))

    year = st.slider('*Вкажіть рік для розрахунку прогноза*', 1998, 2030, 2022)
    
    if st.button('Розрахувати прогнозований обсяг памяті'):
        #year = 2022
        #year_grid = np.arange(min(year), max(year) + 5, 0.1)

        y_pred_moore_law_fitted = exponentialCurve(year, *popt)
        #y_pred_lin_reg_2 = lin_reg_2.predict(poly_reg_2.fit_transform(np.reshape(year, (1, -1))))
        #y_pred_lin_reg_3 = lin_reg_3.predict(poly_reg_3.fit_transform(np.reshape(year, (1, -1))))

        show_msg = "У " + str(year) + " році обсяг пам'яті становитиме, МБ:"
        st.metric(show_msg, y_pred_moore_law_fitted.astype(int))
        #st.metric("Обсяг памяті ", y_pred_lin_reg_2)
        #st.metric("Обсяг памяті ", y_pred_lin_reg_3)




    









