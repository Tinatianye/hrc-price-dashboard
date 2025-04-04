import streamlit as st

# Import libraries
import numpy as np 
import pandas as pd 
import seaborn as sns

import statsmodels.api as sm
from statsmodels.tsa.api import VAR
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tools.eval_measures import rmse, aic

import plotly.express as px 

import warnings
warnings.filterwarnings("ignore")

from datetime import datetime  
from dateutil.relativedelta import relativedelta  

# Streamlit configuration
st.set_page_config(
    page_title="HRC Price Predict Model Dashboard",
    page_icon="⭕",
    layout="wide"
)

st.markdown('''
    <h1 style="font-size: 25px; color: white; text-align: center; background: #0E539A; border-radius: .5rem; margin-bottom: 1rem;">
    HRC Price Predict Model Dashboard
    </h1>''', unsafe_allow_html=True)

@st.cache_data
def get_dd():
    fp = 'hrc_price_CN_JP.csv'  # 更换为更贴近的默认数据源
    return pd.read_csv(fp)

col = st.columns([1.2, 3])
col1 = col[1].columns(3)
col0 = col[0].columns(2)

# 参数输入
sea_freight = col0[0].number_input("**Sea Freight**", value=10)
exchange_rate = col0[1].number_input("**Exchange Rate (Rs/USD)**", value=0.1)

# 数据准备
df = get_dd()
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(by="Date").dropna()
df.set_index('Date', inplace=True)

# 用户选择参数
maxlags = col1[0].number_input("**Maxlags**", value=12, min_value=1, max_value=100)
months = col1[1].number_input(f"**Months ahead (Started in {df.index.tolist()[-1].strftime('%Y-%m-%d')})**", value=17, min_value=1, max_value=50)
country = col1[2].multiselect("Please choose country", ["China", "Japan"], ["China", "Japan"])

# 差分处理
final_df_differenced = df.diff().dropna()
model = VAR(final_df_differenced)
order_results = model.select_order(maxlags=maxlags)
st.write("### VAR Order Selection Criteria")
st.dataframe(order_results.summary().as_text())

model_fitted = model.fit(maxlags)
lag_order = model_fitted.k_ar
forecast_input = final_df_differenced.values[-lag_order:]

# 还原预测值
def invert_transformation(df_train, df_forecast, second_diff=False):
    df_fc = df_forecast.copy()
    columns = df_train.columns
    for col in columns:
        if second_diff:
            df_fc[str(col)+'_1d'] = (df_train[col].iloc[-1]-df_train[col].iloc[-2]) + df_fc[str(col)+'_2d'].cumsum()
        df_fc[str(col)+'_forecast'] = df_train[col].iloc[-1] + df_fc[str(col)+'_1d'].cumsum()
    return df_fc

# 开始预测
fc = model_fitted.forecast(y=forecast_input, steps=months)
fc_period = pd.date_range(
    start=df.index.tolist()[-1] + relativedelta(months=1),
    periods=months, freq='MS')
df_forecast = pd.DataFrame(fc, index=fc_period, columns=df.columns + '_1d')
df_forecast_processed = invert_transformation(df, df_forecast)

# 预测结果展示
df_res = df_forecast_processed[["China HRC (FOB, $/t)_forecast", "Japan HRC (FOB, $/t)_forecast"]]
df_res = df_res.reset_index()
last_row = df_res.iloc[-1]

month = last_row[0]
china_hrc = last_row[1]
japan_hrc = last_row[2]

# 拼接历史和预测
def build_plot_data(x):
    d1 = df[[f'{x} HRC (FOB, $/t)']].copy()
    d1.columns = ["HRC (FOB, $/t)"]
    d1["t"] = f"{x} HRC (FOB, $/t)"
    d2 = df_forecast_processed[[f"{x} HRC (FOB, $/t)_forecast"]].copy()
    d2.columns = ["HRC (FOB, $/t)"]
    d2["t"] = f"{x} HRC (FOB, $/t)_forecast"
    return pd.concat([d1, d2])

all_data = pd.concat([build_plot_data(i) for i in country])
all_data = all_data[all_data.index > df.index[-12]]  # 最近一年数据

fig = px.line(all_data, x=all_data.index, y="HRC (FOB, $/t)", color="t")
fig.update_traces(hovertemplate='%{y}')
fig.update_layout(
    title={
        'text': "/".join(country) + " Forecasting HRC prices",
        'x': 0.5,
        'y': 0.96,
        'xanchor': 'center'
    },
    margin=dict(t=30),
    height=500,
    legend=dict(title="", yanchor="top", y=0.99, xanchor="center", x=0.5, orientation="h"),
    xaxis_title="Date",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)
col[1].plotly_chart(fig, use_container_width=True)

# 计算各国 land price
land_price_data = {
    "China": {
        "hrc": china_hrc,
        "customs": 7.5
    },
    "Japan": {
        "hrc": japan_hrc,
        "customs": 0.0
    }
}

for c in country:
    data = land_price_data[c]
    df_price = pd.DataFrame({
        "Factors": [f"HRC FOB {c}", "Sea Freight", "Basic Customs Duty (%)", "LC charges & Port Charges", "Exchange Rate (Rs/USD)", "Freight from port to city"],
        "Value": [data['hrc'], sea_freight, data['customs'], 10, exchange_rate, 500]
    })
    land_price = exchange_rate * (10 + 1.01 * (data['hrc'] + sea_freight) * (1 + 1.1 * data['customs'])) + 500
    col[0].write(f"**{c}**")
    col[0].dataframe(df_price, hide_index=True)
    col[0].write(f'{c} land price is: **{round(land_price)}**')
