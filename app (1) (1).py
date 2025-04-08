import streamlit as st
import numpy as np 
import pandas as pd 
import plotly.graph_objects as go
import warnings
from statsmodels.tsa.api import VAR
from datetime import datetime  
from dateutil.relativedelta import relativedelta  

warnings.filterwarnings("ignore")

st.set_page_config(    
    page_title="HRC Price Predict Model Dashboard",
    page_icon="⭕",
    layout="wide"
)

@st.cache_data
def get_dd():
    return pd.read_csv('hrc_price_CN_JP.csv')

st.markdown(f'''
    <h1 style="font-size: 25px; color: white; text-align: center; background: #0E539A; border-radius: .5rem; margin-bottom: 1rem;">
    HRC Price Predict Model Dashboard
    </h1>''', unsafe_allow_html=True)

col = st.columns([1.2, 3])
col1 = col[1].columns(3)
col0 = col[0].columns(2)

# Input
df = get_dd()
sea_freight = col0[0].number_input("**Sea Freight**", value=10)
exchange_rate = col0[1].number_input("**Exchange Rate (Rs/USD)**", value=0.1)
upside_pct = col0[0].number_input("**Upside (%)**", value=10)
downside_pct = col0[1].number_input("**Downside (%)**", value=10)

# Prepare

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(by="Date")
final_df = df.dropna()
final_df.set_index('Date', inplace=True)

maxlags = col1[0].number_input("**Maxlags**", value=12, min_value=1, max_value=100)
months = col1[1].number_input(f"**Months ahead (Started in {final_df.index[-1].strftime('%Y-%m-%d')})**", value=17, min_value=1, max_value=50)
country = col1[2].multiselect("Please choose country", ["China", "Japan"], ["China", "Japan"])

final_df_differenced = final_df.diff().dropna()
model = VAR(final_df_differenced)
model_fitted = model.fit(maxlags)
lag_order = model_fitted.k_ar
forecast_input = final_df_differenced.values[-lag_order:]

# Forecast
fc = model_fitted.forecast(y=forecast_input, steps=months)
fc_period = pd.date_range(start=final_df.index[-1] + relativedelta(months=1), periods=months, freq='MS')
df_forecast = pd.DataFrame(fc, index=fc_period, columns=final_df.columns)

def invert_transformation(df_train, df_forecast):
    df_fc = pd.DataFrame(index=df_forecast.index)
    for i, col in enumerate(df_train.columns):
        last_value = df_train[col].iloc[-1]
        df_fc[f"{col}_forecast"] = last_value + df_forecast.iloc[:, i].cumsum()
    return df_fc

df_forecast_processed = invert_transformation(final_df, df_forecast)
df_res = df_forecast_processed[[col for col in df_forecast_processed.columns if col.endswith('_forecast')]].copy()
df_res["Date"] = df_res.index

# Apply upside/downside

def apply_upside_downside(df, column, up_pct, down_pct):
    df[f'{column}_upside'] = df[column] * (1 + up_pct / 100)
    df[f'{column}_downside'] = df[column] * (1 - down_pct / 100)
    return df

for country_name in ["China", "Japan"]:
    colname = f"{country_name} HRC (FOB, $/t)_forecast"
    if colname in df_res.columns:
        df_res = apply_upside_downside(df_res, colname, upside_pct, downside_pct)

# Prepare history for display from 2022-03
history = final_df[[f"{c} HRC (FOB, $/t)" for c in country]].copy()
history = history[history.index >= pd.to_datetime("2022-03-01")]
history = history.reset_index()
history["type"] = "Historical"

# Forecast data
forecast_melted = pd.melt(df_res, id_vars=["Date"], value_vars=[f"{c} HRC (FOB, $/t)_forecast" for c in country], var_name="series", value_name="value")
forecast_melted["type"] = "Forecast"
forecast_melted["series"] = forecast_melted["series"].str.replace("_forecast", "")

# Combine all
history_melted = pd.melt(history, id_vars=["Date"], var_name="series", value_name="value")
history_melted["series"] = history_melted["series"].str.replace(" HRC \(FOB, \$/t\)", "")
full_data = pd.concat([history_melted.assign(type="Historical"), forecast_melted])

# Plot
fig = go.Figure()

for s in full_data["series"].unique():
    for t in ["Historical", "Forecast"]:
        subset = full_data[(full_data["series"] == s) & (full_data["type"] == t)]
        fig.add_trace(go.Scatter(
            x=subset["Date"],
            y=subset["value"],
            name=f"{s} {t}",
            mode="lines",
            line=dict(dash="solid" if t == "Forecast" else "dot")
        ))

# Add uncertainty
for c in country:
    forecast_col = f"{c} HRC (FOB, $/t)_forecast"
    upper = f"{forecast_col}_upside"
    lower = f"{forecast_col}_downside"
    if upper in df_res.columns and lower in df_res.columns:
        fig.add_trace(go.Scatter(
            x=pd.concat([df_res["Date"], df_res["Date"][::-1]]),
            y=pd.concat([df_res[upper], df_res[lower][::-1]]),
            fill='toself',
            fillcolor='rgba(0,100,80,0.2)' if c == "China" else 'rgba(255,0,0,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=True,
            name=f"{c} Upside/Downside"
        ))

fig.update_layout(
    title="Forecasting HRC Prices with Historical Data + Upside/Downside",
    xaxis_title="Date",
    height=500,
    legend=dict(orientation="h", x=0.5, xanchor="center"),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)

col[1].plotly_chart(fig, use_container_width=True)

