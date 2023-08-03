import pandas as pd
import numpy as np
from plotly import graph_objs as go
import time
from datetime import date

import streamlit as st
import yfinance as yf

import lstm_model as m

st.title("Stock Price Prediction System")

start = "2018-01-01"
today = date.today().strftime("%Y-%m-%d")

stocks = ('MSFT', 'AAPL', "SBIN.NS", "GOOG", "TSLA", "AMZN", "SPOT", "META", "SNAP", "ZOMATO.NS", "TATASTEEL.NS", "RPOWER.NS", "ITC.NS")
selected_stock = st.selectbox("Select stock ticker", stocks)

years = st.slider("Years of prediction", 1, 4)
period = years*365

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, start, today)
    data.reset_index(inplace = True)
    return data

data_load_state = st.text("Loading data...")
data = load_data(selected_stock)
data_load_state.text("Loading data...Done!")

st.subheader("Raw data")
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
    fig.layout.update(title_text = 'Plot of raw data', xaxis_rangeslider_visible = True)
    st.plotly_chart(fig)

plot_raw_data()

progress_text = "Operation in progress.... Please wait."
my_bar = st.progress(0, text=progress_text)

for percent_complete in range(100):
    time.sleep(1.2)
    my_bar.progress(percent_complete + 1, text=progress_text)

trainPredictPlot, testPredictPlot, output, closedf = m.lstm_model(data)

def plot_comparison():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='actual_close', ))
    fig.add_trace(go.Scatter(x=data['Date'], y=trainPredictPlot.reshape(1,-1)[0].tolist(), name='train_predicted_close'))
    fig.add_trace(go.Scatter(x=data['Date'], y=testPredictPlot.reshape(1,-1)[0].tolist(), name='test_predicted_close'))
    fig.update_layout(title_text = 'Comparision between Original vs Predicted Close Price', xaxis_title_font_size=10, 
                      yaxis_title_font_size= 10,legend_font_size= 10, xaxis_rangeslider_visible = True)
    st.plotly_chart(fig)

plot_comparison()

#Plotting predicted stock prices of next 30 days
def plot_pred():

    #Creating a dummy plane to plot two graphs one after another
    plot_old = np.arange(1,101)
    plot_pred = np.arange(101,131)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_old, y=closedf, name='Current_Close_price'))
    fig.add_trace(go.Scatter(x=plot_pred, y=output, name='Future_Close_Price'))
    fig.layout.update(title_text = 'Predicting Stock Price of next 30 days', xaxis_rangeslider_visible = True)
    st.plotly_chart(fig)

plot_pred()