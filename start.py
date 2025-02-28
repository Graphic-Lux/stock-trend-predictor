import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import streamlit as st

# Function to fetch stock data
def fetch_stock_data(ticker, period='1y'):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    df.to_csv(f'{ticker}_data.csv')
    return df

# Function to visualize stock data
def plot_stock_data(df, ticker):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Close'], label='Close Price', color='blue')
    plt.title(f'{ticker} Stock Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(plt)

# Function to predict future stock prices using Linear Regression
def predict_stock_trend(df, days=30):
    df['Days'] = np.arange(len(df))
    X = df[['Days']]
    y = df['Close']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict future prices
    future_days = np.arange(len(df), len(df) + days).reshape(-1, 1)
    future_predictions = model.predict(future_days)
    
    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Close'], label='Actual Prices', color='blue')
    plt.plot(pd.date_range(df.index[-1], periods=days+1, freq='D')[1:], future_predictions, label='Predicted Prices', linestyle='dashed', color='red')
    plt.title('Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(plt)
    
    return future_predictions

# Streamlit Web App
st.title("Stock Market Trend Predictor")

ticker = st.text_input("Enter Stock Ticker (e.g., AAPL):", "AAPL")
days = st.slider("Select Prediction Days:", min_value=1, max_value=60, value=30)

if st.button("Fetch Data & Predict"):
    df = fetch_stock_data(ticker)
    st.write("### Stock Data Preview")
    st.write(df.tail())
    plot_stock_data(df, ticker)
    predicted_prices = predict_stock_trend(df, days=days)
    st.write("### Predicted Prices")
    st.write(predicted_prices)
