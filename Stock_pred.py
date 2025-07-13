# streamlit_app.py

import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

st.set_page_config(layout="wide")
st.title("📈 Stock Price Prediction App using Machine Learning")

# Sidebar for user input
st.sidebar.header("Select Stock Parameters")
ticker = st.sidebar.text_input("Stock Ticker (e.g., AAPL, MSFT)", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-12-31"))

if st.sidebar.button("Run Prediction"):

    # Fetch Data
    df = yf.download(ticker, start=start_date, end=end_date)

    if df.empty:
        st.error("❌ No data found. Try another stock ticker or date range.")
    else:
        # Feature Engineering
        df['Daily_Return'] = df['Close'].pct_change()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        df['Volatility'] = df['Daily_Return'].rolling(window=20).std()

        # Drop missing values
        df.dropna(inplace=True)

        # Data Scaling
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Daily_Return', 'MA_20', 'MA_50', 'Volatility']
        scaler = MinMaxScaler()
        df[features] = scaler.fit_transform(df[features])

        # Prepare for ML
        X = df[['Open', 'High', 'Low', 'Volume', 'Daily_Return', 'MA_20', 'MA_50', 'Volatility']]
        y = df['Close']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train Models
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        lr_pred = lr.predict(X_test)

        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)

        # Model Metrics
        lr_r2 = r2_score(y_test, lr_pred)
        rf_r2 = r2_score(y_test, rf_pred)

        # Display Results
        st.subheader("📊 Model Comparison")
        st.markdown(f"**Linear Regression R²:** {lr_r2:.4f}")
        st.markdown(f"**Random Forest R²:** {rf_r2:.4f}")

        # Plots
        st.subheader("📈 Actual vs Predicted Prices (Random Forest)")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(y_test.index, y_test.values, label='Actual Prices', color='blue')
        ax.plot(y_test.index, rf_pred, label='Predicted Prices', color='red', linestyle='--')
        ax.set_title("Random Forest: Actual vs Predicted")
        ax.set_xlabel("Index")
        ax.set_ylabel("Scaled Price")
        ax.legend()
        st.pyplot(fig)

        st.subheader("📉 Feature Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df[features].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        st.pyplot(fig)

        st.subheader("📊 Price and Moving Averages")
        df_unscaled = yf.download(ticker, start=start_date, end=end_date)
        df_unscaled['MA_20'] = df_unscaled['Close'].rolling(window=20).mean()
        df_unscaled['MA_50'] = df_unscaled['Close'].rolling(window=50).mean()

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df_unscaled['Close'], label="Close Price", color="black")
        ax.plot(df_unscaled['MA_20'], label="MA 20", color="green")
        ax.plot(df_unscaled['MA_50'], label="MA 50", color="orange")
        ax.set_title(f"{ticker} Stock Price with Moving Averages")
        ax.legend()
        st.pyplot(fig)
