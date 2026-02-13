import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(page_title="Weather Analytics Dashboard", layout="wide")

# ==============================
# Custom CSS (Redesign Total)
# ==============================

st.markdown("""
<style>
.main {
    background-color: #0f172a;
}
h1, h2, h3 {
    color: #f8fafc;
}
.metric-card {
    background: linear-gradient(135deg, #2563eb, #9333ea);
    padding: 20px;
    border-radius: 15px;
    color: white;
    text-align: center;
    font-size: 18px;
    font-weight: bold;
}
.sidebar .sidebar-content {
    background-color: #1e293b;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# Dataset Setup
# ==============================

DATA_FILE = "weather_dataset.csv"

if not os.path.exists(DATA_FILE):
    dates = pd.date_range(start="2024-01-01", periods=365)
    data = pd.DataFrame({
        "Date": dates,
        "Temperature": np.random.normal(28, 3, 365),
        "Humidity": np.random.normal(75, 5, 365),
        "Rainfall": np.random.normal(10, 4, 365)
    })
    data.to_csv(DATA_FILE, index=False)

df = pd.read_csv(DATA_FILE)
df["Date"] = pd.to_datetime(df["Date"])

# ==============================
# Sidebar Navigation
# ==============================

st.sidebar.title("🌦 Navigation")
menu = st.sidebar.radio("Menu", [
    "Dashboard",
    "Trend Analysis",
    "Forecast ML",
    "Search by Date",
    "Admin Login"
])

# ==============================
# DASHBOARD
# ==============================

if menu == "Dashboard":
    st.title("🌤 Weather Analytics Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.markdown(f'<div class="metric-card">🌡 Avg Temp<br>{df["Temperature"].mean():.2f} °C</div>', unsafe_allow_html=True)
    col2.markdown(f'<div class="metric-card">💧 Avg Humidity<br>{df["Humidity"].mean():.2f} %</div>', unsafe_allow_html=True)
    col3.markdown(f'<div class="metric-card">🌧 Avg Rainfall<br>{df["Rainfall"].mean():.2f} mm</div>', unsafe_allow_html=True)

    st.subheader("📊 Temperature Trend")
    st.line_chart(df.set_index("Date")["Temperature"])

# ==============================
# TREND ANALYSIS
# ==============================

elif menu == "Trend Analysis":
    st.title("📈 Trend Analysis")

    parameter = st.selectbox("Select Parameter", ["Temperature", "Humidity", "Rainfall"])
    st.line_chart(df.set_index("Date")[parameter])

# ==============================
# FORECAST ML
# ==============================

elif menu == "Forecast ML":
    st.title("🔮 Weather Forecast (Simple ML)")

    df["Day"] = np.arange(len(df))

    X = df[["Day"]]
    y = df["Temperature"]

    model = LinearRegression()
    model.fit(X, y)

    future_days = np.arange(len(df), len(df)+7).reshape(-1,1)
    forecast = model.predict(future_days)

    st.subheader("📅 7-Day Temperature Forecast")
    fig, ax = plt.subplots()
    ax.plot(df["Date"], df["Temperature"])
    future_dates = pd.date_range(df["Date"].max(), periods=8)[1:]
    ax.plot(future_dates, forecast)
    st.pyplot(fig)

# ==============================
# SEARCH BY DATE
# ==============================

elif menu == "Search by Date":
    st.title("🔍 Search Weather by Date")

    selected_date = st.date_input("Choose Date")

    result = df[df["Date"] == pd.to_datetime(selected_date)]

    if not result.empty:
        st.success("Data Found ✅")
        st.dataframe(result)
    else:
        st.error("No Data Found")

# ==============================
# ADMIN LOGIN + INPUT
# ==============================

elif menu == "Admin Login":

    st.title("🔐 Admin Access")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if username == "admin" and password == "1234":
        st.success("Login Successful ✅")

        st.subheader("➕ Add New Weather Data")

        new_date = st.date_input("Date")
        new_temp = st.number_input("Temperature")
        new_humidity = st.number_input("Humidity")
        new_rain = st.number_input("Rainfall")

        if st.button("Save Data"):
            new_row = pd.DataFrame({
                "Date": [new_date],
                "Temperature": [new_temp],
                "Humidity": [new_humidity],
                "Rainfall": [new_rain]
            })

            df_updated = pd.concat([df, new_row])
            df_updated.to_csv(DATA_FILE, index=False)

            st.success("Data Saved Permanently ✅")

    else:
        st.warning("Login Required")
