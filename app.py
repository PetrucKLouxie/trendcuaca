import streamlit as st
import pandas as pd
import numpy as np
import requests
import base64
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime

st.set_page_config(page_title="Analisis Cuaca PRO", layout="wide")

# ==============================
# CUSTOM CSS
# ==============================

st.markdown("""
<style>
.main {background-color: #0f172a;}
h1,h2,h3,h4 {color: white;}
.metric-card {
    background: linear-gradient(135deg, #2563eb, #9333ea);
    padding: 20px;
    border-radius: 15px;
    color: white;
    text-align: center;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-bottom">', unsafe_allow_html=True)

st.sidebar.image("logo.png", width=120)
st.sidebar.markdown("""
<center>
<b>Analisis Trend Cuaca</b><br>
Version 1.0<br>
© 2026 Yosef
</center>
""", unsafe_allow_html=True)

st.sidebar.markdown('</div>', unsafe_allow_html=True)


# ==============================
# LOAD DATA FROM GITHUB
# ==============================

@st.cache_data
def load_data():
    repo = st.secrets["GITHUB_REPO"]
    file_path = st.secrets["GITHUB_FILE_PATH"]
    token = st.secrets["GITHUB_TOKEN"]

    url = f"https://api.github.com/repos/{repo}/contents/{file_path}"
    headers = {"Authorization": f"token {token}"}

    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        st.error("Gagal mengambil file dari GitHub")
        st.stop()

    data_json = response.json()

    decoded = base64.b64decode(data_json["content"])
    df = pd.read_csv(pd.io.common.BytesIO(decoded))

    df.columns = df.columns.str.strip()

    if "Date" not in df.columns:
        st.error(f"Kolom tidak ditemukan. Kolom tersedia: {df.columns.tolist()}")
        st.stop()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    return df

df = load_data()

# ==============================
# UPDATE CSV TO GITHUB
# ==============================

def update_github_csv(df):
    repo = st.secrets["GITHUB_REPO"]
    file_path = st.secrets["GITHUB_FILE_PATH"]
    token = st.secrets["GITHUB_TOKEN"]

    url = f"https://api.github.com/repos/{repo}/contents/{file_path}"
    headers = {"Authorization": f"token {token}"}

    response = requests.get(url, headers=headers)
    sha = response.json()["sha"]

    csv_string = df.to_csv(index=False)
    encoded_content = base64.b64encode(csv_string.encode()).decode()

    data = {
        "message": "Update dataset via Streamlit App",
        "content": encoded_content,
        "sha": sha
    }

    requests.put(url, headers=headers, json=data)
    st.cache_data.clear()

# ==============================
# SIDEBAR
# ==============================

st.sidebar.title("🌦 Weather System")
menu = st.sidebar.radio("Telusuri", [
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

    st.subheader("Temperature Trend")
    st.line_chart(df.set_index("Date")["Temperature"])

# ==============================
# TREND
# ==============================

elif menu == "Trend Analysis":

    st.title("📈 Trend Analysis")
    param = st.selectbox("Choose Parameter", ["Temperature","Humidity","Rainfall"])
    st.line_chart(df.set_index("Date")[param])

# ==============================
# FORECAST
# ==============================

elif menu == "Forecast ML":

    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split

    st.title("🤖 Machine Learning Forecast")

    # =========================
    # LOAD DATA
    # =========================
    df = pd.read_csv("dataset.csv")

    df["Date"] = pd.to_datetime(df["Date"])
    df["day"] = df["Date"].dt.day
    df["month"] = df["Date"].dt.month
    df["year"] = df["Date"].dt.year

    # =========================
    # FEATURES & TARGET
    # =========================
    X = df[["day", "month", "year", "Humidity", "Rainfall"]]
    y = df["Temperature"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)

    # =========================
    # FORECAST NEXT DAY
    # =========================
    last_row = df.iloc[-1]

    next_day = last_row["day"] + 1

    future_input = np.array([[ 
        next_day,
        last_row["month"],
        last_row["year"],
        last_row["humidity"],
        last_row["rainfall"]
    ]])

    # Prediction
    prediction = model.predict(future_input)[0]

    # Confidence estimation (std dev from trees)
    preds = np.array([tree.predict(future_input)[0] for tree in model.estimators_])
    std_dev = np.std(preds)
    lower = prediction - 1.96 * std_dev
    upper = prediction + 1.96 * std_dev

    # =========================
    # DESKRIPSI OTOMATIS
    # =========================
    def generate_description(temp, humidity, rainfall):
        if rainfall > 5:
            return "Berpotensi hujan dengan kelembaban tinggi."
        elif temp > 32:
            return "Cuaca panas dengan suhu tinggi."
        elif humidity > 80:
            return "Udara lembab, kemungkinan mendung."
        else:
            return "Cuaca relatif stabil dan normal."

    description = generate_description(prediction, last_row["humidity"], last_row["rainfall"])

    # =========================
    # TAMPILKAN HASIL
    # =========================
    st.metric("Prediksi Suhu Besok", f"{prediction:.2f} °C")

    st.info(description)

    # =========================
    # GRAFIK CONFIDENCE
    # =========================
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=[0],
        y=[prediction],
        mode="markers",
        name="Prediction"
    ))

    fig.add_trace(go.Scatter(
        x=[0, 0],
        y=[lower, upper],
        mode="lines",
        name="Confidence Interval"
    ))

    fig.update_layout(
        title="Confidence Interval Forecast",
        xaxis=dict(showticklabels=False),
        yaxis_title="Temperature (°C)",
        template="plotly_dark"
    )

    st.plotly_chart(fig)

    # =========================
    # AKURASI MODEL
    # =========================
    score = model.score(X_test, y_test)
    st.write(f"Model Accuracy (R² Score): {score:.2f}")


# ==============================
# SEARCH
# ==============================

elif menu == "Search by Date":

    st.title("🔍 Search Data")

    selected_date = st.date_input("Select Date")
    selected_time = st.time_input("Select Time")

    selected_datetime = datetime.combine(selected_date, selected_time)

    result = df[df["Date"] == pd.to_datetime(selected_datetime)]

    if not result.empty:
        st.success("Data Found")
        st.dataframe(result)
    else:
        st.error("No Data Found")

# ==============================
# ADMIN LOGIN + DATE + TIME
# ==============================

elif menu == "Admin Login":

    st.title("🔐 Admin Panel")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if username == "admin" and password == "cuaca123":

        st.success("Login Success")

        st.subheader("➕ Add New Data")

        new_date = st.date_input("Date")
        new_time = st.time_input("Time")

        new_temp = st.number_input("Temperature")
        new_humidity = st.number_input("Humidity")
        new_rain = st.number_input("Rainfall")

        if st.button("Save Data"):

            full_datetime = datetime.combine(new_date, new_time)

            new_row = pd.DataFrame({
                "Date":[full_datetime],
                "Temperature":[new_temp],
                "Humidity":[new_humidity],
                "Rainfall":[new_rain]
            })

            df_updated = pd.concat([df, new_row])

            update_github_csv(df_updated)

            st.success("Data berhasil disimpan ke GitHub ✅")

    else:
        st.warning("Login Required")
