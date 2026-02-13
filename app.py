import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.linear_model import LinearRegression


# =========================
# KONFIGURASI AWAL
# =========================
st.set_page_config(page_title="Dashboard Analisis Cuaca", layout="wide")
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #74ebd5, #ACB6E5);
}

h1 {
    text-align: center;
    color: white;
    font-weight: 700;
}

.kpi-card {
    background: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
    text-align: center;
}

.section-box {
    background: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)
st.markdown("<h1>🌦️ Weather Analytics Dashboard</h1>", unsafe_allow_html=True)

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("dataset.csv")
    df["tanggal"] = pd.to_datetime(df["tanggal"])
    df["bulan"] = df["tanggal"].dt.month
    return df

df = load_data()

# =========================
# SIDEBAR MENU
# =========================
menu = st.sidebar.radio(
    "Menu Navigasi",
    ["📊 Dashboard Trend",
     "🔎 Analisis Harian",
     "📈 Analisis Bulanan",
     "📥 Download Data",
     " 🤖 Forecast ML"]
    
)

# =========================
# 1️⃣ DASHBOARD TREND
# =========================

if menu == "📊 Dashboard Trend":

    st.subheader("Trend Parameter Cuaca")

    parameter = st.selectbox(
        "Pilih Parameter",
        ["suhu_rata2", "kelembaban", "curah_hujan",
         "tekanan_udara", "kecepatan_angin"]
    )

   fig = px.line(
    df,
    x="tanggal",
    y=parameter,
    color_discrete_sequence=["#FF6B6B"],
    title=f"Trend {parameter}"
)

fig.update_layout(
    plot_bgcolor="white",
    paper_bgcolor="white",
    title_font=dict(size=22),
)

    st.plotly_chart(fig, use_container_width=True)

    st.metric("Rata-rata", round(df[parameter].mean(), 2))


# =========================
# 2️⃣ ANALISIS HARIAN
# =========================
elif menu == "🔎 Analisis Harian":

    st.subheader("Analisis Berdasarkan Tanggal")

    tanggal_input = st.date_input("Pilih Tanggal")

    hasil = df[df["tanggal"] == pd.to_datetime(tanggal_input)]

    if not hasil.empty:
        st.dataframe(hasil)

        rata_suhu = df["suhu_rata2"].mean()
        suhu_hari = hasil["suhu_rata2"].values[0]

        if suhu_hari > rata_suhu:
            st.success("Suhu hari ini di atas rata-rata tahunan 🌡️")
        else:
            st.info("Suhu hari ini di bawah rata-rata tahunan")

    else:
        st.warning("Data tidak ditemukan untuk tanggal tersebut")


# =========================
# 3️⃣ ANALISIS BULANAN
# =========================
elif menu == "📈 Analisis Bulanan":

    st.subheader("Rata-rata Bulanan")

    bulanan = df.groupby("bulan").mean(numeric_only=True).reset_index()

    fig = px.bar(bulanan, x="bulan", y="suhu_rata2",
                 title="Rata-rata Suhu Bulanan")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(bulanan)


# =========================
# 4️⃣ DOWNLOAD DATA
# =========================
elif menu == "📥 Download Data":

    st.subheader("Download Dataset")

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="dataset_cuaca.csv",
        mime="text/csv",
    )
# =========================
# 🤖 Forecast ML
# =========================
elif menu == "🤖 Forecast ML":

    st.subheader("Forecast Suhu Menggunakan Linear Regression")

    # Gunakan suhu sebagai contoh
    df_sorted = df.sort_values("tanggal").copy()
    df_sorted["hari_ke"] = np.arange(len(df_sorted))

    X = df_sorted[["hari_ke"]]
    y = df_sorted["suhu_rata2"]

    model = LinearRegression()
    model.fit(X, y)

    # Prediksi 7 hari ke depan
    hari_terakhir = df_sorted["hari_ke"].iloc[-1]
    future_days = np.arange(hari_terakhir + 1, hari_terakhir + 8)
    future_X = future_days.reshape(-1, 1)

    predictions = model.predict(future_X)

    future_dates = pd.date_range(
        df_sorted["tanggal"].iloc[-1] + pd.Timedelta(days=1),
        periods=7
    )

    forecast_df = pd.DataFrame({
        "tanggal": future_dates,
        "forecast_suhu": predictions
    })

    # Gabungkan data lama + forecast
    fig = px.line(df_sorted, x="tanggal", y="suhu_rata2",
                  title="Forecast 7 Hari ke Depan")

    fig.add_scatter(
        x=forecast_df["tanggal"],
        y=forecast_df["forecast_suhu"],
        mode="lines",
        name="Forecast",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.write("Hasil Prediksi 7 Hari Ke Depan")
    st.dataframe(forecast_df)

