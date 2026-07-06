import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px  # Tambahan untuk grafik yang modern

# =========================================================
# KONFIGURASI HALAMAN (Modern & Clean)
# =========================================================
st.set_page_config(
    page_title="Bookstore Sales Forecasting",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk mempercantik UI tambahan
st.markdown("""
    <style>
    .main .block-container {padding-top: 2rem;}
    div[data-testid="stMetricValue"] {font-size: 24px; font-weight: bold; color: #1E3A8A;}
    </style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL, ENCODER, SCALER, DAFTAR FITUR
# =========================================================
@st.cache_resource
def load_artifacts():
    # Menggunakan try-except agar aman jika file belum siap
    try:
        model = joblib.load("model_terbaik.pkl")
        label_encoder = joblib.load("label_encoder.pkl")
        scaler = joblib.load("scaler.pkl")
        feature_columns = joblib.load("feature_columns.pkl")
        return model, label_encoder, scaler, feature_columns
    except Exception as e:
        st.error(f"Gagal memuat model/artifact: {e}")
        return None, None, None, None

model, le, scaler, feature_columns = load_artifacts()

if model is not None:
    MODELS_NEED_SCALING = ("LinearRegression", "SVR")
    model_needs_scaling = type(model).__name__ in MODELS_NEED_SCALING

# =========================================================
# LOAD & OLAH DATA HISTORIS
# =========================================================
@st.cache_data
def load_weekly_sales():
    try:
        df = pd.read_csv("Data Penjualan Toko Buku.csv")
        df.columns = ["id_transaksi", "jenis_item", "jumlah", "tanggal_pembelian", "nama_customer", "total"]
        df = df.dropna().drop_duplicates()
        df["tanggal_pembelian"] = pd.to_datetime(df["tanggal_pembelian"], errors="coerce")
        df = df.dropna()

        df["tahun"] = df["tanggal_pembelian"].dt.year
        df["bulan"] = df["tanggal_pembelian"].dt.month
        df["minggu"] = df["tanggal_pembelian"].dt.isocalendar().week.astype(int)

        weekly_sales = df.groupby(["tahun", "bulan", "minggu", "jenis_item"]).agg({"total": "sum", "jumlah": "sum"}).reset_index()
        weekly_sales.rename(columns={"total": "total_penjualan", "jumlah": "total_item"}, inplace=True)
        weekly_sales["jenis_item_encoded"] = le.transform(weekly_sales["jenis_item"])
        weekly_sales = weekly_sales.sort_values(by=["jenis_item_encoded", "tahun", "minggu"])

        # Feature Engineering (Lag & Rolling)
        for i in range(1, 5):
            weekly_sales[f"lag_{i}"] = weekly_sales.groupby("jenis_item_encoded")["total_penjualan"].shift(i)
        
        weekly_sales["rolling_mean_4"] = weekly_sales.groupby("jenis_item_encoded")["total_penjualan"].transform(lambda x: x.shift(1).rolling(window=4).mean())
        weekly_sales["rolling_std_4"] = weekly_sales.groupby("jenis_item_encoded")["total_penjualan"].transform(lambda x: x.shift(1).rolling(window=4).std())
        
        return weekly_sales.dropna()
    except Exception as e:
        st.sidebar.error(f"Gagal memuat dataset historis: {e}")
        return pd.DataFrame()

if model is not None:
    weekly_sales = load_weekly_sales()

def predict(X_row: pd.DataFrame):
    X_row = X_row[feature_columns]
    if model_needs_scaling:
        X_row = scaler.transform(X_row)
    return model.predict(X_row)

# =========================================================
# SIDEBAR (Memindahkan Info Teknis Agar Dashboard Bersih)
# =========================================================
with st.sidebar:
    st.image("https://img.icons8.com/fluent/96/000000/books.png", width=80)
    st.title("Control Panel")
    st.markdown("---")
    st.subheader("⚙️ Informasi Model")
    if model is not None:
        st.info(f"**Tipe Model:**\n`{type(model).__name__}`")
        st.metric(label="Jumlah Fitur", value=len(feature_columns))
        st.metric(label="Total Kategori Buku", value=len(le.classes_))
    else:
        st.error("Model gagal dimuat. Periksa file pkl Anda.")
    st.markdown("---")
    st.caption("✨ Dashboard v2.0 · Powered by Streamlit")

# =========================================================
# HEADER UTAMA
# =========================================================
st.title("📚 Intelligence Sales Forecasting Dashboard")
st.markdown("Platform analitik prediksi penjualan mingguan toko buku berbasis *Machine Learning*.")
st.markdown("---")

if model is not None:
    tab1, tab2 = st.tabs(["🔮 Prediksi Manual Instan", "📂 Analisis Bulk (Upload CSV)"])

    # =========================================================
    # TAB 1 - INPUT MANUAL (Modern Layout)
    # =========================================================
    with tab1:
        st.subheader("Form Parameter Prediksi")
        
        # Grid input 3 kolom agar lebih hemat tempat dan rapi
        col_in1, col_in2, col_in3 = st.columns(3)
        with col_in1:
            jenis_item = st.selectbox("📚 Pilih Jenis Item", sorted(le.classes_))
            total_item = st.number_input("📦 Estimasi Penjualan Item (Qty)", min_value=0, value=10, step=1)
        with col_in2:
            tahun = st.number_input("📅 Tahun Target", min_value=2010, max_value=2035, value=2026, step=1)
            bulan = st.slider("📆 Bulan Target", min_value=1, max_value=12, value=7)
        with col_in3:
            minggu = st.number_input("🔢 Minggu ke- (1