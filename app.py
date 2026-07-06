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
            minggu = st.number_input("🔢 Minggu ke- (1-53)", min_value=1, max_value=53, value=1, step=1)

        st.markdown(" ")
        if st.button("🚀 Hitung Estimasi Penjualan", type="primary", use_container_width=True):
            item_encoded = le.transform([jenis_item])[0]
            history = weekly_sales[weekly_sales["jenis_item_encoded"] == item_encoded]

            if history.empty:
                st.warning("⚠️ Data historis item minim. Menggunakan fallback nilai default (0).")
                lag_1 = lag_2 = lag_3 = lag_4 = 0
                rolling_mean_4 = rolling_std_4 = 0
            else:
                last_row = history.sort_values(by=["tahun", "minggu"]).iloc[-1]
                lag_1, lag_2, lag_3 = last_row["total_penjualan"], last_row["lag_1"], last_row["lag_2"]
                lag_4 = last_row["lag_3"]
                rolling_mean_4 = history["total_penjualan"].tail(4).mean()
                rolling_std_4 = history["total_penjualan"].tail(4).std()
                if pd.isna(rolling_std_4): rolling_std_4 = 0

            input_df = pd.DataFrame([{
                "tahun": tahun, "bulan": bulan, "minggu": minggu,
                "jenis_item_encoded": item_encoded, "total_item": total_item,
                "lag_1": lag_1, "lag_2": lag_2, "lag_3": lag_3, "lag_4": lag_4,
                "rolling_mean_4": rolling_mean_4, "rolling_std_4": rolling_std_4
            }])

            hasil = predict(input_df)[0]

            # Tampilan Hasil yang Mewah
            st.markdown("### 📊 Hasil Analisis Prediksi")
            res_col1, res_col2 = st.columns([1, 2])
            with res_col1:
                st.metric(
                    label=f"Estimasi Pendapatan ({jenis_item})", 
                    value=f"Rp {hasil:,.0f}"
                )
            with res_col2:
                st.success(f"Model memprediksi penjualan untuk minggu ke-{minggu} di bulan ke-{bulan} akan menghasilkan pendapatan optimal sebesar **Rp {hasil:,.0f}** dengan estimasi kuantitas produk {total_item} pcs.")

            # Visualisasi Tren Historis Item yang Dipilih
            if not history.empty:
                st.markdown("#### 📈 Tren Penjualan Historis Terakhir")
                fig = px.line(
                    history.tail(12), 
                    x="minggu", 
                    y="total_penjualan", 
                    title=f"12 Catatan Transaksi Terakhir - Kategori {jenis_item}",
                    labels={"total_penjualan": "Total Penjualan (Rp)", "minggu": "Minggu Ke-"},
                    markers=True
                )
                fig.update_layout(hovermode="x unified", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)

            with st.expander("🛠️ Periksa Fitur Matriks Internal (X_Row Datatable)"):
                st.dataframe(input_df, use_container_width=True)

    # =========================================================
    # TAB 2 - UPLOAD CSV (Modern Layout & Interactive Chart)
    # =========================================================
    with tab2:
        st.subheader("Pemrosesan Data Skala Besar")
        st.write("Unggah dokumen transaksi ritel untuk otomatisasi transformasi data dan pembuatan prediksi simultan.")
        
        uploaded_file = st.file_uploader("Seret dan letakkan file CSV di sini", type=["csv"])

        if uploaded_file is not None:
            try:
                with st.spinner("Sedang mengekstrak dan menghitung data..."):
                    raw = pd.read_csv(uploaded_file)
                    raw.columns = ["id_transaksi", "jenis_item", "jumlah", "tanggal_pembelian", "nama_customer", "total"]
                    raw = raw.dropna().drop_duplicates()
                    raw["tanggal_pembelian"] = pd.to_datetime(raw["tanggal_pembelian"], errors="coerce")
                    raw = raw.dropna()

                    raw["tahun"] = raw["tanggal_pembelian"].dt.year
                    raw["bulan"] = raw["tanggal_pembelian"].dt.month
                    raw["minggu"] = raw["tanggal_pembelian"].dt.isocalendar().week.astype(int)

                    ws = raw.groupby(["tahun", "bulan", "minggu", "jenis_item"]).agg({"total": "sum", "jumlah": "sum"}).reset_index()
                    ws.rename(columns={"total": "total_penjualan", "jumlah": "total_item"}, inplace=True)

                    dikenal = set(le.classes_)
                    tidak_dikenal = set(ws["jenis_item"].unique()) - dikenal
                    if tidak_dikenal:
                        st.warning(f"⚠️ {len(tidak_dikenal)} kategori baru diabaikan karena belum dikenali model.")
                        ws = ws[ws["jenis_item"].isin(dikenal)]

                    ws["jenis_item_encoded"] = le.transform(ws["jenis_item"])
                    ws = ws.sort_values(by=["jenis_item_encoded", "tahun", "minggu"])

                    for i in range(1, 5):
                        ws[f"lag_{i}"] = ws.groupby("jenis_item_encoded")["total_penjualan"].shift(i)
                    
                    ws["rolling_mean_4"] = ws.groupby("jenis_item_encoded")["total_penjualan"].transform(lambda x: x.shift(1).rolling(window=4).mean())
                    ws["rolling_std_4"] = ws.groupby("jenis_item_encoded")["total_penjualan"].transform(lambda x: x.shift(1).rolling(window=4).std())

                    ws_clean = ws.dropna().copy()

                if ws_clean.empty:
                    st.error("❌ Data tidak cukup untuk diekstraksi ke pola mingguan (Minimal dibutuhkan data rentang waktu > 4 minggu).")
                else:
                    preds = predict(ws_clean)
                    ws_clean["prediksi_total_penjualan"] = preds

                    # Metrik Ringkasan File
                    st.success(f"⚡ Pemrosesan Selesai! Berhasil memprediksi total **{len(ws_clean)} baris** agregat.")
                    
                    m1, m2 = st.columns(2)
                    m1.metric("Total Penjualan Riwayat", f"Rp {ws_clean['total_penjualan'].sum():,.0f}")
                    m2.metric("Total Estimasi Penjualan Depan", f"Rp {ws_clean['prediksi_total_penjualan'].sum():,.0f}")

                    # Chart Interaktif Perbandingan Data Riwayat vs Prediksi
                    st.markdown("#### 🔀 Perbandingan Nilai Aktual vs Hasil Prediksi")
                    fig_compare = px.scatter(
                        ws_clean, 
                        x="total_penjualan", 
                        y="prediksi_total_penjualan",
                        color="jenis_item",
                        labels={"total_penjualan": "Nilai Aktual (Rp)", "prediksi_total_penjualan": "Prediksi Model (Rp)"},
                        title="Sebaran Akurasi Prediksi terhadap Aktual berdasarkan Jenis Item"
                    )
                    fig_compare.add_shape(type="line", line=dict(dash="dash", color="red"), x0=ws_clean["total_penjualan"].min(), y0=ws_clean["total_penjualan"].min(), x1=ws_clean["total_penjualan"].max(), y1=ws_clean["total_penjualan"].max())
                    st.plotly_chart(fig_compare, use_container_width=True)

                    # Tabel Preview Data
                    st.markdown("#### 📋 Lembar Pratinjau Hasil")
                    st.dataframe(
                        ws_clean[["tahun", "bulan", "minggu", "jenis_item", "total_item", "total_penjualan", "prediksi_total_penjualan"]],
                        use_container_width=True
                    )

                    # Tombol download yang mencolok
                    csv_hasil = ws_clean.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Unduh Hasil Prediksi Lengkap (.CSV)",
                        data=csv_hasil,
                        file_name="rekap_prediksi_penjualan.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

            except Exception as e:
                st.error(f"Gagal memproses file: {e}")