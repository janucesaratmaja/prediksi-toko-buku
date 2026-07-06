import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================================================
# KONFIGURASI HALAMAN
# =========================================================
st.set_page_config(
    page_title="Prediksi Penjualan Toko Buku",
    page_icon="📚",
    layout="wide"
)

# =========================================================
# LOAD MODEL, ENCODER, SCALER, DAFTAR FITUR
# =========================================================
@st.cache_resource
def load_artifacts():
    model = joblib.load("model_terbaik.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, label_encoder, scaler, feature_columns

model, le, scaler, feature_columns = load_artifacts()

# Model yang butuh data ter-scale (sesuai notebook training)
MODELS_NEED_SCALING = ("LinearRegression", "SVR")
model_needs_scaling = type(model).__name__ in MODELS_NEED_SCALING


# =========================================================
# LOAD & OLAH DATA HISTORIS (untuk sumber lag & rolling feature)
# =========================================================
@st.cache_data
def load_weekly_sales():
    df = pd.read_csv("Data Penjualan Toko Buku.csv")
    df.columns = [
        "id_transaksi",
        "jenis_item",
        "jumlah",
        "tanggal_pembelian",
        "nama_customer",
        "total"
    ]

    df = df.dropna()
    df = df.drop_duplicates()
    df["tanggal_pembelian"] = pd.to_datetime(df["tanggal_pembelian"], errors="coerce")
    df = df.dropna()

    df["tahun"] = df["tanggal_pembelian"].dt.year
    df["bulan"] = df["tanggal_pembelian"].dt.month
    df["minggu"] = df["tanggal_pembelian"].dt.isocalendar().week.astype(int)

    weekly_sales = df.groupby(
        ["tahun", "bulan", "minggu", "jenis_item"]
    ).agg({"total": "sum", "jumlah": "sum"}).reset_index()

    weekly_sales.rename(
        columns={"total": "total_penjualan", "jumlah": "total_item"},
        inplace=True
    )

    weekly_sales["jenis_item_encoded"] = le.transform(weekly_sales["jenis_item"])

    weekly_sales = weekly_sales.sort_values(by=["jenis_item_encoded", "tahun", "minggu"])

    weekly_sales["lag_1"] = weekly_sales.groupby("jenis_item_encoded")["total_penjualan"].shift(1)
    weekly_sales["lag_2"] = weekly_sales.groupby("jenis_item_encoded")["total_penjualan"].shift(2)
    weekly_sales["lag_3"] = weekly_sales.groupby("jenis_item_encoded")["total_penjualan"].shift(3)
    weekly_sales["lag_4"] = weekly_sales.groupby("jenis_item_encoded")["total_penjualan"].shift(4)

    weekly_sales["rolling_mean_4"] = weekly_sales.groupby("jenis_item_encoded")["total_penjualan"].transform(
        lambda x: x.shift(1).rolling(window=4).mean()
    )
    weekly_sales["rolling_std_4"] = weekly_sales.groupby("jenis_item_encoded")["total_penjualan"].transform(
        lambda x: x.shift(1).rolling(window=4).std()
    )

    return weekly_sales.dropna()


weekly_sales = load_weekly_sales()


def predict(X_row: pd.DataFrame):
    """X_row harus punya kolom persis sesuai feature_columns."""
    X_row = X_row[feature_columns]
    if model_needs_scaling:
        X_row = scaler.transform(X_row)
    pred = model.predict(X_row)
    return pred


# =========================================================
# HEADER
# =========================================================
st.title("📚 Prediksi Penjualan Toko Buku")
st.caption(
    f"Model: **{type(model).__name__}** · "
    f"Fitur yang dipakai: {len(feature_columns)} · "
    f"Kategori item: {len(le.classes_)}"
)

tab1, tab2 = st.tabs(["🔮 Input Manual", "📂 Upload CSV"])

# =========================================================
# TAB 1 - INPUT MANUAL
# =========================================================
with tab1:
    st.subheader("Prediksi Satu Minggu ke Depan")
    st.write(
        "Pilih jenis item, lalu tentukan periode minggu yang ingin diprediksi. "
        "Nilai lag & rolling otomatis diambil dari riwayat penjualan item tersebut."
    )

    col1, col2 = st.columns(2)

    with col1:
        jenis_item = st.selectbox("Jenis Item", sorted(le.classes_))
        tahun = st.number_input("Tahun", min_value=2010, max_value=2035, value=2024, step=1)
        bulan = st.number_input("Bulan", min_value=1, max_value=12, value=1, step=1)

    with col2:
        minggu = st.number_input("Minggu ke- (dalam setahun)", min_value=1, max_value=53, value=1, step=1)
        total_item = st.number_input(
            "Estimasi Jumlah Item Terjual Minggu Ini", min_value=0, value=10, step=1
        )

    if st.button("Prediksi Penjualan", type="primary"):
        item_encoded = le.transform([jenis_item])[0]

        history = weekly_sales[weekly_sales["jenis_item_encoded"] == item_encoded]

        if history.empty:
            st.warning(
                "Belum ada cukup data historis untuk item ini, "
                "sehingga lag & rolling feature memakai nilai 0."
            )
            lag_1 = lag_2 = lag_3 = lag_4 = 0
            rolling_mean_4 = rolling_std_4 = 0
        else:
            last_row = history.sort_values(by=["tahun", "minggu"]).iloc[-1]
            lag_1 = last_row["total_penjualan"]
            lag_2 = last_row["lag_1"]
            lag_3 = last_row["lag_2"]
            lag_4 = last_row["lag_3"]
            rolling_mean_4 = history["total_penjualan"].tail(4).mean()
            rolling_std_4 = history["total_penjualan"].tail(4).std()
            if pd.isna(rolling_std_4):
                rolling_std_4 = 0

        input_df = pd.DataFrame([{
            "tahun": tahun,
            "bulan": bulan,
            "minggu": minggu,
            "jenis_item_encoded": item_encoded,
            "total_item": total_item,
            "lag_1": lag_1,
            "lag_2": lag_2,
            "lag_3": lag_3,
            "lag_4": lag_4,
            "rolling_mean_4": rolling_mean_4,
            "rolling_std_4": rolling_std_4
        }])

        hasil = predict(input_df)[0]

        st.success(f"Prediksi Total Penjualan: **Rp {hasil:,.0f}**")

        with st.expander("Lihat detail fitur yang dipakai model"):
            st.dataframe(input_df)

# =========================================================
# TAB 2 - UPLOAD CSV
# =========================================================
with tab2:
    st.subheader("Prediksi Banyak Baris dari File CSV")
    st.write(
        "Upload file CSV transaksi dengan format kolom yang sama seperti dataset training "
        "(`id_transaksi, jenis_item, jumlah, tanggal pembelian, nama_customer, total`). "
        "Aplikasi akan otomatis mengagregasi menjadi penjualan mingguan dan memprediksi tiap barisnya."
    )

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            raw = pd.read_csv(uploaded_file)
            raw.columns = [
                "id_transaksi", "jenis_item", "jumlah",
                "tanggal_pembelian", "nama_customer", "total"
            ]
            raw = raw.dropna().drop_duplicates()
            raw["tanggal_pembelian"] = pd.to_datetime(raw["tanggal_pembelian"], errors="coerce")
            raw = raw.dropna()

            raw["tahun"] = raw["tanggal_pembelian"].dt.year
            raw["bulan"] = raw["tanggal_pembelian"].dt.month
            raw["minggu"] = raw["tanggal_pembelian"].dt.isocalendar().week.astype(int)

            ws = raw.groupby(
                ["tahun", "bulan", "minggu", "jenis_item"]
            ).agg({"total": "sum", "jumlah": "sum"}).reset_index()

            ws.rename(columns={"total": "total_penjualan", "jumlah": "total_item"}, inplace=True)

            # Item yang belum pernah dilihat saat training akan ditolak label encoder-nya
            dikenal = set(le.classes_)
            tidak_dikenal = set(ws["jenis_item"].unique()) - dikenal
            if tidak_dikenal:
                st.warning(f"Jenis item berikut tidak dikenali model dan akan diabaikan: {sorted(tidak_dikenal)}")
                ws = ws[ws["jenis_item"].isin(dikenal)]

            ws["jenis_item_encoded"] = le.transform(ws["jenis_item"])
            ws = ws.sort_values(by=["jenis_item_encoded", "tahun", "minggu"])

            ws["lag_1"] = ws.groupby("jenis_item_encoded")["total_penjualan"].shift(1)
            ws["lag_2"] = ws.groupby("jenis_item_encoded")["total_penjualan"].shift(2)
            ws["lag_3"] = ws.groupby("jenis_item_encoded")["total_penjualan"].shift(3)
            ws["lag_4"] = ws.groupby("jenis_item_encoded")["total_penjualan"].shift(4)

            ws["rolling_mean_4"] = ws.groupby("jenis_item_encoded")["total_penjualan"].transform(
                lambda x: x.shift(1).rolling(window=4).mean()
            )
            ws["rolling_std_4"] = ws.groupby("jenis_item_encoded")["total_penjualan"].transform(
                lambda x: x.shift(1).rolling(window=4).std()
            )

            ws_clean = ws.dropna().copy()

            if ws_clean.empty:
                st.error(
                    "Setelah feature engineering (lag & rolling 4 minggu), tidak ada baris tersisa. "
                    "Data yang diupload kemungkinan terlalu sedikit / rentang waktunya terlalu pendek."
                )
            else:
                preds = predict(ws_clean)
                ws_clean["prediksi_total_penjualan"] = preds

                st.success(f"Berhasil memprediksi {len(ws_clean)} baris.")
                st.dataframe(
                    ws_clean[[
                        "tahun", "bulan", "minggu", "jenis_item",
                        "total_item", "total_penjualan", "prediksi_total_penjualan"
                    ]]
                )

                csv_hasil = ws_clean.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Hasil Prediksi (CSV)",
                    data=csv_hasil,
                    file_name="hasil_prediksi.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Gagal memproses file: {e}")

st.divider()
st.caption("Dibuat dengan Streamlit · Model dilatih di Google Colab")
