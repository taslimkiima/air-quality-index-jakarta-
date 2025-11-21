# recommender_core.py

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import streamlit as st 

# Import konfigurasi dari file config.py
from config import (
    FILE_ADVANCED, MODEL_CBF_PATH, SCALER_PATH, FITUR_LIST_PATH,
    OPTIMAL_THRESHOLD, REKOMENDASI_TINDAKAN, STATION_COL_NAME
)


# --- FUNGSI MUAT ASET DENGAN CACHING ---

@st.cache_data
def load_data():
    """Memuat data ISPU dari CSV dan melakukan preprocessing dasar."""
    try:
        # Gunakan data yang sudah dipreprocess
        df = pd.read_csv(FILE_ADVANCED)
        df['tanggal_lengkap'] = pd.to_datetime(df['tanggal_lengkap']) 
        return df
    except Exception as e:
        st.error(f"Gagal memuat data: {e}. Pastikan '{FILE_ADVANCED}' ada.")
        return pd.DataFrame()

@st.cache_resource
def load_ml_assets():
    """Memuat model, scaler, dan daftar fitur dari file .pkl."""
    try:
        scaler = joblib.load(SCALER_PATH)
        cbf_model = joblib.load(MODEL_CBF_PATH)
        fitur_list = joblib.load(FITUR_LIST_PATH)
        return scaler, cbf_model, fitur_list
    except Exception as e:
        st.error(f"Gagal memuat aset ML: {e}. Pastikan file .pkl sudah tersedia.")
        return None, None, None

@st.cache_data
def calculate_station_similarity(df, polutan='pm25'):
    """Menghitung matriks kesamaan antar stasiun menggunakan Cosine Similarity."""
    df_pivot = df.pivot_table(
        index='tanggal_lengkap', 
        columns=STATION_COL_NAME, 
        values=polutan
    ).fillna(0)
    item_similarity_matrix = cosine_similarity(df_pivot.T)
    item_similarity_df = pd.DataFrame(
        item_similarity_matrix,
        index=df_pivot.columns,
        columns=df_pivot.columns
    )
    return item_similarity_df


# --- FUNGSI REKOMENDASI KONDISI AKTUAL SAAT INI (Masyarakat) ---
def get_actual_recommendation(kategori):
    """Menentukan rekomendasi aksi berdasarkan kategori ISPU AKTUAL saat ini."""
    kategori_upper = str(kategori).upper()
    if 'BAIK' in kategori_upper:
        return '✅ Aktivitas Normal, Udara Aman'
    elif 'SEDANG' in kategori_upper:
        return '🟡 Batasi Aktivitas Berat di Luar'
    elif 'TIDAK SEHAT' in kategori_upper:
        return '🔴 Hindari Aktivitas Luar, Wajib Masker'
    elif 'SANGAT TIDAK SEHAT' in kategori_upper:
        return '🚨 Sangat Berbahaya! Tetap di Dalam Ruangan'
    elif 'TIDAK ADA DATA' in kategori_upper:
        return '❓ Data Tidak Tersedia'
    else:
        return 'ℹ️ Cek Ulang Status'


# --- FUNGSI REKOMENDASI KEBIJAKAN UNTUK DATA HISTORIS (Pejabat) ---
def get_historical_pejabat_recommendation(row):
    """Menentukan rekomendasi kebijakan berdasarkan data historis aktual."""
    pm25_val = row.get('pm25', 0)
    is_weekday = row.get('hari_dalam_minggu', 0) < 5 
    is_pm_critical = pm25_val > 100 
    is_pm_high = pm25_val > 70 

    if is_pm_critical:
        return "DARURAT: WFH/Pembatasan Kendaraan & Prioritas RTH."
    elif is_pm_high and is_weekday:
        return "MITIGASI: Uji Emisi Ketat & Tinjauan Operasional Industri."
    else:
        return "RUTIN: Monitoring & Investasi Jangka Panjang (LEZ/RTH)."


# --- FUNGSI STYLING UNTUK HISTORICAL TRACKING ---
def highlight_historical_recommendation(val):
    """Memberikan warna latar belakang pada kategori di tabel historis."""
    val = str(val).upper()
    if '🚨' in val or '🔴' in val or 'DARURAT' in val:
        return 'background-color: #ffe0e0; color: #cc0000'  
    elif '🟡' in val or 'MITIGASI' in val:
        return 'background-color: #fffacd; color: #b8860b'  
    elif '✅' in val or 'RUTIN' in val:
        return 'background-color: #e6ffe6; color: #008000'  
    else:
        return ''


# --- FUNGSI UTAMA REKOMENDASI HYBRID (PREDIKSI) ---
def get_hybrid_recommendation(data_input_df, target_stasiun, sim_df, scaler, cbf_model, fitur_list):
    """Menjalankan sistem rekomendasi Hybrid (CBF + CF + Fusion) untuk PREDIKSI."""
    if scaler is None or cbf_model is None:
        return {"Error": "Aset model belum dimuat. Periksa log error."}
        
    input_row = data_input_df.iloc[0]
    
    # --- A. Content-Based Filtering (CBF) - PREDIKSI ---
    data_input_clean = pd.DataFrame([input_row]).reindex(columns=fitur_list).fillna(0)
    if not data_input_clean.empty and not data_input_clean.isnull().all().all():
        data_input_scaled = scaler.transform(data_input_clean)
        cbf_proba = cbf_model.predict_proba(data_input_scaled)[0][1] 
        cbf_prediction = 1 if cbf_proba >= OPTIMAL_THRESHOLD else 0 
    else:
        cbf_proba = 0.0
        cbf_prediction = 0

    rekomendasi_utama = REKOMENDASI_TINDAKAN.get(cbf_prediction, "Error dalam prediksi kategori.")
    
    # --- B. Collaborative Filtering (CF) ---
    cf_output = "Tidak ada peringatan korelasi."
    if target_stasiun in sim_df.columns:
        similar_stations = sim_df[target_stasiun].sort_values(ascending=False).index.tolist()
        if target_stasiun in similar_stations: similar_stations.remove(target_stasiun)
        
        if similar_stations:
            top_similar_stasiun = similar_stations[0] 
            korelasi_score = sim_df.loc[target_stasiun, top_similar_stasiun]
            cf_output = (f"Stasiun dengan pola polusi terdekat: **{top_similar_stasiun}** (Korelasi: {korelasi_score:.2f}). "
                         f"Kualitas udara cenderung mengikuti pola lokasi tersebut.")
            
    # --- C. Fusion Output dan Rekomendasi Pejabat ---
    pm25_val = input_row.get('pm25', 0)
    is_weekday = input_row.get('hari_dalam_minggu', 0) < 5 
    is_pm_critical = pm25_val > 100 
    is_pm_high = pm25_val > 70 

    if is_pm_critical:
        rekomendasi_pejabat = (
            "TINDAKAN DARURAT: Terapkan kebijakan WFH atau pembatasan kendaraan berat (genap-ganjil) di zona ini selama 24 jam ke depan. "
            "PERENCANAAN JANGKA MENENGAH: Segera finalisasi insentif bagi pengguna kendaraan listrik dan percepat konversi transportasi publik ke energi bersih."
        )
    elif is_pm_high and is_weekday:
        rekomendasi_pejabat = (
            "PERKETAT UJI EMISI: Lakukan uji emisi mendadak di jalanan dan di titik keluar/masuk kawasan industri terdekat. "
            "TATA RUANG: Kaji ulang izin operasional industri yang berdekatan. Tingkatkan efisiensi jalur Transjakarta dan KRL untuk mengurangi penggunaan mobil pribadi."
        )
    else:
        rekomendasi_pejabat = (
            "PEMBANGUNAN BERKELANJUTAN: Lanjutkan pemantauan rutin dan investasikan dana untuk proyek "
            "hijau seperti pengembangan kawasan bebas kendaraan bermotor (Low Emission Zone) dan penambahan 20% Ruang Terbuka Hijau (RTH) di lokasi korelasi tinggi."
        )
    
    
    return {
        "Stasiun Target": target_stasiun,
        "Status Prediksi (CBF)": "TIDAK SEHAT" if cbf_prediction == 1 else "AMAN/SEDANG",
        "Probabilitas TIDAK SEHAT": cbf_proba,
        "Rekomendasi Tindakan Primer": rekomendasi_utama, 
        "Peringatan Situasional (CF)": cf_output,
        "Rekomendasi Kebijakan (Pejabat)": rekomendasi_pejabat
    }