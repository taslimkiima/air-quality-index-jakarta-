import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os

# --- 1. Konfigurasi dan Muat Aset (TIDAK BERUBAH) ---
FILE_ADVANCED = 'data_ispu_preprocess_final_ADVANCED.csv'
MODEL_CBF_PATH = 'model_cbf_rekomendasi.pkl'
SCALER_PATH = 'scaler_rekomendasi.pkl'
FITUR_LIST_PATH = 'fitur_list.pkl'
OPTIMAL_THRESHOLD = 0.70 
REKOMENDASI_TINDAKAN = {
    0: "Kualitas udara AMAN. Tetap pantau kondisi, terutama saat jam sibuk.",
    1: "WASPADA TINGKAT TINGGI! Kualitas Udara diprediksi TIDAK SEHAT. Wajib gunakan masker N95 dan batasi aktivitas fisik di luar ruangan.",
}
STATION_COL_NAME = 'stasiun' 

# --- 2. Fungsi Pembantu: Menghitung Matriks Kesamaan Stasiun (CF) (TIDAK BERUBAH) ---
def calculate_station_similarity(df, polutan='pm25'):
    # ... (fungsi calculate_station_similarity) ...
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

# --- 3. Fungsi Utama: Sistem Rekomendasi Hybrid ---

def get_hybrid_recommendation(data_input_df, target_stasiun):
    # Muat Aset (TIDAK BERUBAH)
    try:
        scaler = joblib.load(SCALER_PATH)
        cbf_model = joblib.load(MODEL_CBF_PATH)
        fitur_list = joblib.load(FITUR_LIST_PATH)
        df_full = pd.read_csv(FILE_ADVANCED)
    except FileNotFoundError as e:
        return {"Error": f"Aset model atau data tidak ditemukan: {e}. Pastikan Anda sudah menjalankan script pelatihan."}

    # Hitung Matriks Kesamaan (CF) (TIDAK BERUBAH)
    sim_df = calculate_station_similarity(df_full)
    
    # --- A. Content-Based Filtering (CBF) ---
    data_input_clean = data_input_df.reindex(columns=fitur_list).fillna(df_full[fitur_list].mean())
    data_input_scaled = scaler.transform(data_input_clean)
    cbf_proba = cbf_model.predict_proba(data_input_scaled)[0][1] 
    cbf_prediction = 1 if cbf_proba >= OPTIMAL_THRESHOLD else 0 
    rekomendasi_utama = REKOMENDASI_TINDAKAN.get(cbf_prediction, "Error dalam prediksi kategori.")
    
    # --- B. Collaborative Filtering (CF) (TIDAK BERUBAH) ---
    cf_output = "Tidak ada peringatan korelasi."
    if target_stasiun in sim_df.columns:
        similar_stations = sim_df[target_stasiun].sort_values(ascending=False).index.tolist()
        if target_stasiun in similar_stations: similar_stations.remove(target_stasiun)
        if similar_stations:
            top_similar_stasiun = similar_stations[0] 
            korelasi_score = sim_df.loc[target_stasiun, top_similar_stasiun]
            cf_output = (f"Stasiun dengan pola polusi terdekat: **{top_similar_stasiun}** (Korelasi: {korelasi_score:.2f}). "
                         f"Kualitas udara cenderung mengikuti pola lokasi tersebut.")
        
    # --- C. Fusion Output dan Rekomendasi Pejabat (PERBAIKAN LOGIKA) ---
    
    rekomendasi_pejabat = ""
    pm25_val = data_input_df['pm25'].iloc[0]
    is_weekday = data_input_df['hari_dalam_minggu'].iloc[0] < 5
    is_pm_critical = pm25_val > 100 # Batas TIDAK SEHAT

    if is_pm_critical:
        # Pemicu Terkuat: Polusi sangat tinggi
        rekomendasi_pejabat = "TINDAKAN DARURAT: Terapkan kebijakan WFH atau batasi kendaraan di zona ini selama 24 jam ke depan."
    elif pm25_val > 70 and is_weekday:
        # Pemicu Sedang: Polusi tinggi pada Hari Kerja (kaitannya dengan emisi)
        rekomendasi_pejabat = "Perketat Uji Emisi pada kendaraan niaga di sekitar stasiun ini selama jam puncak hari kerja."
    else:
        # Pemicu Default: Polusi rendah/sedang atau di Akhir Pekan
        rekomendasi_pejabat = "Lanjutkan pemantauan rutin. Pertimbangkan penambahan RTH di lokasi korelasi tinggi."
    
    
    return {
        "Stasiun Target": target_stasiun,
        "Status Prediksi (CBF)": "TIDAK SEHAT" if cbf_prediction == 1 else "AMAN/SEDANG",
        "Probabilitas TIDAK SEHAT": f"{cbf_proba*100:.1f}%",
        "Rekomendasi Tindakan Primer": rekomendasi_utama,
        "Peringatan Situasional (CF)": cf_output,
        "Rekomendasi Kebijakan (Pejabat)": rekomendasi_pejabat
    }

# --- 4. Contoh Penggunaan (Simulasi) (TIDAK BERUBAH) ---
if __name__ == '__main__':
    # ... (kode simulasi dihilangkan untuk ringkasan, tetapi Anda akan menjalankannya) ...
    print("--- SIMULASI SISTEM REKOMENDASI HYBRID ---")
    
    try:
        df_full = pd.read_csv(FILE_ADVANCED)
        fitur_list = joblib.load(FITUR_LIST_PATH)

        # Skenario 1: Ambil data untuk simulasi kondisi TIDAK SEHAT (PM2.5 > 100)
        sample_row_data = df_full[df_full['pm25'] > 100].iloc[0] 
        target_station = sample_row_data[STATION_COL_NAME]

        # Konversi data sample menjadi DataFrame 1 baris (simulasi input real-time)
        input_data_df = pd.DataFrame([sample_row_data], columns=df_full.columns)
        
        # Jalankan rekomendasi
        results = get_hybrid_recommendation(input_data_df, target_station)
        
        print("\n===========================================")
        print("✅ HASIL REKOMENDASI FUSION HYBRID")
        print("===========================================")
        for key, value in results.items():
            print(f"- {key}: {value}")
            
    except Exception as e:
        print(f"\n❌ Gagal menjalankan simulasi. Pastikan file aset (.csv dan .pkl) ada.")
        print(f"Detail Error: {e}")