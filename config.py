# config.py

import os

# --- KONFIGURASI PATH FILE ---
FILE_ADVANCED = 'data_ispu_preprocess_final_ADVANCED.csv'
MODEL_CBF_PATH = 'model_cbf_rekomendasi.pkl'
SCALER_PATH = 'scaler_rekomendasi.pkl'
FITUR_LIST_PATH = 'fitur_list.pkl'

# --- PARAMETER REKOMENDASI ---
OPTIMAL_THRESHOLD = 0.70 
STATION_COL_NAME = 'stasiun' 

# Mapping untuk output rekomendasi tindak lanjut (Masyarakat)
REKOMENDASI_TINDAKAN = {
    0: "Kualitas udara AMAN. Tetap pantau kondisi, terutama saat jam sibuk.",
    1: "WASPADA TINGKAT TINGGI! Kualitas Udara diprediksi TIDAK SEHAT. Wajib gunakan masker N95 dan batasi aktivitas fisik di luar ruangan.",
}

# --- PERBAIKAN: STATION MAP DAN FUNGSI NORMALISASI ---
STATION_MAP = {
    'DKI1': 'DKI1 Bunderan HI', 'DKI1 Bunderan HI': 'DKI1 Bunderan HI',
    'DKI2': 'DKI2 Kelapa Gading', 'DKI2 Kelapa Gading': 'DKI2 Kelapa Gading',
    'DKI3': 'DKI3 Jagakarsa', 'DKI3 Jagakarsa': 'DKI3 Jagakarsa',
    'DKI4': 'DKI4 Lubang Buaya', 'DKI4 Lubang Buaya': 'DKI4 Lubang Buaya',
    'DKI5': 'DKI5 Kebon Jeruk Jakarta Barat', 'DKI5 Kebon Jeruk Jakarta Barat': 'DKI5 Kebon Jeruk Jakarta Barat',
    'Kebon Jeruk Jakarta Barat': 'DKI5 Kebon Jeruk Jakarta Barat',
    'Bunderan HI': 'DKI1 Bunderan HI', 'Kelapa Gading': 'DKI2 Kelapa Gading',
    'Jagakarsa': 'DKI3 Jagakarsa', 'Lubang Buaya': 'DKI4 Lubang Buaya',
    'DKI5 (Kebon Jeruk) Jakarta Barat': 'DKI5 Kebon Jeruk Jakarta Barat'
}

def normalize_station(station_name):
    """Menyeragamkan nama stasiun berdasarkan STATION_MAP."""
    if isinstance(station_name, str):
        standardized = station_name.strip()
        if standardized in STATION_MAP:
            return STATION_MAP[standardized]
        parts = standardized.split()
        if parts and parts[0] in ['DKI1', 'DKI2', 'DKI3', 'DKI4', 'DKI5']:
             return STATION_MAP.get(parts[0], standardized)
        return standardized 
    return station_name