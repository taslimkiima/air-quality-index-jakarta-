import pandas as pd
import os
import re 
import numpy as np

# --- 1. Definisikan Nama Kolom Standar ---
KOLOM_STANDAR = [
    'periode_data', 'tanggal_lengkap', 'tahun', 'bulan', 'hari', 'stasiun',
    'pm10', 'pm25', 'so2', 'co', 'o3', 'no2',
    'max_ispu', 'parameter_kritis', 'kategori'
]

# --- 2. Pemetaan Stasiun untuk Normalisasi (Sama) ---
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


# --- 3. Fungsi Pemrosesan Utama: Memuat, Mengubah Nama, dan Membersihkan ---
def load_and_standardize(file_path, year):
    print(f"Memproses data tahun {year} dari file: {file_path}...")
    try:
        df = pd.read_excel(file_path, engine='openpyxl') 
    except FileNotFoundError:
        print(f"❌ ERROR: File tidak ditemukan di {file_path}. Lewati.")
        return None
    except Exception as e:
        print(f"❌ ERROR saat membaca file {file_path}: {e}. Lewati.")
        return None

    # Membersihkan nama kolom dan mengubah ke huruf kecil
    df.columns = df.columns.str.strip().str.lower()
    
    # --- 3a. Pemetaan Kolom Standard ---
    
    # Pemetaan untuk kolom polutan dan ISPU (Menggunakan errors='ignore')
    column_mapping_common = {
        'max': 'max_ispu', 
        'critical': 'parameter_kritis', 
        'categori': 'kategori', 
        'lokasi_spku': 'stasiun',
        'pm_10': 'pm10', 'pm_duakomalima': 'pm25', 
        'pm_sepuluh': 'pm10', 'sulfur_dioksida': 'so2',
        'karbon_monoksida': 'co', 'ozon': 'o3', 'nitrogen_dioksida': 'no2',
        # Pemetaan Tanggal (Variasi Lama: 'tanggal')
        'tanggal': 'date_source', 
    }
    df = df.rename(columns=column_mapping_common, errors='ignore')
    
    # Penanganan PM2.5 yang Hilang (Terutama 2020)
    if 'pm25' not in df.columns: 
        df['pm25'] = pd.NA 
        
    df['tahun'] = year 

    # --- 3b. Pemrosesan Tanggal Terpadu (LOGIKA FINAL UNTUK 2020, 2021, 2023) ---
    
    # KASUS 1: Tanggal sudah dalam satu kolom (date_source) - Berlaku untuk 2020, 2021, 2023
    if 'date_source' in df.columns:
        
        # 1. Konversi ke string dan Hapus Komponen Waktu/Spasi
        df['date_source'] = df['date_source'].astype(str).str.split().str[0]
        
        # 2. Parsing Tanggal (Robust)
        # Kita menggunakan dayfirst=False karena data 2020 Anda terlihat YYYY-MM-DD
        df['tanggal_lengkap'] = pd.to_datetime(
            df['date_source'], 
            errors='coerce', 
            dayfirst=False
        ) 
        
        # 3. Ekstrak komponen tanggal yang benar
        df['bulan'] = df['tanggal_lengkap'].dt.month
        df['hari'] = df['tanggal_lengkap'].dt.day
        df['tahun'] = df['tanggal_lengkap'].dt.year 
        
        df.drop(columns=['date_source'], errors='ignore', inplace=True)
    
    else:
        # Fallback jika kolom tanggal tidak terdeteksi
        df['tanggal_lengkap'] = pd.NaT
        df['bulan'] = pd.NA
        df['hari'] = pd.NA
        df['tahun'] = year 

    # 3c. Normalisasi Stasiun & Finalisasi
    if 'stasiun' in df.columns:
        df['stasiun'] = df['stasiun'].apply(normalize_station)
    
    # Ambil hanya kolom standar dengan urutan yang benar
    standardized_df = df.reindex(columns=KOLOM_STANDAR)
    print(f"✅ Data tahun {year} berhasil diseragamkan.")
    return standardized_df

# --- 4. Konfigurasi File Input dan Eksekusi (Fokus pada 2020, 2021, 2023) ---
dataset = 'dataset'

data_files_subset = {
    2020: os.path.join(dataset, 'ispu_jakarta_2020.xlsx'), 
    2021: os.path.join(dataset, 'ispu_jakarta_2021.xlsx'),
    2022: os.path.join(dataset, 'ispu_jakarta_2022.xlsx'), # Dihapus Sementara
    2023: os.path.join(dataset, 'ispu_jakarta_2023.xlsx'),
    # 2024: os.path.join(dataset, 'ispu_jakarta_2024_2025.xlsx'), # Dihapus Sementara
    # 2025: os.path.join(dataset, 'ispu_jakarta_2024_2025.xlsx'), # Dihapus Sementara
}

all_dataframes = []

for year, file_path in data_files_subset.items():
    df_standard = load_and_standardize(file_path, year)
    if df_standard is not None:
        all_dataframes.append(df_standard)

# --- 5. Menggabungkan dan Menyimpan Hasil (Sama) ---
if all_dataframes:
    final_df = pd.concat(all_dataframes, ignore_index=True)
    
    # Konversi kolom polutan ke numerik 
    kolom_ispu = ['pm10', 'pm25', 'so2', 'co', 'o3', 'no2', 'max_ispu']
    for col in kolom_ispu:
        final_df[col] = pd.to_numeric(final_df[col], errors='coerce') 

    OUTPUT_FILE = 'data_kualitas_udara_gabungan_2020_2021_2022_2023.csv'
    final_df.to_csv(OUTPUT_FILE, index=False)
    
    print("\n==============================================")
    print("✅ PROSES GABUNG DATA SELESAI!")
    print(f"Total baris data yang berhasil digabungkan: {len(final_df)}")
    print(f"Hasil disimpan di: {OUTPUT_FILE}")
    print("==============================================")

    print("\nContoh data yang sudah seragam:")
    print(final_df[['tanggal_lengkap', 'stasiun', 'pm10', 'pm25', 'max_ispu', 'kategori']].head(10))

else:
    print("\n❌ Gagal menggabungkan data karena tidak ada file yang berhasil dimuat.")