import pandas as pd
import os
import numpy as np

# --- 1. Definisikan Nama Kolom Standar ---
KOLOM_STANDAR = [
    'periode_data', 'tanggal_lengkap', 'tahun', 'bulan', 'hari', 'stasiun',
    'pm10', 'pm25', 'so2', 'co', 'o3', 'no2',
    'max_ispu', 'parameter_kritis', 'kategori'
]

# Konfigurasi input/output (Sama)
FILE_INPUT = 'dataset/ispu_jakarta_2024_2025.xlsx'
OUTPUT_CSV = 'ispu_2024_2025_date_fixed.csv'

def fix_2024_2025_date_columns(file_path, target_year):
    """
    Mengambil data dari file 2024/2025 dan merekonstruksi kolom tanggal lengkap.
    """
    print(f"--- Memproses data tahun {target_year} ---")
    
    # 1. Muat data
    try:
        df = pd.read_excel(file_path, engine='openpyxl')
    except FileNotFoundError:
        print(f"❌ ERROR: File {file_path} tidak ditemukan.")
        return None
    
    df.columns = df.columns.str.strip().str.lower()
    
    # --- Identifikasi dan Perbaiki Kolom ---
    
    # 2. Filter data (Karena satu file mencakup 2024 dan 2025)
    df['tahun_data_temp'] = df['periode_data'].astype(str).str[:4].astype(int)
    df_filtered = df[df['tahun_data_temp'] == target_year].copy()
    
    if df_filtered.empty:
        print(f"   [PERINGATAN]: Tidak ada data untuk tahun {target_year} di file ini.")
        return None
    
    # 3. Konstruksi Tanggal Lengkap (YYYY-MM-DD)
    
    # Renaming sementara untuk konstruksi tanggal
    df_filtered.rename(columns={'tanggal': 'hari_num_temp'}, errors='ignore', inplace=True)
    
    # Kolom sumber: 'bulan', 'hari_num_temp', 'tahun_data_temp'
    
    df_filtered['bulan'] = df_filtered['bulan'].fillna(1).astype(int)
    df_filtered['hari_num_temp'] = df_filtered['hari_num_temp'].fillna(1).astype(int)
    
    # Konstruksi string tanggal yang benar
    df_filtered['tanggal_string'] = (
        df_filtered['tahun_data_temp'].astype(str) + '-' + 
        df_filtered['bulan'].astype(str).str.zfill(2) + '-' + 
        df_filtered['hari_num_temp'].astype(str).str.zfill(2)
    )
    
    # Konversi ke datetime
    df_filtered['tanggal_lengkap'] = pd.to_datetime(df_filtered['tanggal_string'], errors='coerce')
    
    # 4. Verifikasi dan Output
    df_output = df_filtered[['tanggal_lengkap', 'stasiun', 'pm_duakomalima', 'kategori', 'periode_data']].head(5)
    print("✅ Tanggal berhasil direkonstruksi. Contoh 5 baris:")
    print(df_output.to_string(index=False)) 
    
    # Hapus kolom temporer sebelum return
    df_filtered.drop(columns=['tanggal_string'], errors='ignore', inplace=True)
    
    return df_filtered

def standardize_and_save_final_2024_data(df_final_raw, output_csv):
    """
    Memetakan kolom dari format 2024/2025 ke KOLOM_STANDAR sebelum disimpan.
    """
    
    # 1. Pemetaan Kolom Akhir (Mengambil semua kolom polutan yang benar)
    final_rename_map = {
        'pm_duakomalima': 'pm25', 
        'pm_sepuluh': 'pm10',
        'sulfur_dioksida': 'so2', # Tambahan
        'karbon_monoksida': 'co', # Tambahan
        'ozon': 'o3', # Tambahan
        'nitrogen_dioksida': 'no2', # Tambahan
        'max': 'max_ispu', 
        'parameter_pencemar_kritis': 'parameter_kritis',
        'tahun_data_temp': 'tahun',
        'hari_num_temp': 'hari',
        'bulan': 'bulan' # Memastikan bulan ikut terpetakan
    }
    
    df_final = df_final_raw.copy()
    
    # Terapkan Renaming
    df_final = df_final.rename(columns=final_rename_map, errors='ignore')
    
    # 2. Finalisasi Komponen Tanggal/Waktu dari kolom yang sudah diperbaiki
    df_final['tahun'] = df_final['tanggal_lengkap'].dt.year.fillna(df_final.get('tahun', pd.NA))
    df_final['bulan'] = df_final['tanggal_lengkap'].dt.month.fillna(df_final.get('bulan', pd.NA))
    df_final['hari'] = df_final['tanggal_lengkap'].dt.day.fillna(df_final.get('hari', pd.NA))
    
    # 3. Ambil hanya KOLOM_STANDAR
    df_final = df_final.reindex(columns=KOLOM_STANDAR)
    
    # 4. Simpan Hasil Akhir
    df_final.to_csv(output_csv, index=False)
    
    return df_final


if __name__ == '__main__':
    # --- EKSEKUSI ---
    df_2024 = fix_2024_2025_date_columns(FILE_INPUT, 2024)
    df_2025 = fix_2024_2025_date_columns(FILE_INPUT, 2025)
    
    if df_2024 is not None or df_2025 is not None:
        
        # Gabungkan hasil 2024 dan 2025 mentah
        data_to_concat = [df for df in [df_2024, df_2025] if df is not None]
        df_merged_raw = pd.concat(data_to_concat, ignore_index=True)
        
        # Standardisasi dan Simpan Final
        df_standardized = standardize_and_save_final_2024_data(df_merged_raw, OUTPUT_CSV)
        
        print(f"\n=======================================================")
        print(f"Total data 2024 & 2025 yang berhasil disatukan: {len(df_standardized)} baris")
        print(f"Hasil verifikasi tanggal disimpan di: {OUTPUT_CSV}")
        print(f"=======================================================")