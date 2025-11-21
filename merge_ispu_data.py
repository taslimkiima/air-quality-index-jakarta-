import pandas as pd
import os
import numpy as np

# --- 1. KONFIGURASI FILE INPUT & OUTPUT ---
FILE_2020_2023 = 'data_kualitas_udara_gabungan_2020_2021_2022_2023.csv'
FILE_2024_2025 = 'ispu_2024_2025_date_fixed.csv'
FINAL_OUTPUT_FILE = 'data_kualitas_udara_gabungan_final.csv'

# --- 2. FINAL CLEANUP CONFIG ---
KOLOM_POLUTAN = ['pm10', 'pm25', 'so2', 'co', 'o3', 'no2', 'max_ispu']

def final_merge_and_clean():
    print("--- ⚙️ TAHAP AKHIR: PENGGABUNGAN DATA & FINAL CLEANUP ---")
    
    # Muat file 2020-2023
    try:
        df_a = pd.read_csv(FILE_2020_2023)
        print(f"✅ Dimuat: {FILE_2020_2023} ({len(df_a)} baris)")
    except FileNotFoundError:
        print(f"❌ ERROR: File {FILE_2020_2023} tidak ditemukan. Proses dibatalkan.")
        return

    # Muat file 2024-2025
    try:
        df_b = pd.read_csv(FILE_2024_2025)
        print(f"✅ Dimuat: {FILE_2024_2025} ({len(df_b)} baris)")
    except FileNotFoundError:
        print(f"❌ ERROR: File {FILE_2024_2025} tidak ditemukan. Proses dibatalkan.")
        return

    # --- 3. PENGGABUNGAN (CONCATENATION) ---
    final_df = pd.concat([df_a, df_b], ignore_index=True)
    print(f"\n✅ Total baris setelah penggabungan: {len(final_df)}")
    
    # --- 4. FINAL CLEANUP DAN KONVERSI NUMERIK ---
    
    # Pastikan kolom tanggal adalah datetime (Wajib, karena CSV bisa menyimpannya sebagai string)
    final_df['tanggal_lengkap'] = pd.to_datetime(final_df['tanggal_lengkap'], errors='coerce')
    
    # Konversi kolom polutan ke numerik
    for col in KOLOM_POLUTAN:
        final_df[col] = pd.to_numeric(final_df[col], errors='coerce') 

    # --- 5. SIMPAN HASIL ---
    final_df.to_csv(FINAL_OUTPUT_FILE, index=False)
    
    print("\n==============================================")
    print("✅ PROSES FINAL MERGE SELESAI!")
    print(f"Total baris data yang berhasil digabungkan: {len(final_df)}")
    print(f"Hasil disimpan di: {FINAL_OUTPUT_FILE}")
    print("==============================================")

    # --- PERBAIKAN TAMPILAN: Menggunakan to_string() ---
    print("\nContoh 5 baris terakhir (Termasuk data 2024/2025):")
    print(final_df[['tanggal_lengkap', 'stasiun', 'pm10', 'pm25', 'max_ispu', 'kategori']].tail(5).to_string(index=False))

# --- EKSEKUSI UTAMA ---
if __name__ == '__main__':
    final_merge_and_clean()