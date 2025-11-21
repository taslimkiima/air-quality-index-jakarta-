import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib 

# --- A. KONFIGURASI DAN DEFINISI ---
FILE_DATA = 'data_kualitas_udara_gabungan_final.csv' 
OUTPUT_FILE_ADVANCED = 'data_ispu_preprocess_final_ADVANCED.csv'
MODEL_CBF_PATH = 'model_cbf_rekomendasi.pkl'
SCALER_PATH = 'scaler_rekomendasi.pkl'
FITUR_LIST_PATH = 'fitur_list.pkl'

POLUTAN_COLS = ['pm10', 'pm25', 'so2', 'co', 'o3', 'no2']
WINDOW_SIZE = 7

# --- B. FUNGSI UTAMA: BUILD ASSET & TRAIN MODEL ---
def build_assets_and_train():
    print("--- ⚙️ TAHAP 1: MEMUAT DAN MEMBERSIHKAN DATA GABUNGAN ---")
    
    try:
        df = pd.read_csv(FILE_DATA)
    except FileNotFoundError:
        print(f"❌ ERROR: File '{FILE_DATA}' tidak ditemukan. Mohon pastikan script merging sudah berjalan.")
        return

    # 1. Pembersihan & Imputasi Dasar
    df['tanggal_lengkap'] = pd.to_datetime(df['tanggal_lengkap'], errors='coerce')
    
    # Filter Data Leakage (Hanya stasiun valid)
    df = df[df['stasiun'].astype(str).str.startswith('DKI')].copy()
    
    # --- PERBAIKAN KRITIS #1: Tentukan dan Hapus Duplikat pada Kunci Primer ---
    
    # 1. Pastikan kolom 'jam' dihitung untuk Kunci Primer (Asumsi data Anda harian, jam = 0)
    df['jam'] = df['tanggal_lengkap'].dt.hour.fillna(0).astype(int) 
    
    # 2. Definisikan Kunci Primer (yang harus unik)
    PRIMARY_KEY = ['stasiun', 'tanggal_lengkap', 'jam']
    
    initial_rows = len(df)
    
    # 3. Menghapus duplikat. Jika ada baris tumpang tindih pada Kunci Primer, hanya ambil yang pertama.
    df.drop_duplicates(subset=PRIMARY_KEY, keep='first', inplace=True)
    print(f"   [Pembersihan Duplikat Kunci]: {initial_rows - len(df)} baris duplikat (dengan Kunci Primer yang sama) dihapus.")
    
    # --- PERBAIKAN KRITIS #2: Urutkan Data SECARA KETAT sebelum Lag/Roll ---
    # Wajib diurutkan berdasarkan Stasiun, Tanggal, dan Jam secara kronologis.
    df = df.sort_values(by=PRIMARY_KEY).reset_index(drop=True)

    # Imputasi & Outlier
    for col in POLUTAN_COLS:
        # Imputasi dilakukan per stasiun untuk mengisi gap (Forward Fill)
        df[col] = df.groupby('stasiun')[col].ffill() 
        df[col] = df[col].fillna(df[col].mean()) # Isi sisa NaN dengan mean global
        batas_atas = df[col].quantile(0.99)
        df[col] = np.where(df[col] > batas_atas, batas_atas, df[col]) # Batasi outlier
    
    print("✅ Pembersihan dan Imputasi Dasar Selesai.")
    
    # --- 2. Feature Engineering Siklus Waktu ---
    # Kolom jam sudah dibuat
    df['hari_dalam_minggu'] = df['tanggal_lengkap'].dt.dayofweek.fillna(0).astype(int) # Senin=0, Minggu=6
    df['nomor_bulan'] = df['tanggal_lengkap'].dt.month.fillna(0).astype(int)
    df['musim'] = (df['nomor_bulan'] % 12 + 3) // 3

    # --- 3. ADVANCED FEATURE ENGINEERING (Lagged & Rolling) ---
    print("\n--- 🧠 TAHAP 2: ADVANCED FEATURE ENGINEERING (Lag/Roll) ---")

    for col in POLUTAN_COLS:
        # Lagged Features (t-1): Menggunakan GROUPBY untuk mencegah data leak antar stasiun.
        df[f'{col}_lag1'] = df.groupby('stasiun')[col].shift(1)
        
        # Rolling Average (7 hari): Menggunakan GROUPBY dan ROLLING pada data yang sudah diurutkan
        df[f'{col}_roll{WINDOW_SIZE}'] = df.groupby('stasiun')[col].rolling(
            window=WINDOW_SIZE, min_periods=1
        ).mean().reset_index(level=0, drop=True)

    # Isi NaN pada kolom fitur yang baru dibuat (Lag/Roll)
    lag_roll_cols = [c for c in df.columns if '_lag1' in c or f'_roll{WINDOW_SIZE}' in c]
    for col in lag_roll_cols:
        # Mengisi nilai NaN pada data awal dengan mean kolom tersebut.
        df[col] = df[col].fillna(df[col].mean())

    # Hapus semua baris yang mungkin masih memiliki NaN pada kolom krusial (biasanya baris pertama)
    df_clean = df.dropna().reset_index(drop=True)
    print(f"   [Pembersihan NaN Final]: {len(df) - len(df_clean)} baris dengan NaN di Lag/Roll (akibat data sangat awal) dihapus.")
    df = df_clean
    
    # --- 4. One-Hot Encoding (OHE) ---
    df_ohe_stasiun = pd.get_dummies(df['stasiun'], prefix='stasiun', dtype=bool)
    df_ohe_kategori = pd.get_dummies(df['kategori'], prefix='kategori', dtype=bool)
    df = pd.concat([df, df_ohe_stasiun, df_ohe_kategori], axis=1)

    # Hapus kolom yang tidak relevan untuk input model
    KOLOM_YANG_DIHAPUS = ['periode_data', 'max_ispu', 'tahun', 'bulan', 'hari', 'parameter_kritis']
    df_clean = df.drop(columns=KOLOM_YANG_DIHAPUS, errors='ignore')
    
    # Simpan Data Advanced FE
    df_clean.to_csv(OUTPUT_FILE_ADVANCED, index=False)
    print(f"✅ Dataset Advanced FE ({len(df_clean)} baris) tersimpan di: {OUTPUT_FILE_ADVANCED}")

    # --- 5. PELATIHAN MODEL CBF & PENYIMPANAN ASET ---
    print("\n--- 🤖 TAHAP 3: PELATIHAN MODEL CBF & PENYIMPANAN ASET ---")
    
    # Definisikan Fitur (X) dan Target (Y)
    fitur_input = [col for col in df_clean.columns if col not in ['tanggal_lengkap', 'stasiun', 'kategori']]
    fitur_input = [col for col in fitur_input if not col.startswith('kategori_')] # Hapus kolom OHE kategori dari X

    X = df_clean[fitur_input].fillna(0) # Sudah diisi di atas, tapi jaga-jaga
    Y = df_clean['kategori_TIDAK SEHAT'] # Target: Klasifikasi TIDAK SEHAT (1) atau tidak (0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)
    cbf_model = LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced')
    cbf_model.fit(X_train, Y_train)

    # Simpan Aset Model
    joblib.dump(cbf_model, MODEL_CBF_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(fitur_input, FITUR_LIST_PATH)
    
    print(f"--- ✅ ASET SIAP! Model, Scaler, dan Fitur List (.pkl) tersimpan.")

# --- EKSEKUSI UTAMA ---
if __name__ == '__main__':
    build_assets_and_train()