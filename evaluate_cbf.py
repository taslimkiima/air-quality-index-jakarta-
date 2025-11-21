from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import joblib

# --- 1. Konfigurasi dan Muat Aset ---
FILE_ADVANCED = 'data_ispu_preprocess_final_ADVANCED.csv'
MODEL_CBF_PATH = 'model_cbf_rekomendasi.pkl'
SCALER_PATH = 'scaler_rekomendasi.pkl'
FITUR_LIST_PATH = 'fitur_list.pkl'

print("--- Memuat Aset dan Data untuk Evaluasi ---")
try:
    df_clean = pd.read_csv(FILE_ADVANCED)
    cbf_model = joblib.load(MODEL_CBF_PATH)
    scaler = joblib.load(SCALER_PATH)
    fitur_list = joblib.load(FITUR_LIST_PATH)
except FileNotFoundError as e:
    print(f"❌ ERROR: Aset tidak ditemukan. Pastikan file (.csv, .pkl) sudah dibuat di langkah pelatihan.")
    print(f"Detail: {e}")
    exit()

# --- 2. Persiapan Data Uji (Replikasi Pembagian Data) ---

# Definisikan X dan Y (Menggunakan fitur yang sama seperti saat pelatihan)
X = df_clean[fitur_list].fillna(df_clean[fitur_list].mean())
Y = df_clean['kategori_TIDAK SEHAT'] # Target: 1 jika TIDAK SEHAT

# Scaling X (data penuh)
# Penting: Gunakan scaler yang sudah dilatih (scaler.transform)
X_scaled = scaler.transform(X)

# Bagi Data Ulang (Wajib menggunakan random_state=42 yang sama)
# Ini memastikan X_test dan Y_test identik dengan yang digunakan saat pelatihan
X_train, X_test, Y_train, Y_test = train_test_split(
    X_scaled, Y, test_size=0.2, random_state=42
)

# --- 3. Evaluasi Model ---

print("\n--- 🔬 Hasil Evaluasi Model Content-Based Filtering (CBF) ---")

# Prediksi pada data uji
Y_pred = cbf_model.predict(X_test)

# 1. Confusion Matrix
print("\n1. Confusion Matrix (Matriks Kebingungan):")
cm = confusion_matrix(Y_test, Y_pred)
print(cm)
print("\n[Baris: Aktual; Kolom: Prediksi] | [0: AMAN/SEDANG, 1: TIDAK SEHAT]")


# 2. Classification Report (Metrik Kunci)
print("\n2. Classification Report (Precision, Recall, F1-Score):")
report = classification_report(
    Y_test, Y_pred, target_names=['0 (AMAN/SEDANG)', '1 (TIDAK SEHAT)']
)
print(report)