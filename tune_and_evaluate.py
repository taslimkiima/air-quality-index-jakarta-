from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import joblib

# --- 1. KONFIGURASI DAN MUAT ASET ---
FILE_ADVANCED = 'data_ispu_preprocess_final_ADVANCED.csv'
MODEL_CBF_PATH = 'model_cbf_rekomendasi.pkl'
SCALER_PATH = 'scaler_rekomendasi.pkl'
FITUR_LIST_PATH = 'fitur_list.pkl'

# --- PARAMETER TUNING ---
# Ambang Batas Prediksi Baru (Default adalah 0.5)
# Kita naikkan untuk mengurangi False Positives (Alarm Palsu)
NEW_THRESHOLD = 0.70 

try:
    df_clean = pd.read_csv(FILE_ADVANCED)
    cbf_model = joblib.load(MODEL_CBF_PATH)
    scaler = joblib.load(SCALER_PATH)
    fitur_list = joblib.load(FITUR_LIST_PATH)
except FileNotFoundError as e:
    print(f"❌ ERROR: Aset tidak ditemukan. Pastikan semua file (.csv, .pkl) sudah dibuat.")
    print(f"Detail: {e}")
    exit()

# --- 2. PERSIAPAN DATA UJI ---
# Definisikan X dan Y (sesuai yang digunakan saat pelatihan)
X = df_clean[fitur_list].fillna(df_clean[fitur_list].mean())
Y = df_clean['kategori_TIDAK SEHAT'] 

# Scaling dan Pembagian Data Uji (Wajib random_state=42 yang sama)
X_scaled = scaler.transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(
    X_scaled, Y, test_size=0.2, random_state=42
)

# --- 3. EVALUASI DAN TUNING MODEL ---
print("--- 🛠️ EVALUASI SETELAH PENYESUAIAN THRESHOLD ---")

# Prediksi Probabilitas
# Mengambil probabilitas bahwa kelasnya adalah 1 (TIDAK SEHAT)
Y_proba = cbf_model.predict_proba(X_test)[:, 1] 

# Menyesuaikan Prediksi berdasarkan NEW_THRESHOLD
Y_tuned_pred = (Y_proba >= NEW_THRESHOLD).astype(int)

# 1. Confusion Matrix
print(f"\n1. Confusion Matrix (Amb. Batas = {NEW_THRESHOLD}):")
cm = confusion_matrix(Y_test, Y_tuned_pred)
print(cm)
print("[Baris: Aktual; Kolom: Prediksi] | [0: AMAN/SEDANG, 1: TIDAK SEHAT]")


# 2. Classification Report (Metrik Kunci)
print("\n2. Classification Report:")
report = classification_report(
    Y_test, Y_tuned_pred, target_names=['0 (AMAN/SEDANG)', '1 (TIDAK SEHAT)']
)
print(report)

# --- 4. Interpretasi Perubahan ---
tn, fp, fn, tp = cm.ravel()
new_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
new_recall = tp / (tp + fn) if (tp + fn) > 0 else 0

print("\n--- 💡 ANALISIS PERUBAHAN METRIK ---")
print(f"Precision Awal (0.55) vs. Tuning ({new_precision:.2f}): Perubahan ini bertujuan untuk mengurangi Alarm Palsu.")
print(f"Recall Awal (0.89) vs. Tuning ({new_recall:.2f}): Recall mungkin menurun sedikit, tapi keandalan meningkat.")