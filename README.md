
## 🧠 Arsitektur & Metodologi Utama

1.  **Content-Based Filtering (CBF):**
      * **Tujuan:** Prediksi dini status $\mathbf{TIDAK\ SEHAT}$ (24 jam ke depan).
      * **Kekuatan:** Model mencapai $\mathbf{Recall\ 92\%}$, yang sangat penting untuk meminimalkan risiko bahaya polusi yang terlewatkan.
2.  **Collaborative Filtering (CF):**
      * **Tujuan:** Peringatan Situasional. Mengidentifikasi pola polusi yang berkorelasi tinggi antar stasiun (Cosine Similarity).
3.  ***Rule-Based Fusion:***
      * **Tujuan:** Menghasilkan saran aksi spesifik (merah/kuning/hijau) berdasarkan ambang batas risiko dan konteks data (misalnya, **$\mathbf{\text{PM}_{2.5}} \geq 70$ pada Hari Kerja**).

### 🎯 Output Kunci Aplikasi

| Modul | Tujuan Output | Contoh Visualisasi |
| :--- | :--- | :--- |
| **Sistem Proaktif** | Memberikan $\mathbf{Aksi\ Cepat}$ (Masyarakat) dan $\mathbf{Saran\ Intervensi\ Kebijakan}$ terstruktur (Pejabat). | Kartu $\mathbf{Darurat}$ (🔴) untuk WFH, atau $\mathbf{Mitigasi}$ (🟡) untuk Uji Emisi. |
| **Dashboard KPI** | Menyajikan analisis tren jangka panjang ($\mathbf{2020}-\mathbf{2025}$) dan Log *Tracking* Rekomendasi Historis. | Grafik $\mathbf{PM}_{2.5}$ Bulanan dan Metrik Stasiun Paling Kritis. |

-----

## 🚀 Panduan Eksekusi Lokal

### Persyaratan Sistem

  * Python 3.8+
  * `pip` dan `git` terinstal

### Langkah 1: Kloning Repositori

```bash
git clone https://github.com/taslimkiima/air-quality-hybrid-recommender.git
cd air-quality-hybrid-recommender
```

### Langkah 2: Instalasi Dependensi

```bash
pip install -r requirements.txt
```

*(Catatan: Anda perlu membuat file `requirements.txt` dari lingkungan Anda: `pip freeze > requirements.txt`)*

### Langkah 3: Persiapan Aset Data & Model

Sistem bergantung pada data *preprocessing* dan model yang sudah dilatih. Pastikan file-file berikut ada di direktori utama Anda:

  * `data_ispu_preprocess_final_ADVANCED.csv`
  * `model_cbf_rekomendasi.pkl`
  * `scaler_rekomendasi.pkl`
  * `fitur_list.pkl`

### Langkah 4: Jalankan Aplikasi Streamlit

```bash
streamlit run app.py
```

Aplikasi akan terbuka secara otomatis di *browser* Anda.

-----

