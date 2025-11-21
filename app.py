# app.py

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
# Import semua fungsi yang dibutuhkan dari recommender_core
from recommender_core import (
    load_data, load_ml_assets, calculate_station_similarity, 
    get_hybrid_recommendation, get_actual_recommendation, 
    highlight_historical_recommendation, 
    get_historical_pejabat_recommendation
)
# Mengambil STATION_COL_NAME dan fungsi normalisasi dari config
from config import STATION_COL_NAME, STATION_MAP, normalize_station 

# --- KONFIGURASI TEMA DAN JUDUL APLIKASI ---
APP_TITLE = "Atmosfera-X: Platform Intelligent Recommendation"

# --- FUNGSI TEMA DIHAPUS, HANYA FUNGSI UTAMA LAINNYA ---
def display_usage_guide():
    """Menampilkan panduan penggunaan aplikasi."""
    st.header("📖 Cara Penggunaan Aplikasi")
    st.markdown("---")
    
    st.subheader("1. Dashboard KPI Historis")
    st.write("Menyajikan analisis kinerja kualitas udara DKI Jakarta.")
    st.markdown(
        """
        - **Filter Tahun:** Gunakan filter di atas grafik untuk melihat tren PM2.5 rata-rata bulanan pada tahun-tahun tertentu.
        - **Tracking Historis:** Melihat catatan rekomendasi **Masyarakat** dan **Kebijakan Pejabat** yang seharusnya diberikan pada masa lalu.
        """
    )

    st.subheader("2. Sistem Rekomendasi Proaktif")
    st.write("Inti dari platform, memberikan panduan aksi cepat untuk kondisi saat ini dan masa depan.")
    st.markdown(
        """
        - **Aksi Cepat (Aktual):** Saran tindakan segera berdasarkan data terbaru yang tercatat.
        - **Prediksi (Proaktif):** Saran pencegahan 24 jam ke depan dari model **Hybrid** Anda (Content-Based Filtering).
        - **Rekomendasi Kebijakan:** Saran terperinci untuk **Pejabat Berwenang** (berdasarkan Fusi/Data Aktual) untuk tindakan Jangka Pendek (Darurat), Menengah (Mitigasi), dan Panjang (Rutin/Investasi).
        """
    )
    
# --- MAIN APP START ---
st.set_page_config(
    page_title=APP_TITLE,
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Muat Data dan Model
df_full = load_data()
scaler, cbf_model, fitur_list = load_ml_assets()

if df_full.empty:
    st.error("Gagal memuat data. Mohon pastikan file CSV dan model ada.")
    st.stop()
    
# Pengaturan Tema di Sidebar (DIHAPUS)

# Load aset
sim_df = calculate_station_similarity(df_full)

# --- MEMBUAT DAFTAR STASIUN UNIK DAN BERSIH ---
# 1. Terapkan normalisasi ke semua nama stasiun mentah di DataFrame
df_full['stasiun_normal'] = df_full[STATION_COL_NAME].astype(str).apply(normalize_station)

# 2. Ambil hanya nama stasiun yang unik dan sudah bersih
all_stations_clean = sorted(df_full['stasiun_normal'].unique().tolist())
# ===================================================================


# Header Utama Aplikasi
st.markdown(f"<h1>{APP_TITLE}</h1>", unsafe_allow_html=True)
st.markdown("Analisis Data Kualitas Udara DKI Jakarta (2020-2025)", unsafe_allow_html=True)
st.markdown("---")


# Sidebar untuk Navigasi
page = st.sidebar.radio("Pilih Tampilan", ["Cara Penggunaan", "Dashboard KPI Historis", "Sistem Rekomendasi Proaktif"])

if page == "Cara Penggunaan":
    display_usage_guide()

elif page == "Dashboard KPI Historis":
    
    st.header("📈 Dashboard KPI & Tracking Rekomendasi Historis")
    st.markdown("Lacak tren polusi dan efektivitas rekomendasi aktual di masa lalu.")
    st.markdown("---")
    
    # --- Bagian 1: KPI Historis dengan FILTER TAHUN ---
    all_years = sorted(df_full['tanggal_lengkap'].dt.year.unique())
    selected_years = st.multiselect("Filter Tahun", options=all_years, default=all_years)
    
    df_filtered = df_full[df_full['tanggal_lengkap'].dt.year.isin(selected_years)]
    
    if df_filtered.empty:
        st.warning("Tidak ada data untuk tahun yang dipilih.")
        st.stop()
        
    df_filtered['Bulan_Tahun'] = df_filtered['tanggal_lengkap'].dt.strftime('%Y-%m')
    kpi_monthly = df_filtered.groupby('Bulan_Tahun')['pm25'].mean().reset_index()
    
    # 3. Metrik
    col_metric_1, col_metric_2, col_metric_3 = st.columns(3)
    
    with col_metric_1:
        st.metric(label="Periode Analisis (Terfilter)", value=f"{min(selected_years)} - {max(selected_years)}")
    with col_metric_2:
        st.metric(label="PM2.5 Rata-rata Global (Terfilter)", value=f"{df_filtered['pm25'].mean():.2f}")
    with col_metric_3:
        # Gunakan stasiun normal untuk Kritis
        worst_station = df_filtered[df_filtered['kategori'] == 'TIDAK SEHAT']['stasiun_normal'].mode().iloc[0] if not df_filtered.empty else "N/A"
        st.metric(label="Stasiun Paling Kritis (Terfilter)", value=worst_station)

    # 4. Chart dengan Altair (untuk kontrol axis miring)
    st.subheader("Tren PM2.5 Rata-rata Bulanan")
    
    base = alt.Chart(kpi_monthly).encode(
        x=alt.X('Bulan_Tahun', axis=alt.Axis(title='Bulan dan Tahun', labelAngle=-45)),
        y=alt.Y('pm25', title='Rata-rata PM2.5 (µg/m³)'),
        tooltip=['Bulan_Tahun', alt.Tooltip('pm25', format='.2f')]
    )

    line = base.mark_line(color='#0056b3').properties(
        title='Perbandingan Kualitas Udara Bulanan'
    ).interactive()

    st.altair_chart(line, use_container_width=True)

    st.markdown("---")

    # --- Bagian 2: Tracking Rekomendasi Historis (FINAL) ---
    st.subheader("📚 Log Rekomendasi Historis (100 Data Terbaru)")
    
    # 1. Buat kolom Rekomendasi Aktual (Masyarakat)
    df_full['Rekomendasi_Aktual_Masyarakat'] = df_full['kategori'].apply(get_actual_recommendation)
    
    # 2. Buat kolom Rekomendasi Kebijakan (Pejabat)
    df_full['Rekomendasi_Kebijakan_Pejabat'] = df_full.apply(get_historical_pejabat_recommendation, axis=1)
    
    # 3. Kolom yang ditampilkan
    df_tracking = df_full[[
        'tanggal_lengkap', 'stasiun_normal', 'kategori', 'pm25', 'Rekomendasi_Aktual_Masyarakat', 'Rekomendasi_Kebijakan_Pejabat'
    ]].tail(100).sort_values('tanggal_lengkap', ascending=False).reset_index(drop=True)
    
    # 4. Tampilkan dengan styling
    st.dataframe(
        df_tracking.style.applymap(
            highlight_historical_recommendation, 
            subset=['Rekomendasi_Aktual_Masyarakat', 'Rekomendasi_Kebijakan_Pejabat']
        ), 
        use_container_width=True
    )


elif page == "Sistem Rekomendasi Proaktif":
    
    st.header("🔮 Sistem Rekomendasi Proaktif (Hybrid)")
    st.markdown("Gabungan rekomendasi **Aktual (sekarang)** dan **Prediksi (24 jam)** untuk tindakan proaktif.")
    st.markdown("---")
    
    # --- Input Widget Simulasi ---
    selected_station = st.selectbox("Pilih Stasiun Target", options=all_stations_clean) # Menggunakan all_stations_clean
    
    # Filter DataFrame menggunakan nama stasiun yang sudah dinormalisasi
    df_latest_by_station = df_full[df_full['stasiun_normal'] == selected_station].sort_values('tanggal_lengkap', ascending=False)
    
    if df_latest_by_station.empty:
        st.warning("Data tidak tersedia untuk stasiun ini.")
        st.stop()

    latest_data_row = df_latest_by_station.iloc[[0]]
    input_df_for_hybrid = latest_data_row
    
    tanggal_aktual = latest_data_row['tanggal_lengkap'].dt.strftime('%Y-%m-%d %H:%M:%S').iloc[0]
    kategori_aktual = latest_data_row['kategori'].iloc[0]
    
    st.info(f"Data Aktual Terakhir: **{selected_station}** pada **{tanggal_aktual}** (Kategori ISPU: **{kategori_aktual}**)")
    
    
    # --- KARTU REKOMENDASI MASYARAKAT ---
    st.subheader("👥 Rekomendasi untuk Masyarakat (Aksi Cepat & Proaktif)")
    
    col_aktual, col_prediksi = st.columns(2)
    
    # A. Rekomendasi AKTUAL (Kondisi Saat Ini)
    with col_aktual:
        st.markdown("#### 🗣️ Aksi Cepat (Kondisi AKTUAL)")
        rekomendasi_aktual = get_actual_recommendation(kategori_aktual)
        
        if '🔴' in rekomendasi_aktual or '🚨' in rekomendasi_aktual:
            st.error(f"### {rekomendasi_aktual}")
        elif '🟡' in rekomendasi_aktual:
            st.warning(f"### {rekomendasi_aktual}")
        else:
            st.success(f"### {rekomendasi_aktual}")
        
        st.caption(f"Saran ini berdasarkan kategori ISPU **{kategori_aktual}** yang tercatat saat ini.")

    # B. Rekomendasi PREDIKSI (Masa Depan)
    results_prediksi = get_hybrid_recommendation(
        input_df_for_hybrid, selected_station, sim_df, scaler, cbf_model, fitur_list
    )

    with col_prediksi:
        st.markdown("#### 🔭 Prediksi (Tindakan Proaktif 24 Jam)")
        
        status_prediksi = results_prediksi.get("Status Prediksi (CBF)")
        rekomendasi_prediksi = results_prediksi['Rekomendasi Tindakan Primer']

        if status_prediksi == "TIDAK SEHAT":
            st.warning(f"### {rekomendasi_prediksi}")
        else:
            st.success(f"### {rekomendasi_prediksi}")

        st.caption(f"Prediksi model CBF: **{status_prediksi}** (Probabilitas TIDAK SEHAT: **{results_prediksi.get('Probabilitas TIDAK SEHAT', 0)*100:.1f}%**).")
        
    st.markdown("---")
    
    
    # --- REKOMENDASI PEJABAT BERWENANG ---
    st.subheader("🏛️ Rekomendasi Kebijakan (Pejabat Berwenang)")
    
    rekomendasi_pejabat = results_prediksi['Rekomendasi Kebijakan (Pejabat)']
    
    col_pejabat_1, col_pejabat_2 = st.columns([2, 1])

    with col_pejabat_1:
        st.markdown("#### Saran Kebijakan Berbasis Prediksi:")
        if 'TINDAKAN DARURAT' in rekomendasi_pejabat:
            st.error(f"**🔴 PERINGATAN TINGKAT TERTINGGI (DARURAT):** {rekomendasi_pejabat}")
        elif 'PERKETAT UJI EMISI' in rekomendasi_pejabat:
            st.warning(f"**🟡 TINJAUAN SEGERA (MITIGASI):** {rekomendasi_pejabat}")
        else:
            st.info(f"**🟢 TINDAKAN RUTIN (MONITORING):** {rekomendasi_pejabat}")
        
    with col_pejabat_2:
        st.markdown("#### Data Pendukung:")
        st.metric(label="PM2.5 Aktual Terakhir", value=f"{latest_data_row['pm25'].iloc[0]:.2f} µg/m³")
        st.metric(label="Status Korelasi (CF)", value="Tinggi" if "Korelasi" in results_prediksi['Peringatan Situasional (CF)'] else "Normal")

    st.markdown("---")

    # --- Collaborative Filtering (CF) ---
    st.subheader("🔗 Data Insight (Collaborative Filtering)")
    st.caption(results_prediksi['Peringatan Situasional (CF)'])