import pandas as pd

# Nama file yang akan dicek
FILE_DATA = 'data_kualitas_udara_gabungan_final.csv'

def cek_duplikat_data(file_path):
    print(f"--- 📊 MEMUAT DATA DARI: {file_path} ---")
    
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"❌ ERROR: File '{file_path}' tidak ditemukan.")
        return

    # --- 1. Persiapan Kunci Primer (Primary Key) ---
    
    # Konversi kolom tanggal menjadi format datetime yang benar
    df['tanggal_lengkap'] = pd.to_datetime(df['tanggal_lengkap'], errors='coerce')
    
    # Hitung kolom jam 
    if 'jam' not in df.columns:
        df['jam'] = df['tanggal_lengkap'].dt.hour.fillna(0).astype(int) 
    
    # Kunci Primer yang harus unik dalam Time Series: Lokasi + Waktu
    PRIMARY_KEY = ['stasiun', 'tanggal_lengkap', 'jam']
    
    print(f"\nTotal Baris Awal: {len(df)}")
    
    # --- 2. Cek Duplikat Sempurna (Semua Kolom Sama) ---
    
    duplikat_sempurna = df.duplicated().sum()
    print("\n--- Cek 1: Duplikat Sempurna (Absolute Duplicate) ---")
    print(f"Jumlah Duplikat Sempurna: {duplikat_sempurna}")
    
    if duplikat_sempurna > 0:
        # Tampilkan 5 contoh baris yang duplikat (Menggunakan to_string)
        print("\nContoh 5 Duplikat Sempurna (Termasuk baris pertamanya):")
        display_cols = ['stasiun', 'tanggal_lengkap', 'kategori', 'pm25']
        duplikat_df = df[df.duplicated(keep=False)].sort_values(by=PRIMARY_KEY).head(5)
        print(duplikat_df[display_cols].to_string(index=False))
        
    # --- 3. Cek Duplikat Kunci Primer (Inkonsistensi Time Series) ---
    
    # Hitung baris yang memiliki Duplikat Kunci Primer
    duplikat_kunci = df.duplicated(subset=PRIMARY_KEY, keep='first').sum()
    
    print("\n--- Cek 2: Duplikat Kunci Primer (Inkonsistensi Waktu/Lokasi) ---")
    print(f"Jumlah Baris yang Tumpang Tindih pada Kunci Primer: {duplikat_kunci}")
    
    if duplikat_kunci > 0:
        print("\nContoh 5 Duplikat Kunci Primer (Waktu yang Sama, Data Berbeda/Sama):")
        display_cols = ['stasiun', 'tanggal_lengkap', 'jam', 'pm25', 'kategori']
        
        # Ambil semua baris yang terlibat dalam duplikasi kunci (keep=False)
        duplikat_kunci_df = df[df.duplicated(subset=PRIMARY_KEY, keep=False)].sort_values(by=PRIMARY_KEY).head(5)
        
        # Tampilkan 5 contoh (Menggunakan to_string)
        print(duplikat_kunci_df[display_cols].to_string(index=False))
        
    print("\n--- ✅ Pengecekan Duplikat Selesai ---")

# --- EKSEKUSI UTAMA ---
if __name__ == '__main__':
    cek_duplikat_data(FILE_DATA)