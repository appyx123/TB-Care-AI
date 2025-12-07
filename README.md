# 🫁 TB Care AI: Multimodal Tuberculosis Screening

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![License](https://img.shields.io/badge/License-MIT-green)

**TB Care AI** adalah sistem purwarupa (prototype) untuk skrining awal penyakit Tuberkulosis (TB) menggunakan pendekatan **Multimodal Fusion**. Sistem ini menggabungkan analisis sinyal suara batuk (menggunakan Deep Learning) dan data klinis pasien untuk menghasilkan skor risiko yang lebih akurat dan mengurangi tingkat kesalahan diagnosis awal.

---

## 🌟 Fitur Utama

1. **Analisis Audio Batuk (AI):**
   * Menggunakan model Deep Learning (CNN + LSTM).
   * Ekstraksi fitur audio canggih: **MFCC** (Mel-frequency cepstral coefficients) dan **Mel-Spectrogram**.
   * Memecah audio panjang menjadi segmen 0.5 detik untuk analisis mendetail.

2. **Analisis Data Klinis (Rule-Based):**
   * Mempertimbangkan faktor risiko medis seperti usia, riwayat TB, gejala batuk > 2 minggu, keringat malam, dan penurunan berat badan.
   * Mendeteksi tanda bahaya ("Red Flag") seperti batuk berdarah.

3. **Hybrid Decision Logic:**
   * Menggunakan strategi *weighted fusion* (70% Skor Klinis + 30% Skor AI) untuk meminimalkan *false positive*.
   * Memprioritaskan gejala medis kritis di atas prediksi AI murni.

---

## 🛠️ Persyaratan Sistem

Sebelum menjalankan aplikasi, pastikan komputer Anda memiliki:

* **Python 3.8 - 3.10**
* **FFmpeg** (Wajib untuk memproses file audio MP3/WAV)

### Cara Install FFmpeg:
* **Windows:** [Download FFmpeg](https://ffmpeg.org/download.html), ekstrak, lalu tambahkan folder `bin` ke `PATH` environment variables.
* **Linux (Ubuntu/Debian):** `sudo apt-get install ffmpeg`
* **Mac:** `brew install ffmpeg`

---

## 🚀 Cara Menjalankan Project

Ikuti langkah-langkah berikut untuk menjalankan aplikasi di komputer lokal Anda:

### 1. Clone Repository
```bash
git clone https://github.com/appyx123/TB-Care-AI.git
cd TB-Care-AI
```

### 2. Siapkan Virtual Environment
Sangat disarankan menggunakan virtual environment agar library tidak bentrok dengan sistem lain.

**Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Library Python
Install semua dependensi yang diperlukan melalui file `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 4. Pastikan File Model Tersedia
Pastikan dua file berikut ada di dalam folder utama proyek (root directory):
* `tb_multimodal_final.keras` (File Model Deep Learning)
* `age_scaler.pkl` (File Scaler untuk normalisasi input usia)

**Catatan:** Jika Anda mendownload repo ini dari GitHub, file model mungkin tidak disertakan karena ukurannya yang besar. Pastikan Anda menyalin file model hasil training ke folder ini.

### 5. Jalankan Aplikasi
```bash
streamlit run main.py
```

Aplikasi akan otomatis terbuka di browser Anda (biasanya di `http://localhost:8501`).

---

## 🧠 Penjelasan Teknis (Logic)

Sistem ini menggunakan logika penggabungan skor (Score Fusion) untuk menentukan hasil akhir:

1. **AI Score ($S_{ai}$):** Model Deep Learning memproses audio batuk. Probabilitas tertinggi dari potongan audio diambil sebagai skor risiko audio (Range: 0.0 - 1.0).

2. **Clinical Score ($S_{clin}$):** Algoritma manual menghitung skor berdasarkan bobot gejala (Contoh: Batuk Darah = +30 poin, Keringat Malam = +15 poin). Skor dinormalisasi menjadi 0.0 - 1.0.

3. **Final Decision ($S_{final}$):**
   
   Rumus penggabungan yang digunakan adalah:
   
   $$S_{final} = (0.7 \times S_{clin}) + (0.3 \times S_{ai})$$
   
   **Pengecualian:** Jika terdeteksi gejala **Batuk Berdarah (Hemoptysis)**, sistem akan mengabaikan perhitungan di atas dan memaksa skor risiko menjadi **0.95 (Sangat Berisiko)** sebagai langkah keamanan medis.

---

## 📂 Struktur Folder

```
TB-Care-AI/
├── .gitignore              # File yang diabaikan oleh Git (venv, __pycache__, dll)
├── README.md               # Dokumentasi proyek
├── requirements.txt        # Daftar library Python yang dibutuhkan
├── main.py                 # Kode utama aplikasi Streamlit (Backend & Frontend)
├── age_scaler.pkl          # Scaler Scikit-Learn (Wajib ada)
└── tb_multimodal_final.keras  # Model Deep Learning (Wajib ada)
```

---

## ⚠️ Disclaimer

⚠️ **PENTING:**

* Aplikasi ini dikembangkan untuk tujuan **penelitian dan purwarupa (research prototype)**.
* Aplikasi ini **BUKAN pengganti diagnosis medis profesional**.
* Hasil **POSITIF** menunjukkan indikasi risiko dan memerlukan **pemeriksaan lanjut (TCM/Rontgen)**.
* Hasil **NEGATIF** tidak menjamin bebas sepenuhnya dari penyakit.
* **Selalu konsultasikan masalah kesehatan Anda dengan dokter di fasilitas kesehatan terdekat.**

---

## 👨‍💻 Author

Dikembangkan oleh **[Muhammad Rafli dan Nahwa Kaka Saputra Anggareksa]** untuk Riset Deteksi Dini Tuberkulosis.

---

## 📄 License

Project ini dilisensikan di bawah [MIT License](LICENSE).

---

## 🤝 Kontribusi

Kontribusi, issue, dan feature request sangat diterima! Silakan buka [issues page](https://github.com/appyx123/TB-Care-AI/issues) untuk diskusi.

---

## 📧 Kontak

Jika ada pertanyaan atau saran, silakan hubungi:
* Email: [Raflyofficial6122@Gmail.com]
* GitHub: [@appyx123](https://github.com/appyx123)
