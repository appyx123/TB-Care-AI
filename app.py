import streamlit as st
import pandas as pd
import numpy as np
import librosa
import tensorflow as tf
import pickle
import os
from tensorflow.keras.models import load_model

# ==========================================
# 1. KONFIGURASI
# ==========================================
SAMPLE_RATE = 16000
DURATION = 0.5 
N_MFCC = 40
N_MELS = 128
THRESHOLD = 0.4771 

# --- KONFIGURASI BOBOT (STRATEGI BARU) ---
WEIGHT_FORM = 0.60  # Form memegang kendali 70%
WEIGHT_AI = 0.40    # Audio hanya pengaruh 30%

# Custom Loss (Hanya formalitas untuk load model)
def binary_focal_loss(gamma=2., alpha=0.25):
    from tensorflow.keras import backend as K
    def binary_focal_loss_fixed(y_true, y_pred):
        return K.binary_crossentropy(y_true, y_pred)
    return binary_focal_loss_fixed

# ==========================================
# 2. LOGIKA: KLINIS & AUDIO
# ==========================================

# [BARU] Fungsi Menghitung Skor Klinis Manual
def calculate_clinical_score(age, tb_prior, cough_2w, hemoptysis, weight_loss, fever, sweat, smoke):
    score = 0
    # Bobot Gejala (Prioritas Medis)
    if hemoptysis == "Ya": score += 30   # Tanda bahaya utama
    if weight_loss == "Ya": score += 20  # Gejala sistemik
    if sweat == "Ya": score += 15        # Khas TBC
    if cough_2w == "Ya": score += 15     # Definisi suspek
    if tb_prior == "Ya": score += 10     # Risiko kambuh
    if fever == "Ya": score += 5
    
    # Faktor Risiko
    if smoke == "Ya": score += 5
    if age > 60 or age < 5: score += 5
    
    # Normalisasi jadi 0.0 - 1.0 (Max skor teoritis ~105, kita cap di 100)
    final_score = min(score / 100.0, 1.0)
    return final_score

def extract_segment_features(audio_segment, sr):
    """Mengubah potongan audio mentah menjadi fitur siap input model"""
    try:
        # TILING (Ulang 4x agar jadi 2 detik)
        audio_tiled = np.tile(audio_segment, 4) 

        # Ekstraksi LSTM (MFCC)
        mfcc = librosa.feature.mfcc(y=audio_tiled, sr=sr, n_mfcc=N_MFCC)
        zcr = librosa.feature.zero_crossing_rate(y=audio_tiled)
        rms = librosa.feature.rms(y=audio_tiled)
        feat_1d = np.vstack([mfcc, zcr, rms]).T 

        # Ekstraksi CNN (Spectrogram)
        melspec = librosa.feature.melspectrogram(y=audio_tiled, sr=sr, n_mels=N_MELS)
        melspec_db = librosa.power_to_db(melspec, ref=np.max)
        min_val, max_val = melspec_db.min(), melspec_db.max()
        melspec_norm = (melspec_db - min_val) / (max_val - min_val + 1e-8)
        feat_2d = melspec_norm[..., np.newaxis]

        return feat_1d, feat_2d
    except:
        return None, None

def process_long_audio(uploaded_file):
    """Memecah audio panjang menjadi beberapa chunk 0.5 detik"""
    try:
        audio, sr = librosa.load(uploaded_file, sr=SAMPLE_RATE)
        target_len = int(SAMPLE_RATE * DURATION)
        total_len = len(audio)
        
        chunks_1d = []
        chunks_2d = []
        
        # A. JIKA AUDIO PENDEK (< 0.5s) -> Padding
        if total_len < target_len:
            audio_pad = np.pad(audio, (0, target_len - total_len))
            f1, f2 = extract_segment_features(audio_pad, sr)
            if f1 is not None:
                chunks_1d.append(f1); chunks_2d.append(f2)
                
        # B. JIKA AUDIO PANJANG (> 0.5s) -> Sliding Window
        else:
            num_chunks = int(np.ceil(total_len / target_len))
            for i in range(num_chunks):
                start = i * target_len
                end = start + target_len
                segment = audio[start:end]
                
                if len(segment) < target_len:
                    segment = np.pad(segment, (0, target_len - len(segment)))
                
                # Skip silence (biar gak nambah bias negatif)
                if np.max(np.abs(segment)) > 0.01:
                    f1, f2 = extract_segment_features(segment, sr)
                    if f1 is not None:
                        chunks_1d.append(f1); chunks_2d.append(f2)
        
        if len(chunks_1d) == 0: return None, None
        return np.array(chunks_1d), np.array(chunks_2d)

    except Exception as e:
        return None, None

# ==========================================
# 3. LOAD RESOURCE
# ==========================================
@st.cache_resource
def load_resources():
    model = load_model('tb_multimodal_final.keras', compile=False)
    with open('age_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

try:
    model, scaler = load_resources()
except:
    st.error("❌ File model/scaler hilang! Pastikan ada di folder yang sama.")
    st.stop()

# ==========================================
# 4. TAMPILAN WEB (UI) - Tetap Seperti Semula
# ==========================================
st.set_page_config(page_title="TB Care AI", page_icon="🫁", layout="centered")

st.markdown("<h1 style='text-align: center; color: #2E86C1;'>🫁 TB Care AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Sistem Skrining Tuberkulosis Berbasis Multimodal Deep Learning</p>", unsafe_allow_html=True)
st.divider()

with st.form("main_form"):
    st.subheader("1. Data Pasien")
    
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Usia", 1, 100, 25)
        sex = st.selectbox("Gender", ["Laki-laki", "Perempuan"])
        tb_prior = st.radio("Riwayat TB Sebelumnya?", ["Tidak", "Ya"])
        hemoptysis = st.radio("Batuk Berdarah?", ["Tidak", "Ya"])
        
    with c2:
        cough_2w = st.radio("Batuk > 2 Minggu?", ["Tidak", "Ya"])
        fever = st.radio("Demam / Menggigil?", ["Tidak", "Ya"])
        weight_loss = st.radio("BB Turun Drastis?", ["Tidak", "Ya"])
        night_sweats = st.radio("Keringat Malam?", ["Tidak", "Ya"])
        smoke = st.radio("Perokok Aktif/Pasif?", ["Tidak", "Ya"])

    st.divider()
    st.subheader("2. Analisis Suara")
    st.info("💡 **Tips:** Rekam batuk sejelas mungkin. Durasi bebas.")
    audio_file = st.file_uploader("Upload File (WAV/MP3)", type=['wav', 'mp3', 'ogg'])

    analyze = st.form_submit_button("🔍 MULAI ANALISIS", type="primary")

# ==========================================
# 5. EKSEKUSI PREDIKSI (Logika Baru)
# ==========================================
if analyze:
    if audio_file is None:
        st.warning("⚠️ Mohon upload suara batuk dulu.")
    else:
        with st.spinner("⏳ Sedang memproses sinyal suara & data klinis..."):
            # --- STEP A: PREDIKSI SUARA (AI) ---
            # Prepare Tabular Input for Model
            def map_val(x): return 1 if x == "Ya" else 0
            in_age = scaler.transform([[age]])[0][0]
            in_sex = 1 if sex == "Laki-laki" else 0
            
            tab_features = np.array([[
                in_age, in_sex, 
                map_val(tb_prior), map_val(cough_2w), map_val(hemoptysis), 
                map_val(weight_loss), map_val(fever), map_val(night_sweats), map_val(smoke)
            ]])

            chunks_1d, chunks_2d = process_long_audio(audio_file)
            
            ai_risk_score = 0.0
            if chunks_1d is not None and len(chunks_1d) > 0:
                # Batch predict
                num_chunks = len(chunks_1d)
                tab_features_batch = np.repeat(tab_features, num_chunks, axis=0)
                probs = model.predict([chunks_1d, chunks_2d, tab_features_batch], verbose=0)
                
                # Ambil skor tertinggi dari audio
                ai_risk_score = np.max(probs)
            
            # --- STEP B: HITUNG SKOR KLINIS (MANUAL) ---
            # Kita hitung ulang berdasarkan bobot gejala untuk mengontrol AI
            clinical_risk_score = calculate_clinical_score(
                age, tb_prior, cough_2w, hemoptysis, weight_loss, fever, night_sweats, smoke
            )
            
            # --- STEP C: FUSION (PENGGABUNGAN) ---
            # Cek Red Flag dulu (Batuk Darah = Bahaya)
            if hemoptysis == "Ya":
                final_prob = 0.95 # Paksa Tinggi
                note = "⚠️ **PERHATIAN:** Gejala batuk berdarah terdeteksi. Sistem memprioritaskan ini sebagai Risiko Tinggi."
            else:
                # Rumus Weighted: (70% Form) + (30% Audio)
                final_prob = (clinical_risk_score * WEIGHT_FORM) + (ai_risk_score * WEIGHT_AI)
                note = "✅ Analisis gabungan Form & Audio selesai."

            # --- STEP D: TAMPILAN HASIL ---
            st.divider()
            
            col_res1, col_res2 = st.columns([1, 2])
            
            with col_res1:
                # Tentukan Label
                if final_prob > THRESHOLD:
                    st.error("## 🔴 POSITIF")
                    st.metric("Tingkat Risiko", f"{final_prob*100:.1f}%")
                else:
                    st.success("## 🟢 NEGATIF")
                    st.metric("Tingkat Risiko", f"{final_prob*100:.1f}%")
            
            with col_res2:
                st.write("**Rincian Penilaian:**")
                # Tampilkan progress bar gabungan
                st.progress(float(final_prob), text="Total Skor Risiko")
                
                # Tampilkan Breakdown agar transparan
                st.write(f"- 📋 Skor Gejala (Form): **{clinical_risk_score*100:.0f}%**")
                st.write(f"- 🎤 Skor Suara (AI): **{ai_risk_score*100:.0f}%**")
                
                st.markdown("---")
                if final_prob > THRESHOLD:
                    st.write(note)
                    st.warning("⚠️ **Rekomendasi:** Segera lakukan pemeriksaan dahak (TCM) di Faskes terdekat.")
                else:
                    st.success("✅ **Rekomendasi:** Tidak ditemukan indikasi kuat TB. Jaga kesehatan dan gunakan masker jika masih batuk.")