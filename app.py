# app.py
import os, pickle, streamlit as st
import numpy as np
import librosa

# ---- MUST MATCH TRAINING ORDER ----
FEATURE_COLS = [
    'mean_spectral_centroid','std_spectral_centroid',
    'mean_spectral_bandwidth','std_spectral_bandwidth',
    'mean_spectral_contrast','mean_spectral_flatness','mean_spectral_rolloff',
    'zero_crossing_rate','rms_energy',
    'mean_pitch','min_pitch','max_pitch','std_pitch',
    'spectral_skew','spectral_kurtosis',
    'energy_entropy','log_energy',
    'mfcc_1_mean','mfcc_1_std','mfcc_2_mean','mfcc_2_std',
    'mfcc_3_mean','mfcc_3_std','mfcc_4_mean','mfcc_4_std',
    'mfcc_5_mean','mfcc_5_std','mfcc_6_mean','mfcc_6_std',
    'mfcc_7_mean','mfcc_7_std','mfcc_8_mean','mfcc_8_std',
    'mfcc_9_mean','mfcc_9_std','mfcc_10_mean','mfcc_10_std',
    'mfcc_11_mean','mfcc_11_std','mfcc_12_mean','mfcc_12_std',
    'mfcc_13_mean','mfcc_13_std'
]

def extract_features(file_path, sr=22050):
    y, sr = librosa.load(file_path, sr=sr, mono=True)

    features = []

    # 1. Spectral
    sc = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features += [np.mean(sc), np.std(sc)]

    sb = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    features += [np.mean(sb), np.std(sb)]

    scon = librosa.feature.spectral_contrast(y=y, sr=sr)
    features += [np.mean(scon)]

    sflat = librosa.feature.spectral_flatness(y=y)[0]
    features += [np.mean(sflat)]

    sroll = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    features += [np.mean(sroll)]

    # 2. Temporal
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    features += [np.mean(zcr)]

    rms = librosa.feature.rms(y=y)[0]
    features += [np.mean(rms)]

    # 3. Pitch
    pitches, mags = librosa.piptrack(y=y, sr=sr)
    mask = mags > np.median(mags)
    p = pitches[mask]
    if p.size:
        features += [np.mean(p), np.min(p), np.max(p), np.std(p)]
    else:
        features += [0, 0, 0, 0]

    # 4. Higher order spectral stats
    S = np.abs(librosa.stft(y))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    mu = np.mean(S_db)
    features += [np.mean((S_db - mu)**3), np.mean((S_db - mu)**4)]

    # 5. Energy
    frame_length, hop_length = 2048, 512
    energy = np.array([np.sum(y[i:i+frame_length]**2)
                       for i in range(0, len(y), hop_length)])
    if energy.size:
        p_energy = energy / (energy.sum() + 1e-12)
        entropy = -np.sum(p_energy * np.log2(p_energy + 1e-12))
        features += [entropy, np.log(energy.sum() + 1e-12)]
    else:
        features += [0, 0]

    # 6. MFCCs (13×[mean,std] = 26)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        features += [np.mean(mfccs[i]), np.std(mfccs[i])]

    features = np.array(features, dtype=float).reshape(1, -1)
    assert features.shape[1] == 43, f"Got {features.shape[1]} features, expected 43"
    return features

# ---- Load the ONE pipeline we trained above ----
with open("rf_gender_pipeline.pkl", "rb") as f:
    pipe = pickle.load(f)

st.title("🎙️ Human Voice Classifier")
uploaded = st.file_uploader("Upload Audio File", type=["wav", "mp3", "ogg"])

if uploaded is not None:
    # Save temp
    file_path = os.path.join(os.path.dirname(__file__), "temp.wav")
    with open(file_path, "wb") as f:
        f.write(uploaded.read())
    st.write(f"✅ Saved: {file_path}")

    try:
        X = extract_features(file_path)          # RAW 43-dim vector
        st.write("Raw features (first 8):", X[0][:8])

        # Optional: peek at scaled values (for sanity)
        X_scaled = pipe.named_steps['scaler'].transform(X)
        st.write("Scaled (first 8):", X_scaled[0][:8])

        pred = pipe.predict(X)[0]                # pipeline scales internally
        st.success(f"Prediction: **{'Male' if pred==1 else 'Female'}**")

    except Exception as e:
        st.error(f"Error: {e}")
