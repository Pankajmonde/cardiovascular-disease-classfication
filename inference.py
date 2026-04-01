import os
import numpy as np
import librosa
import joblib
from scipy.stats import skew, kurtosis
from scipy.interpolate import interp1d
from skimage.feature import graycomatrix, graycoprops

# Paths
MODEL_DIR = "models"
SVM_MODEL_PATH = os.path.join(MODEL_DIR, "svm_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
INDICES_PATH = os.path.join(MODEL_DIR, "selected_indices.npy")

loaded_model = None
loaded_scaler = None
selected_indices = None

def load_models():
    global loaded_model, loaded_scaler, selected_indices
    if loaded_model is None:
        if not os.path.exists(SVM_MODEL_PATH):
            raise FileNotFoundError("Model artifacts not found. Please train the model first by running Cardio.py")
        loaded_model = joblib.load(SVM_MODEL_PATH)
        loaded_scaler = joblib.load(SCALER_PATH)
        selected_indices = np.load(INDICES_PATH)

SAMPLE_RATE = 4000
N_MFCC = 13

def create_iir_cqt_spectrogram(audio_signal, sr, n_bins=68, bins_per_octave=12):
    if len(audio_signal) < 8192:
        audio_signal = np.pad(audio_signal, (0, 8192 - len(audio_signal)), 'constant')
    cqt = np.abs(librosa.cqt(audio_signal, sr=sr, n_bins=n_bins, bins_per_octave=bins_per_octave))
    cqt_db = librosa.power_to_db(cqt**2, ref=np.max)
    if cqt_db.shape[1] != 100:
        f = interp1d(np.linspace(0, 1, cqt_db.shape[1]), cqt_db, axis=1, kind='linear')
        cqt_db = f(np.linspace(0, 1, 100))
    return cqt_db

def preprocess_spectrogram(spectrogram):
    mu_I = np.mean(spectrogram)
    sigma_I = np.std(spectrogram)
    return (spectrogram - mu_I) / (sigma_I + 1e-8)

def extract_spectral_features(spectrogram, audio_signal, sr):
    mfcc = librosa.feature.mfcc(y=audio_signal, sr=sr, n_mfcc=N_MFCC)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta_mean = np.mean(mfcc_delta, axis=1)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio_signal, sr=sr))
    
    stft = np.abs(librosa.stft(audio_signal))
    n_fft = stft.shape[0]
    low_band = np.sum(stft[:n_fft//3, :])
    mid_band = np.sum(stft[n_fft//3:2*n_fft//3, :])
    high_band = np.sum(stft[2*n_fft//3:, :])
    total_energy = low_band + mid_band + high_band
    band_energy_ratio = total_energy / (np.sum(stft) + 1e-8)
    
    return np.concatenate([mfcc_mean, mfcc_delta_mean, [spectral_centroid], [band_energy_ratio]])
    
def extract_texture_features(spectrogram):
    spec_norm = ((spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min() + 1e-8) * 255).astype(np.uint8)
    if spec_norm.shape[0] > 128 or spec_norm.shape[1] > 128:
        spec_norm = spec_norm[::2, ::2]
    glcm = graycomatrix(spec_norm, [1], [0], levels=256, symmetric=True, normed=True)
    return np.array([
        graycoprops(glcm, 'contrast')[0, 0],
        graycoprops(glcm, 'correlation')[0, 0],
        graycoprops(glcm, 'energy')[0, 0],
        graycoprops(glcm, 'homogeneity')[0, 0]
    ])

def extract_statistical_features(spectrogram):
    spec_flat = spectrogram.flatten()
    mean_val = np.mean(spec_flat)
    variance_val = np.var(spec_flat)
    skewness_val = skew(spec_flat)
    kurtosis_val = kurtosis(spec_flat)
    spec_normalized = np.abs(spec_flat) / (np.sum(np.abs(spec_flat)) + 1e-8)
    entropy_val = -np.sum(spec_normalized * np.log(spec_normalized + 1e-8))
    return np.array([mean_val, variance_val, skewness_val, kurtosis_val, entropy_val])

def extract_iir_cqt_features(spectrogram):
    S_cqt = spectrogram
    epsilon = 1e-8
    max_val = np.max(S_cqt)
    min_val = np.min(S_cqt)
    cqt_contrast = (max_val - min_val) / (max_val + min_val + epsilon)
    
    T = S_cqt.shape[1]
    temporal_stability = sum(np.linalg.norm(S_cqt[:, t+1] - S_cqt[:, t]) for t in range(T - 1)) / (T - 1)
    
    total_power = np.sum(np.abs(S_cqt)**2)
    harmonic_indices = np.arange(0, S_cqt.shape[0], S_cqt.shape[0]//5)
    harmonic_power = np.sum(np.abs(S_cqt[harmonic_indices, :])**2)
    harmonic_coherence = harmonic_power / (total_power + epsilon)
    
    return np.array([cqt_contrast, temporal_stability, harmonic_coherence])

def predict_audio(audio_path):
    load_models()
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    
    cqt_spec = create_iir_cqt_spectrogram(y, sr)
    spec_processed = preprocess_spectrogram(cqt_spec)
    
    spectral_feats = extract_spectral_features(spec_processed, y, sr)
    texture_feats = extract_texture_features(spec_processed)
    statistical_feats = extract_statistical_features(spec_processed)
    cqt_feats = extract_iir_cqt_features(spec_processed)
    
    F = np.concatenate([spectral_feats, texture_feats, statistical_feats, cqt_feats])
    F = F.reshape(1, -1)
    
    F_normalized = loaded_scaler.transform(F)
    F_selected = F_normalized[:, selected_indices]
    
    prediction = loaded_model.predict(F_selected)[0]
    proba = loaded_model.predict_proba(F_selected)[0]
    
    confidence = float(max(proba))
    result_class = "Abnormal" if prediction == 1 else "Normal"
    
    details = {
        'spectral_centroid': float(spectral_feats[26]),
        'cqt_contrast': float(cqt_feats[0]),
        'temporal_stability': float(cqt_feats[1]),
        'entropy': float(statistical_feats[4])
    }
    
    return {
        'prediction': result_class,
        'confidence': confidence,
        'features': details
    }
