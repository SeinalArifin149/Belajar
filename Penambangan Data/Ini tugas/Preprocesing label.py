import numpy as np
import pandas as pd
import os
import zipfile
import tempfile
import cv2
from tkinter import Tk, filedialog
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
import pywt

# --- Function untuk cek file gambar ---
def is_valid_image(file_name):
    return file_name.lower().endswith((".png", ".jpg", ".jpeg"))

# --- Ekstraksi fitur GLCM ---
def extract_glcm_features(img):
    glcm = graycomatrix(img, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    ASM = graycoprops(glcm, 'ASM')[0, 0]
    return [contrast, dissimilarity, homogeneity, energy, correlation, ASM]

# --- Ekstraksi fitur Wavelet Haar ---
def extract_wavelet_features(img):
    coeffs = pywt.dwt2(img, 'haar')
    cA, (cH, cV, cD) = coeffs
    features = [
        np.mean(cA), np.std(cA),
        np.mean(cH), np.std(cH),
        np.mean(cV), np.std(cV),
        np.mean(cD), np.std(cD)
    ]
    return features

# --- Preprocessing dan Ekstraksi Fitur ---
def preprocess_and_extract(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None or img.size == 0:
        print(f"Skipping {image_path}: Invalid or empty image")
        return None

    img = cv2.resize(img, (256, 256))

    glcm_features = extract_glcm_features(img)
    wavelet_features = extract_wavelet_features(img)

    features = glcm_features + wavelet_features
    return features

# --- Buka File ZIP atau Folder ---
def select_dataset():
    root = Tk()
    root.withdraw()
    path = filedialog.askdirectory(title="Pilih Folder Dataset atau ZIP")
    return path

# --- Load Semua Gambar ---
def load_images_from_path(path):
    image_paths = []
    if path.endswith('.zip'):
        tmpdir = tempfile.TemporaryDirectory()
        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir.name)
        for root, _, files in os.walk(tmpdir.name):
            for file in files:
                if is_valid_image(file):
                    image_paths.append(os.path.join(root, file))
    else:
        for root, _, files in os.walk(path):
            for file in files:
                if is_valid_image(file):
                    image_paths.append(os.path.join(root, file))
    return image_paths

# --- MAIN ---
path = select_dataset()
if path:
    image_paths = load_images_from_path(path)
    all_features = []
    image_names = []

    for img_path in image_paths:
        features = preprocess_and_extract(img_path)
        if features:
            all_features.append(features)
            image_names.append(os.path.basename(img_path))

    X = np.array(all_features)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    kmeans = KMeans(n_clusters=2, random_state=42)
    labels = kmeans.fit_predict(X_pca)

    # --- Plot Cluster ---
    plt.figure(figsize=(8,6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
    plt.title('Clustering Sel Cancer(2 Cluster)')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.colorbar(label='Cluster')
    plt.show()

    # --- Simpan Hasil ---
    result_df = pd.DataFrame({
        'Image Name': image_names,
        'Cluster Label': labels
    })
    result_df.to_csv("clustering_results.csv", index=False)
    print("Proses selesai. Hasil disimpan di clustering_results.csv")
else:
    print("Tidak ada file yang dipilih.")
