import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import pywt
import tkinter as tk
from tkinter import filedialog
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import threshold_otsu
from skimage.transform import resize
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- PILIH FOLDER GAMBAR DENGAN TKINTER ---
root = tk.Tk()
root.withdraw()
folder_selected = filedialog.askdirectory(title='Pilih folder gambar untuk prediksi')
print(f"Folder yang dipilih: {folder_selected}")

# --- EKSTRAKSI FITUR PADA CITRA ---
def extract_features(img):
    # Resize gambar ke 150x150 piksel
    img_resized = resize(img, (150, 150), anti_aliasing=True)

    # Konversi ke grayscale (hitam putih)
    gray = rgb2gray(img_resized)

    # --- TRANSFORMASI WAVELET (HAAR) ---
    coeffs = pywt.wavedec2(gray, 'haar', level=2)
    cA2, (cH2, cV2, cD2), _ = coeffs
    wave_feat = np.concatenate([
        cA2.flatten(), cH2.flatten(), cV2.flatten(), cD2.flatten()
    ])

    # --- GLCM (opsional, kalau ingin tambahan fitur tekstur) ---
    gray_uint8 = (gray * 255).astype(np.uint8)
    glcm = graycomatrix(gray_uint8, distances=[1], angles=[0], symmetric=True, normed=True)
    glcm_props = [graycoprops(glcm, prop).ravel()[0] for prop in (
        'contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM'
    )]

    # Gabungkan fitur wavelet dan GLCM (opsional)
    feature_vector = np.concatenate([
        wave_feat[:2000],  # Batasi panjang agar efisien
        glcm_props         # Fitur tekstur
    ])
    return feature_vector, wave_feat, glcm_props

# --- MUAT DATA DAN LABELNYA DARI FOLDER ---
Categories = ['GANAS', 'JINAK']
features, labels, wavelet_data, glcm_data = [], [], [], []

for category in Categories:
    folder = os.path.join(folder_selected, category)
    if not os.path.exists(folder):
        print(f'Direktori tidak ditemukan: {folder}')
        continue
    for img_name in os.listdir(folder):
        if img_name.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder, img_name)
            img = imread(img_path)
            try:
                feat, wave_feat, glcm_props = extract_features(img)
                features.append(feat)
                labels.append(Categories.index(category))
                wavelet_data.append(wave_feat)
                glcm_data.append(glcm_props)
            except Exception as e:
                print(f"Gagal memproses {img_path}: {e}")

# --- KONVERSI FITUR & LABEL KE NUMPY ARRAY ---
X = np.array(features)
y = np.array(labels)

# --- SIMPAN FITUR WAVELET KE CSV ---
wavelet_df = pd.DataFrame(wavelet_data, columns=[f'wavelet_{i+1}' for i in range(len(wavelet_data[0]))])
wavelet_df['label'] = labels  # Menambahkan label kelas
wavelet_df.to_csv('wavelet_features.csv', index=False)
print("Fitur Wavelet telah disimpan ke 'wavelet_features.csv'.")

# --- PCA (untuk visualisasi) ---
pca = PCA(n_components=2)  # Kurangi ke 2 dimensi
X_pca = pca.fit_transform(X)

# --- SIMPAN FITUR PCA KE CSV ---
pca_df = pd.DataFrame(X_pca, columns=['PCA_komponen_1', 'PCA_komponen_2'])
pca_df['label'] = labels  # Menambahkan label kelas
pca_df.to_csv('pca_features.csv', index=False)
print("Fitur PCA telah disimpan ke 'pca_features.csv'.")

# --- LDA (untuk klasifikasi) ---
lda = LDA(n_components=1)  # Karena hanya 2 kelas
X_lda = lda.fit_transform(X, y)

lda_df = pd.DataFrame(X_lda, columns=['LDA_komponen_1'])
lda_df['label'] = labels  # Menambahkan label kelas
lda_df.to_csv('lda_features.csv', index=False)

print(f"Jumlah fitur setelah PCA: {X_pca.shape[1]}")
print(f"Jumlah fitur setelah LDA: {X_lda.shape[1]}")

# --- VISUALISASI PCA & LDA ---
plt.figure(figsize=(12, 6))

# PCA plot
plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', edgecolor='k', s=80)
plt.title("PCA (2 komponen)")
plt.xlabel("Komponen 1")
plt.ylabel("Komponen 2")
plt.colorbar()

# LDA plot
plt.subplot(1, 2, 2)
plt.scatter(X_lda, np.zeros_like(X_lda), c=y, cmap='coolwarm', edgecolor='k', s=80)
plt.title("LDA (1 komponen)")
plt.xlabel("Komponen 1")
plt.colorbar()

plt.tight_layout()
plt.show()

# --- PEMBAGIAN DATA LATIH & UJI ---
X_train, X_test, y_train, y_test = train_test_split(X_lda, y, test_size=0.2, stratify=y, random_state=42)

# --- TRAINING MODEL SVM DENGAN GRIDSEARCHCV ---
svc = svm.SVC(probability=True)
param_grid = {'C': [1, 10], 'gamma': [0.01, 0.001], 'kernel': ['rbf']}
model = GridSearchCV(svc, param_grid)
model.fit(X_train, y_train)

# --- EVALUASI MODEL ---
y_pred = model.predict(X_test)
print(f"Akurasi: {accuracy_score(y_test, y_pred)*100:.2f}%")
print(classification_report(y_test, y_pred, target_names=Categories))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=Categories, yticklabels=Categories)
plt.xlabel('Prediksi')
plt.ylabel('Aktual')
plt.title('Confusion Matrix')
plt.show()

# --- SIMPAN HASIL PREDIKSI DAN PROBABILITAS KE CSV ---
y_pred_prob = model.predict_proba(X_test)
results_df = pd.DataFrame(y_pred_prob, columns=[f'Probabilitas_{i}' for i in range(y_pred_prob.shape[1])])
results_df['Prediksi'] = y_pred
results_df['Label Aktual'] = y_test
results_df.to_csv('prediction_results.csv', index=False)
print("Hasil prediksi dan probabilitas telah disimpan ke 'prediction_results.csv'.")

# --- RINGKASAN DATA ---
print(f"Jumlah gambar GANAS: {sum(np.array(labels) == 0)}")
print(f"Jumlah gambar JINAK: {sum(np.array(labels) == 1)}")

print(f"Jumlah gambar pada data latih: {X_train.shape[0]}")
print(f"Jumlah gambar pada data uji: {X_test.shape[0]}")

# Menampilkan direktori kerja saat ini
print("Direktori kerja saat ini:", os.getcwd())
