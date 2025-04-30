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

# --- PILIH FOLDER DENGAN TKINTER ---
root = tk.Tk()
root.withdraw()
folder_selected = filedialog.askdirectory(title='Pilih folder gambar untuk prediksi')
print(f"Folder yang dipilih: {folder_selected}")

# --- EKSTRAKSI FITUR FUNGSIONAL ---
def extract_features(img):
    # Resize
    img_resized = resize(img, (150, 150), anti_aliasing=True)

    # Grayscale
    gray = rgb2gray(img_resized)

    # Segmentasi sederhana (threshold global)
    seg_thresh = gray > 0.5
    seg_feat = seg_thresh.flatten()

    # Deteksi Tepi (Canny)
    edge = cv2.Canny((gray * 255).astype(np.uint8), 100, 200)
    edge_feat = edge.flatten()

    # Thresholding (Otsu)
    try:
        thresh = threshold_otsu(gray)
        binarized = gray > thresh
    except:
        binarized = gray > 0.5
    bin_feat = binarized.flatten()

    # Wavelet Haar
    coeffs = pywt.wavedec2(gray, 'haar', level=2)
    cA2, (cH2, cV2, cD2), _ = coeffs
    wave_feat = np.concatenate([cA2.flatten(), cH2.flatten(), cV2.flatten(), cD2.flatten()])

    # GLCM (tekstur)
    gray_uint8 = (gray * 255).astype(np.uint8)
    glcm = graycomatrix(gray_uint8, distances=[1], angles=[0], symmetric=True, normed=True)
    glcm_props = [graycoprops(glcm, prop).ravel()[0] for prop in ('contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM')]

    # Gabungkan semua fitur
    feature_vector = np.concatenate([
        seg_feat[:1000],    # subset agar tidak terlalu besar
        edge_feat[:1000],
        bin_feat[:1000],
        wave_feat[:1000],
        glcm_props
    ])
    return feature_vector

# --- MUAT GAMBAR DAN LABEL ---
Categories = ['GANAS', 'JINAK']
features, labels = [], []

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
                feat = extract_features(img)
                features.append(feat)
                labels.append(Categories.index(category))
            except Exception as e:
                print(f"Gagal memproses {img_path}: {e}")

# --- DATA KE NUMPY ---
X = np.array(features)
y = np.array(labels)

print(f"Jumlah fitur sebelum PCA dan LDA: {X.shape[1]}")

# --- PCA TRANSFORMASI ---
pca = PCA(n_components=2)  # Atur ke 2 untuk visualisasi
X_pca = pca.fit_transform(X)

# --- LDA TRANSFORMASI ---
lda = LDA(n_components=1)  # LDA hanya bisa satu komponen dengan 2 kelas
X_lda = lda.fit_transform(X, y)

print(f"Jumlah fitur setelah PCA: {X_pca.shape[1]}")
print(f"Jumlah fitur setelah LDA: {X_lda.shape[1]}")

# --- VISUALISASI PCA DAN LDA ---
plt.figure(figsize=(12, 6))

# Plot PCA
plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', edgecolor='k', s=80)
plt.title("PCA (2 komponen)")
plt.xlabel("Komponen 1")
plt.ylabel("Komponen 2")
plt.colorbar()

# Plot LDA
plt.subplot(1, 2, 2)
plt.scatter(X_lda, np.zeros_like(X_lda), c=y, cmap='coolwarm', edgecolor='k', s=80)
plt.title("LDA (1 komponen)")
plt.xlabel("Komponen 1")
plt.colorbar()

plt.tight_layout()
plt.show()

# --- TRAINING MODEL ---
X_train, X_test, y_train, y_test = train_test_split(X_lda, y, test_size=0.2, stratify=y, random_state=42)

svc = svm.SVC(probability=True)
param_grid = {'C': [1, 10], 'gamma': [0.01, 0.001], 'kernel': ['rbf']}
model = GridSearchCV(svc, param_grid)
model.fit(X_train, y_train)

# --- EVALUASI ---
y_pred = model.predict(X_test)
print(f"Akurasi: {accuracy_score(y_test, y_pred)*100:.2f}%")
print(classification_report(y_test, y_pred, target_names=Categories))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=Categories, yticklabels=Categories)
plt.xlabel('Prediksi')
plt.ylabel('Aktual')
plt.title('Confusion Matrix')
plt.show()

print(f"Jumlah gambar GANAS: {sum(np.array(labels) == 0)}")
print(f"Jumlah gambar JINAK: {sum(np.array(labels) == 1)}")

print(f"Jumlah gambar pada data latih: {X_train.shape[0]}")
print(f"Jumlah gambar pada data uji: {X_test.shape[0]}")

