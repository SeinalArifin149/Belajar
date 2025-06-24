import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import filedialog
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.feature import graycomatrix, graycoprops
from skimage.transform import resize
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pywt

# --- PILIH FOLDER GAMBAR ---
root = tk.Tk()
root.withdraw()
folder_selected = filedialog.askdirectory(title='Pilih folder gambar untuk prediksi')
print(f"Folder yang dipilih: {folder_selected}")

# --- EKSTRAKSI FITUR CITRA ---
def extract_features(img):
    img_resized = resize(img, (150, 150), anti_aliasing=True)
    gray = rgb2gray(img_resized)
    coeffs = pywt.wavedec2(gray, 'haar', level=2)
    cA2, (cH2, cV2, cD2), _ = coeffs
    wave_feat = np.concatenate([cA2.flatten(), cH2.flatten(), cV2.flatten(), cD2.flatten()])
    gray_uint8 = (gray * 255).astype(np.uint8)
    glcm = graycomatrix(gray_uint8, distances=[1], angles=[0], symmetric=True, normed=True)
    glcm_props = [graycoprops(glcm, prop).ravel()[0] for prop in (
        'contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM'
    )]
    feature_vector = np.concatenate([wave_feat[:2000], glcm_props])
    return feature_vector

# --- LOAD DATASET DARI FOLDER ---
Categories = ['GANAS', 'JINAK']
features, labels = [], []

for category in Categories:
    folder = os.path.join(folder_selected, category)
    if not os.path.exists(folder): continue
    for img_name in os.listdir(folder):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder, img_name)
            try:
                img = imread(img_path)
                feat = extract_features(img)
                features.append(feat)
                labels.append(Categories.index(category))
            except:
                continue

X = np.array(features)
y = np.array(labels)

# --- NORMALISASI ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- PCA & LDA ---
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

lda = LDA(n_components=1)
X_lda = lda.fit_transform(X_scaled, y)

# --- VISUALISASI REDUKSI DIMENSI ---
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', edgecolor='k', s=60)
plt.title("PCA (2 Komponen)")
plt.xlabel("Komponen 1")
plt.ylabel("Komponen 2")

plt.subplot(1, 2, 2)
plt.scatter(X_lda, np.zeros_like(X_lda), c=y, cmap='coolwarm', edgecolor='k', s=60)
plt.title("LDA (1 Komponen)")
plt.xlabel("Komponen 1")
plt.tight_layout()
plt.show()

# --- TRAINING & CONFUSION MATRIX ---
def train_svm(X_data, y_data, model_name):
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, stratify=y_data, random_state=42)
    model = GridSearchCV(svm.SVC(probability=True), {'C': [1, 10], 'gamma': [0.01, 0.001], 'kernel': ['rbf']})
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred) * 100
    print(f"\n=== {model_name} ===")
    print(f"Akurasi: {acc:.2f}%")
    print(classification_report(y_test, y_pred, target_names=Categories))
    return acc, y_test, y_pred

# --- FUNGSI CONFUSION MATRIX LENGKAP ---
def plot_confusion_matrix(y_true, y_pred, title_prefix=""):
    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred) * 100
    TP = cm[0][0]
    TN = cm[1][1]
    FP = cm[1][0]
    FN = cm[0][1]
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=Categories, yticklabels=Categories)
    plt.xlabel('Prediksi')
    plt.ylabel('Aktual')
    plt.title(f'{title_prefix}\nAkurasi: {accuracy:.2f}%')

    ax = plt.gca()
    metrics_text = f'Presisi (GANAS): {precision:.2%}   Recall (GANAS): {recall:.2%}'
    plt.text(0.5, -0.2, metrics_text, ha='center', va='center', transform=ax.transAxes, fontsize=10)

    plt.tight_layout()
    plt.show()

# --- TRAINING MODEL DAN VISUALISASI CONFUSION MATRIX ---
akurasi_pca, y_test_pca, y_pred_pca = train_svm(X_pca, y, "PCA + SVM")
akurasi_lda, y_test_lda, y_pred_lda = train_svm(X_lda, y, "LDA + SVM")

plot_confusion_matrix(y_test_pca, y_pred_pca, "Confusion Matrix PCA + SVM")
plot_confusion_matrix(y_test_lda, y_pred_lda, "Confusion Matrix LDA + SVM")

# --- GRAFIK BANDINGKAN AKURASI ---
plt.figure(figsize=(6, 4))
plt.bar(['PCA + SVM', 'LDA + SVM'], [akurasi_pca, akurasi_lda], color=['skyblue', 'salmon'])
plt.title('Perbandingan Akurasi Model')
plt.ylabel('Akurasi (%)')
plt.ylim(0, 100)
for i, acc in enumerate([akurasi_pca, akurasi_lda]):
    plt.text(i, acc + 1, f"{acc:.2f}%", ha='center')
plt.tight_layout()
plt.show()
