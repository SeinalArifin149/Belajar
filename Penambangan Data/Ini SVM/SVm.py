import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import filedialog

from skimage.io import imread
from skimage.transform import resize

from sklearn import svm
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- PILIH FOLDER DENGAN TKINTER ---
root = tk.Tk()
root.withdraw()  # hide main window
folder_selected = filedialog.askdirectory(title='Pilih folder gambar untuk prediksi')
print(f"Folder yang dipilih: {folder_selected}")

# --- DATA TRAINING ---
Categories = ['GANAS', 'JINAK']
flat_data_arr = []
target_arr = []
datadir = folder_selected


for i in Categories:
    print(f'Memuat... kategori: {i}')
    path = os.path.join(datadir, i)
    
    if not os.path.exists(path):
        print(f'Direktori tidak ditemukan: {path}')
        continue
    
    for img in os.listdir(path):
        if img.endswith(('.png', '.jpg', '.jpeg')):
            img_array = imread(os.path.join(path, img))
            img_resized = resize(img_array, (150, 150, 3))
            flat_data_arr.append(img_resized.flatten())
            target_arr.append(Categories.index(i))

    print(f'Kategori {i} berhasil dimuat')

flat_data = np.array(flat_data_arr)
target = np.array(target_arr)

df = pd.DataFrame(flat_data)
df['Target'] = target

x = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Membagi data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=77, stratify=y)

# Training model
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.0001, 0.001, 0.1, 1], 'kernel': ['rbf', 'poly']}
svc = svm.SVC(probability=True)
model = GridSearchCV(svc, param_grid)
model.fit(x_train, y_train)

# Evaluasi model
y_pred = model.predict(x_test)
akurasi = accuracy_score(y_pred, y_test)
print(f"Model ini memiliki akurasi {akurasi*100:.2f}%")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=Categories, yticklabels=Categories)
plt.xlabel('Prediksi')
plt.ylabel('Aktual')
plt.title('Matriks Confusion')
plt.show()

print(classification_report(y_test, y_pred, target_names=Categories))

# --- MEMBACA SEMUA GAMBAR DARI FOLDER YANG DIPILIH ---
image_paths = []
for img_file in os.listdir(folder_selected):
    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_paths.append(os.path.join(folder_selected, img_file))

# --- PREDIKSI DAN TAMPILKAN GAMBAR ---
fig, axs = plt.subplots(2, 5, figsize=(20, 12))
fig.subplots_adjust(hspace=0.5)

for i, path in enumerate(image_paths[:10]):  # maksimal 10 gambar
    img = imread(path)
    axs[i//5, i%5].imshow(img)
    axs[i//5, i%5].axis('off')

    img_resize = resize(img, (150, 150, 3))
    l = [img_resize.flatten()]
    probability = model.predict_proba(l)

    axs[i//5, i%5].text(0.5, -0.2, f'GANAS: {probability[0][0]*100:.2f}%', size=12, color='black',
                ha="center", transform=axs[i//5, i%5].transAxes)

    axs[i//5, i%5].text(0.5, -0.3, f'JINAK: {probability[0][1]*100:.2f}%', size=12, color='black',
                ha="center", transform=axs[i//5, i%5].transAxes)

    axs[i//5, i%5].set_title(f'Predicted: {Categories[model.predict(l)[0]]}')

plt.tight_layout()
plt.show()
