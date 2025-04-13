import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
from tkinter import Tk, filedialog

# ========== Buka dialog untuk pilih folder ==========
root = Tk()
root.withdraw()  # Supaya gak muncul window Tkinter kosong
image_folder = filedialog.askdirectory(title="Pilih Folder Gambar")  # Dialog pilih folder

# Kalau user cancel (tidak pilih folder)
if not image_folder:
    print("Folder tidak dipilih.")
    exit()

# ========== Baca hasil clustering ==========
df = pd.read_csv('clustering_results.csv')

# Lihat cluster yang ada
cluster_labels = df["Cluster Label"].unique()

# Tampilkan contoh gambar tiap cluster
for cluster_label in cluster_labels:
    cluster_images = df[df["Cluster Label"] == cluster_label]['Image Name'].values
    sample_images = cluster_images[:5]  # Ambil 5 contoh gambar dari cluster ini
    
    plt.figure(figsize=(15, 5))
    plt.suptitle(f"Cluster {cluster_label}", fontsize=16)
    
    for i, image_name in enumerate(sample_images):
        img_path = os.path.join(image_folder, image_name)
        
        if not os.path.exists(img_path):
            print(f"File gambar {image_name} tidak ditemukan di folder.")
            continue
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        plt.subplot(1, 5, i + 1)
        plt.imshow(img)
        plt.title(image_name)
        plt.axis('off')
    
    plt.show()
