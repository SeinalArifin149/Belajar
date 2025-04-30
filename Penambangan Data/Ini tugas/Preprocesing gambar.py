import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
import tkinter as tk
from tkinter import filedialog

# Fungsi untuk ekstrak fitur GLCM
def extract_glcm_features(image):
    try:
        # Resize gambar supaya seragam
        image = cv2.resize(image, (128, 128))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Buat GLCM
        glcm = graycomatrix(gray, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)

        # Ekstrak properti
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
        correlation = graycoprops(glcm, 'correlation')[0, 0]
        asm = graycoprops(glcm, 'ASM')[0, 0]

        return [contrast, dissimilarity, homogeneity, energy, correlation, asm]
    except Exception as e:
        print(f"❌ Error saat ekstrak fitur: {e}")
        return None

# Fungsi utama
def process_images_to_csv(folder_path, output_csv):
    data = []
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    print(f"📂 {len(image_files)} gambar ditemukan. Mulai ekstraksi...")

    for idx, filename in enumerate(image_files):
        file_path = os.path.join(folder_path, filename)
        image = cv2.imread(file_path)

        if image is not None:
            features = extract_glcm_features(image)
            if features:  # kalau ekstrak berhasil
                label = filename  # Simpan nama file gambar lengkap
                data.append(features + [label])
            else:
                print(f"⚠️ Gagal ekstrak fitur dari {filename}")
        else:
            print(f"⚠️ Gagal membaca gambar: {filename}")

        if idx % 10 == 0:
            print(f"✅ {idx}/{len(image_files)} gambar diproses...")

    if data:
        # Definisi kolom
        columns = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM', 'filename']

        # Buat DataFrame
        df = pd.DataFrame(data, columns=columns)

        # Simpan ke CSV
        df.to_csv(output_csv, index=False)
        print(f"\n🎯 Data berhasil disimpan ke {output_csv}")
    else:
        print("❌ Tidak ada data yang bisa disimpan.")

# GUI untuk pilih folder
def select_folder_and_process():
    root = tk.Tk()
    root.withdraw()  # Sembunyikan window utama

    # Pilih folder
    folder_path = filedialog.askdirectory(title="Pilih Folder Gambar")

    if folder_path:
        # Pilih lokasi dan nama file CSV
        output_csv = filedialog.asksaveasfilename(
            title="Simpan File CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )

        if output_csv:
            process_images_to_csv(folder_path, output_csv)
        else:
            print("❌ Batal menyimpan CSV.")
    else:
        print("❌ Folder tidak dipilih.")

# Jalankan program
if __name__ == "__main__":
    select_folder_and_process()
