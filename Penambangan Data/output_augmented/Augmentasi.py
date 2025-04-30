import os
from tkinter import Tk, filedialog
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

# --- SETUP TKINTER ---
root = Tk()
root.withdraw()  # Hide main window

# Pilih banyak folder
print("Pilih folder input (bisa pilih satu folder dulu, lalu ulangi jika perlu)")
input_folders = []

while True:
    lanjut = input("Tambah folder baru? (y/n): ")  # Tanya dulu sebelum pilih folder
    if lanjut.lower() != 'y':
        break

    folder = filedialog.askdirectory(title="Pilih Folder Gambar")
    if folder:
        input_folders.append(folder)
        print(f"✅ Folder ditambahkan: {folder}")
    else:
        print("⚠️ Tidak ada folder dipilih.")

# Pastikan ada folder input
if not input_folders:
    print("❌ Tidak ada folder yang dipilih. Program dihentikan.")
    exit()

# Folder output untuk hasil augmentasi
output_folder = filedialog.askdirectory(title="Pilih Folder Output untuk Augmentasi")
if not output_folder:
    output_folder = 'output_augmented'
os.makedirs(output_folder, exist_ok=True)

# Setup augmentasi
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# --- PROSES ---
for folder in input_folders:
    if not os.path.exists(folder):
        print(f"❌ Folder {folder} tidak ditemukan!")
        continue

    for filename in os.listdir(folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder, filename)
            try:
                img = load_img(img_path)
                x = img_to_array(img)
                x = x.reshape((1,) + x.shape)

                base_name = os.path.splitext(filename)[0]

                # Buat beberapa augmentasi per gambar
                i = 0
                for batch in datagen.flow(x, batch_size=1,
                                          save_to_dir=output_folder,
                                          save_prefix=base_name,
                                          save_format='jpg'):
                    i += 1
                    if i >= 5:  # 5 augmentasi per gambar
                        break
                print(f"✅ Augmentasi selesai untuk {filename}")
            except Exception as e:
                print(f"⚠️ Gagal proses {filename}: {e}")

print("🎉 Semua augmentasi selesai!")
y