import pandas as pd
import tkinter as tk
from tkinter import filedialog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import joblib  # untuk simpan model

# Fungsi untuk memilih file CSV
def select_csv_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Pilih File CSV",
        filetypes=[("CSV files", "*.csv")]
    )
    return file_path

# Fungsi untuk memilih lokasi penyimpanan model
def save_model(model):
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.asksaveasfilename(
        title="Simpan Model",
        defaultextension=".pkl",
        filetypes=[("Pickle files", "*.pkl")]
    )
    if file_path:
        joblib.dump(model, file_path)
        print(f"✅ Model berhasil disimpan di {file_path}")
    else:
        print("❌ Model tidak disimpan.")

# Fungsi utama untuk training SVM
def train_svm_from_csv(csv_file):
    # Baca data
    data = pd.read_csv(csv_file)
    
    # Pisahkan fitur dan label
    X = data.drop('label', axis=1)
    y = data['label']
    
    # Split data untuk training dan testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Buat model SVM
    svm_model = SVC(kernel='linear', C=1)
    svm_model.fit(X_train, y_train)
    
    # Prediksi
    y_pred = svm_model.predict(X_test)
    
    # Evaluasi
    print("\n📄 Classification Report:")
    print(classification_report(y_test, y_pred))
    print("\n🌀 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Simpan model
    save_model(svm_model)

# Jalankan program
if __name__ == "__main__":
    csv_file = select_csv_file()
    if csv_file:
        train_svm_from_csv(csv_file)
    else:
        print("❌ File CSV tidak dipilih.")
