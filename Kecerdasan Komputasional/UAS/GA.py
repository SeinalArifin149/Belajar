import random 

# Data dasar
matkul_list = ["A", "B", "C", "D", "E"]
asisten_dict = {
    "A": "Rina",
    "B": "Budi",
    "C": "Citra",
    "D": "Deni",
    "E": "Eka"
}
lab_list = ["Lab1", "Lab2", "Lab3"]
hari_list = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat"]
slot_list = ["Pagi", "Siang", "Sore", "Malam"]

sesi_per_matkul = 4  # Misal 4 sesi/matkul → Total 5x4 = 20 sesi

# ==== Buat Gen (1 sesi praktikum) ====
def buat_gen(matkul):
    asisten = asisten_dict[matkul]
    hari = hari_list[random.randint(0, 4)]
    slot = slot_list[random.randint(0, 3)]
    lab = lab_list[random.randint(0, 2)]
    return [matkul, asisten, hari, slot, lab]

# ==== Buat Kromosom (jadwal lengkap) ====
def buat_kromosom():
    kromosom = []
    for matkul in matkul_list:
        for _ in range(sesi_per_matkul):
            kromosom.append(buat_gen(matkul))
    return kromosom

# ==== Hitung Fitness ====
def hitung_fitness(kromosom):
    konflik = 0
    lab_waktu = {}
    asisten_waktu = {}

    for sesi in kromosom:
        matkul, asisten, hari, slot, lab = sesi
        waktu = (hari, slot)

        # Cek konflik laboratorium
        if (waktu, lab) in lab_waktu:
            konflik += 1
        else:
            lab_waktu[(waktu, lab)] = True

        # Cek konflik asisten
        if (waktu, asisten) in asisten_waktu:
            konflik += 1
        else:
            asisten_waktu[(waktu, asisten)] = True

    return 1 / (1 + konflik)

# ==== Seleksi: ambil 2 terbaik ====
def seleksi(populasi):
    fitnesses = []
    for krom in populasi:
        fitnesses.append((hitung_fitness(krom), krom))
    fitnesses.sort(reverse=True, key=lambda x: x[0])
    return [fitnesses[0][1], fitnesses[1][1]]

# ==== Crossover: satu titik ====
def crossover(p1, p2):
    titik = random.randint(1, len(p1)-2)
    c1 = p1[:titik] + p2[titik:]
    c2 = p2[:titik] + p1[titik:]
    return c1, c2

# ==== Mutasi ====
def mutasi(kromosom, rate=0.1):
    for i in range(len(kromosom)):
        if random.random() < rate:
            kromosom[i][2] = hari_list[random.randint(0, 4)]   # Hari
            kromosom[i][3] = slot_list[random.randint(0, 3)]   # Slot
            kromosom[i][4] = lab_list[random.randint(0, 2)]    # Lab
    return kromosom

# ==== Algoritma Genetika ====
def genetic_algorithm(generasi=100, ukuran_populasi=10):
    populasi = []
    for _ in range(ukuran_populasi):
        populasi.append(buat_kromosom())

    for gen in range(generasi):
        elit = seleksi(populasi)
        anak_baru = []

        while len(anak_baru) < ukuran_populasi:
            p1, p2 = elit[0], elit[1]
            c1, c2 = crossover(p1, p2)
            anak_baru.append(mutasi(c1))
            if len(anak_baru) < ukuran_populasi:
                anak_baru.append(mutasi(c2))

        populasi = anak_baru
        fit = hitung_fitness(elit[0])
        print("Generasi", gen+1, "- Fitness terbaik:", round(fit, 4))
        if fit == 1.0:
            break

    return seleksi(populasi)[0]

# ==== Jalankan ====
hasil = genetic_algorithm()

# ==== Tampilkan Jadwal ====
print("\nJadwal Praktikum Terbaik:")
for sesi in hasil:
    print(f"{sesi[0]} - {sesi[1]} - {sesi[2]} - {sesi[3]} - {sesi[4]}")
