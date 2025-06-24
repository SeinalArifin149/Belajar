# Fungsi objektif yang ingin diminimalkan
def fungsi(x, y):
    return x**2 + y**2

# Parameter PSO
jumlah_partikel = 5
jumlah_iterasi = 5
batas_bawah = -10
batas_atas = 10
w = 0.5      # inertia weight
c1 = 1.5     # cognitive
c2 = 1.5     # social

# Fungsi nilai acak tetap (simulasi pengganti random)
def fixed_random(i):
    return 0.5 + (i % 3) * 0.1  # hasil: 0.5, 0.6, 0.7, 0.5, ...

# Inisialisasi partikel (tanpa random)
partikel = []
for i in range(jumlah_partikel):
    posisi = [batas_bawah + i * 4, batas_atas - i * 4]
    kecepatan = [1, -1]
    nilai = fungsi(posisi[0], posisi[1])
    partikel.append({
        'posisi': posisi,
        'kecepatan': kecepatan,
        'nilai': nilai,
        'pbest': posisi[:],
        'pbest_nilai': nilai
    })

# Inisialisasi global best
gbest = min(partikel, key=lambda x: x['nilai'])
gbest_posisi = gbest['posisi'][:]
gbest_nilai = gbest['nilai']

# Iterasi PSO
for iterasi in range(jumlah_iterasi):
    print(f"\nIterasi ke-{iterasi+1}")
    for i in range(jumlah_partikel):
        p = partikel[i]
        for d in range(2):  # dimensi x dan y
            r1 = fixed_random(i)
            r2 = fixed_random(i+1)
            # Update kecepatan
            p['kecepatan'][d] = (
                w * p['kecepatan'][d]
                + c1 * r1 * (p['pbest'][d] - p['posisi'][d])
                + c2 * r2 * (gbest_posisi[d] - p['posisi'][d])
            )
            # Update posisi
            p['posisi'][d] += p['kecepatan'][d]
            # Batas bawah-atas
            if p['posisi'][d] < batas_bawah:
                p['posisi'][d] = batas_bawah
            elif p['posisi'][d] > batas_atas:
                p['posisi'][d] = batas_atas

        # Update nilai dan pbest
        p['nilai'] = fungsi(p['posisi'][0], p['posisi'][1])
        if p['nilai'] < p['pbest_nilai']:
            p['pbest'] = p['posisi'][:]
            p['pbest_nilai'] = p['nilai']

        if p['nilai'] < gbest_nilai:
            gbest_posisi = p['posisi'][:]
            gbest_nilai = p['nilai']

        print(f"Partikel {i+1} Posisi: {p['posisi']}, Nilai: {p['nilai']:.4f}")

# Hasil akhir
print("\n=== HASIL AKHIR ===")
print(f"Koordinat Minimum: x = {gbest_posisi[0]:.4f}, y = {gbest_posisi[1]:.4f}")
print(f"Nilai Minimum: f(x,y) = {gbest_nilai:.4f}")
