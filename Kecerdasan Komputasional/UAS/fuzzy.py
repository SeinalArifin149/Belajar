# === Fungsi keanggotaan fuzzy (linier segitiga/manual) ===
def rendah(x, a, b):
    if x <= a:
        return 1.0
    elif x >= b:
        return 0.0
    else:
        return (b - x) / (b - a)

def sedang(x, a, b, c):
    if x <= a or x >= c:
        return 0.0
    elif x == b:
        return 1.0
    elif x < b:
        return (x - a) / (b - a)
    else:
        return (c - x) / (c - b)

def tinggi(x, a, b):
    if x <= a:
        return 0.0
    elif x >= b:
        return 1.0
    else:
        return (x - a) / (b - a)

# === Fungsi Sugeno output ===
def sugeno_output(w1, z1, w2, z2, w3, z3):
    return (w1*z1 + w2*z2 + w3*z3) / (w1 + w2 + w3)

# === Aturan fuzzy manual ===
# IF F1 rendah AND F3 rendah THEN kelas 0 → output = 0
# IF F1 sedang AND F3 sedang THEN kelas 1 → output = 1
# IF F1 tinggi AND F3 tinggi THEN kelas 2 → output = 2

# === Proses klasifikasi ===
def klasifikasi(f1, f3):
    # Nilai keanggotaan
    u_f1_low = rendah(f1, 0.4, 0.55)
    u_f1_med = sedang(f1, 0.45, 0.55, 0.65)
    u_f1_high = tinggi(f1, 0.55, 0.7)

    u_f3_low = rendah(f3, 0.1, 0.35)
    u_f3_med = sedang(f3, 0.2, 0.4, 0.55)
    u_f3_high = tinggi(f3, 0.4, 0.65)

    # Derajat kebenaran rule (AND = MIN)
    w1 = min(u_f1_low, u_f3_low)
    w2 = min(u_f1_med, u_f3_med)
    w3 = min(u_f1_high, u_f3_high)

    # Output Sugeno masing-masing rule
    z1 = 0  # kelas 0
    z2 = 1  # kelas 1
    z3 = 2  # kelas 2

    if (w1 + w2 + w3) == 0:
        return -1  # tidak bisa diproses

    # Defuzzifikasi Sugeno (bobot * output)
    z = sugeno_output(w1, z1, w2, z2, w3, z3)
    return round(z)

# === Uji data ===
print("F1    F3    Label  Prediksi")
for row in data:
    f1 = row[0]
    f3 = row[2]
    label = row[5]
    pred = klasifikasi(f1, f3)
    print(f"{f1:.2f}  {f3:.2f}    {label}      {pred}")
