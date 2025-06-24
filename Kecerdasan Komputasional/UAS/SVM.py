import matplotlib.pyplot as plt
import numpy as np

kelas_A = [(2, 3), (3, 3), (4, 5), (5, 4), (6, 5)]
kelas_B = [(2, 0), (3, 1), (4, 1), (5, 2), (6, 3)]

# Titik support vector (manual)
x_pos = (4, 5)  
x_neg = (4, 1)  

# Hitung vektor bobot w = x_pos - x_neg
w1 = x_pos[0] - x_neg[0]  # 0
w2 = x_pos[1] - x_neg[1]  # 4

# Titik tengah
mid_x = (x_pos[0] + x_neg[0]) / 2  # 4
mid_y = (x_pos[1] + x_neg[1]) / 2  # 3

# Hitung bias
b = - (w1 * mid_x + w2 * mid_y)  # -12

# Persamaan hyperplane: 0*x + 4*y -12 = 0 → y = 3

# Visualisasi
plt.figure(figsize=(8, 6))

# Plot titik kelas A dan B
kelas_A_x, kelas_A_y = zip(*kelas_A)
kelas_B_x, kelas_B_y = zip(*kelas_B)
plt.scatter(kelas_A_x, kelas_A_y, color='blue', label='Kelas +1')
plt.scatter(kelas_B_x, kelas_B_y, color='red', label='Kelas -1')

# Plot support vectors
plt.scatter(*x_pos, color='blue', edgecolor='black', s=100, marker='o', label='Support Vector +1')
plt.scatter(*x_neg, color='red', edgecolor='black', s=100, marker='o', label='Support Vector -1')

# Plot hyperplane y = 3
plt.axhline(y=3, color='green', linestyle='--', label='Hyperplane (y=3)')

plt.title('Visualisasi Data dan Hyperplane SVM (Manual)')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.xlim(1, 7)
plt.ylim(-1, 6)

plt.show()
