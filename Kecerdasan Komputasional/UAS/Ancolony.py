# === Data jarak antar kota ===
distances = [
    [0, 29, 20, 21, 16, 31, 100, 12, 4, 31],
    [29, 0, 15, 29, 28, 40, 72, 21, 29, 41],
    [20, 15, 0, 15, 14, 25, 81, 9, 23, 27],
    [21, 29, 15, 0, 4, 12, 92, 12, 25, 13],
    [16, 28, 14, 4, 0, 16, 94, 9, 20, 16],
    [31, 40, 25, 12, 16, 0, 95, 24, 36, 3],
    [100, 72, 81, 92, 94, 95, 0, 90, 101, 99],
    [12, 21, 9, 12, 9, 24, 90, 0, 15, 25],
    [4, 29, 23, 25, 20, 36, 101, 15, 0, 35],
    [31, 41, 27, 13, 16, 3, 99, 25, 35, 0],
]

# === Parameter ACO ===
num_cities = 10
num_ants = 10
num_iterations = 5
alpha = 1
beta = 5
rho = 0.5
Q = 100

# === Inisialisasi pheromone ===
pheromone = [[1 for _ in range(num_cities)] for _ in range(num_cities)]

# === Fungsi bantuan ===
def total_distance(path):
    d = 0
    for i in range(len(path)-1):
        d += distances[path[i]][path[i+1]]
    d += distances[path[-1]][path[0]]
    return d

def probability(current, unvisited, tabu):
    total = 0
    probs = []
    for city in unvisited:
        tau = pheromone[current][city] ** alpha
        eta = (1 / distances[current][city]) ** beta
        p = tau * eta
        probs.append((city, p))
        total += p
    # normalisasi
    normalized = []
    for city, p in probs:
        normalized.append((city, p / total if total > 0 else 0))
    return normalized

# Pilih kota berikutnya secara deterministik: probabilitas terbesar
def select_next(prob_list):
    best_city = -1
    best_prob = -1
    for city, prob in prob_list:
        if prob > best_prob:
            best_prob = prob
            best_city = city
    return best_city

# Evaporasi pheromone
def evaporate():
    for i in range(num_cities):
        for j in range(num_cities):
            pheromone[i][j] *= (1 - rho)

# Update pheromone berdasarkan jalur
def update_pheromone(path, distance):
    for i in range(len(path)-1):
        a = path[i]
        b = path[i+1]
        pheromone[a][b] += Q / distance
        pheromone[b][a] += Q / distance
    # Tambah pheromone kembali ke kota awal
    a = path[-1]
    b = path[0]
    pheromone[a][b] += Q / distance
    pheromone[b][a] += Q / distance

# === Proses ACO ===
best_path = []
best_length = 9999999

for iteration in range(num_iterations):
    all_paths = []
    all_lengths = []

    for ant in range(num_ants):
        start = ant % num_cities
        path = [start]
        unvisited = list(range(num_cities))
        unvisited.remove(start)

        current = start
        while unvisited:
            probs = probability(current, unvisited, path)
            next_city = select_next(probs)
            path.append(next_city)
            unvisited.remove(next_city)
            current = next_city

        length = total_distance(path)
        all_paths.append(path)
        all_lengths.append(length)

        if length < best_length:
            best_length = length
            best_path = path[:]

    evaporate()
    for i in range(num_ants):
        update_pheromone(all_paths[i], all_lengths[i])

    print("Iterasi", iteration+1, "→ Rute terbaik sementara:", best_path, "| Jarak:", best_length)

# === Hasil akhir ===
print("\nRUTE TERBAIK:")
for kota in best_path:
    print("Kota", kota, end=" → ")
print("Kota", best_path[0])
print("Total jarak:", best_length)
