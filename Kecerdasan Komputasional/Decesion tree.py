data = [
    ['bensin', 4, 'depan', 'sedan'],
    ['bensin', 2, 'depan', 'sedan'],
    ['bensin', 8, 'belakang', 'minibus'],
    ['diesel', 6, 'depan', 'minibus'],
    ['bensin', 5, 'belakang', 'minibus'],
    ['diesel', 8, 'belakang', 'minibus'],
    ['diesel', 8, 'belakang', 'minibus'],
    ['bensin', 4, 'depan', 'sedan'],
    ['diesel', 7, 'belakang', 'minibus'],
    ['bensin', 4, 'depan', 'sedan'],
    ['bensin', 2, 'depan', 'sedan'],
    ['diesel', 6, 'belakang', 'minibus'],
    ['diesel', 8, 'depan', 'minibus'],
    ['bensin', 4, 'depan', 'sedan'],
    ['diesel', 2, 'depan', 'sedan'],
    ['bensin', 4, 'depan', 'sedan'],
    ['bensin', 8, 'belakang', 'minibus'],
    ['bensin', 4, 'depan', 'sedan'],
    ['diesel', 8, 'belakang', 'minibus'],
    ['diesel', 8, 'belakang', 'minibus'],
    ['bensin', 4, 'depan', 'sedan'],
    ['bensin', 4, 'depan', 'minibus'],
    ['bensin', 4, 'depan', 'sedan'],
    ['bensin', 4, 'depan', 'sedan'],
    ['bensin', 2, 'depan', 'sedan'],
    ['bensin', 8, 'belakang', 'minibus'],
    ['diesel', 6, 'depan', 'minibus'],
    ['bensin', 5, 'belakang', 'minibus'],
    ['diesel', 8, 'belakang', 'minibus'],
    ['diesel', 4, 'belakang', 'minibus'],
    ['bensin', 4, 'depan', 'sedan'],
    ['diesel', 5, 'belakang', 'minibus'],
    ['bensin', 4, 'depan', 'sedan']
]
# Node class
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

# Decision Tree class
class DecisionTree:
    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        if self._semua_sama(y):
            print(f"{'|  ' * depth}Semua label sama: {y[0]}")
            return Node(value=y[0])

        if len(X) == 0 or len(X[0]) == 0:
            return Node(value=self._label_terbanyak(y))

        fitur_terbaik, threshold_terbaik = self._cari_split_terbaik(X, y, depth)

        if fitur_terbaik is None:
            return Node(value=self._label_terbanyak(y))

        kiri = [i for i in range(len(X)) if X[i][fitur_terbaik] <= threshold_terbaik]
        kanan = [i for i in range(len(X)) if X[i][fitur_terbaik] > threshold_terbaik]

        print(f"{'|  ' * depth}Split: fitur {fitur_terbaik} <= {threshold_terbaik}")

        kiri_node = self._build_tree([X[i] for i in kiri], [y[i] for i in kiri], depth + 1)
        kanan_node = self._build_tree([X[i] for i in kanan], [y[i] for i in kanan], depth + 1)

        return Node(feature=fitur_terbaik, threshold=threshold_terbaik, left=kiri_node, right=kanan_node)

    def _cari_split_terbaik(self, X, y, depth):
        fitur_terbaik = None
        threshold_terbaik = None
        gain_terbaik = -1
        entropy_parent = self._entropy(y)

        print(f"{'|  ' * depth}Entropy parent = {entropy_parent:.4f}")

        for f in range(len(X[0])):
            nilai_unik = []
            for row in X:
                if row[f] not in nilai_unik:
                    nilai_unik.append(row[f])

            for t in nilai_unik:
                kiri_y = [y[i] for i in range(len(X)) if X[i][f] <= t]
                kanan_y = [y[i] for i in range(len(X)) if X[i][f] > t]

                if len(kiri_y) == 0 or len(kanan_y) == 0:
                    continue

                e_kiri = self._entropy(kiri_y)
                e_kanan = self._entropy(kanan_y)
                total = len(y)
                e_child = (len(kiri_y) / total) * e_kiri + (len(kanan_y) / total) * e_kanan
                gain = entropy_parent - e_child

                print(f"{'|  ' * depth}  [F{f} <= {t}] Gain = {gain:.4f}")

                if gain > gain_terbaik:
                    gain_terbaik = gain
                    fitur_terbaik = f
                    threshold_terbaik = t

        if fitur_terbaik is not None:
            print(f"{'|  ' * depth}>> Pilih fitur {fitur_terbaik} dengan threshold {threshold_terbaik}, Gain = {gain_terbaik:.4f}")

        return fitur_terbaik, threshold_terbaik

    def _entropy(self, y):
        jumlah = {}
        for label in y:
            if label not in jumlah:
                jumlah[label] = 0
            jumlah[label] += 1

        total = len(y)
        hasil = 0
        for label in jumlah:
            p = jumlah[label] / total
            hasil -= p * self._log2(p)
        return hasil

    def _log2(self, x):
        hasil = 0
        while x < 1:
            x *= 2
            hasil -= 1
        return -hasil * (x - 1) if x != 1 else 0

    def _semua_sama(self, y):
        for i in range(1, len(y)):
            if y[i] != y[0]:
                return False
        return True

    def _label_terbanyak(self, y):
        frek = {}
        for label in y:
            if label not in frek:
                frek[label] = 0
            frek[label] += 1
        return max(frek, key=frek.get)

    def predict(self, X):
        return [self._prediksi(x, self.root) for x in X]

    def _prediksi(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._prediksi(x, node.left)
        else:
            return self._prediksi(x, node.right)

    def print_tree(self, node=None, depth=0):
        if node is None:
            node = self.root
        indent = "    " * depth
        if node.value is not None:
            print(f"{indent}└── {node.value}")
            return
        print(f"{indent}[F{node.feature} <= {node.threshold}]")
        print(f"{indent}├──", end="")
        self.print_tree(node.left, depth + 1)
        print(f"{indent}└──", end="")
        self.print_tree(node.right, depth + 1)

# Data training


# Encode data kategorikal jadi angka
def encode_data(data):
    encoder = [{}, {}, {}]
    X, y = [], []
    for row in data:
        fitur = []
        for i in range(3):
            val = row[i]
            if val not in encoder[i]:
                encoder[i][val] = len(encoder[i])
            fitur.append(encoder[i][val])
        X.append(fitur)
        y.append(row[-1])
    return X, y

# Jalankan program
X, y = encode_data(data)

tree = DecisionTree()
tree.fit(X, y)

pred = tree.predict(X)
for i in range(len(pred)):
    print(f"Sample {i+1}: Prediksi = {pred[i]}, Asli = {y[i]}")

# Cetak struktur pohonnya
print("\nVisualisasi Pohon Keputusan:")
tree.print_tree()
