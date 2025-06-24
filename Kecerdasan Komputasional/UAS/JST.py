# === Fungsi Sigmoid (tanpa import math) ===
def exp(x):
    n = 50
    res = 1.0
    for i in range(n, 0, -1):
        res = 1 + x * res / i
    return res

def sigmoid(x):
    return 1 / (1 + exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# === Fungsi untuk membuat angka acak sederhana ===
seed = 1
def rand():
    global seed
    seed = (seed * 9301 + 49297) % 233280
    return seed / 233280.0

# === Inisialisasi bobot ===
def init_weights(inputs, hidden, outputs):
    w1 = [[(rand() * 2 - 1) for _ in range(hidden)] for _ in range(inputs)]
    w2 = [[(rand() * 2 - 1) for _ in range(outputs)] for _ in range(hidden)]
    return w1, w2

# === Forward ===
def forward(X, w1, w2):
    hidden = []
    for i in range(len(w1[0])):
        z = 0
        for j in range(len(X)):
            z += X[j] * w1[j][i]
        hidden.append(sigmoid(z))

    output = []
    for i in range(len(w2[0])):
        z = 0
        for j in range(len(hidden)):
            z += hidden[j] * w2[j][i]
        output.append(sigmoid(z))
    return hidden, output

# === Backpropagation ===
def backward(X, hidden, output, target, w1, w2, lr):
    # Output error
    out_err = [0]*len(output)
    for i in range(len(output)):
        out_err[i] = (target[i] - output[i]) * sigmoid_derivative(output[i])

    # Hidden error
    hid_err = [0]*len(hidden)
    for i in range(len(hidden)):
        err = 0
        for j in range(len(out_err)):
            err += out_err[j] * w2[i][j]
        hid_err[i] = err * sigmoid_derivative(hidden[i])

    # Update w2
    for i in range(len(w2)):
        for j in range(len(w2[0])):
            w2[i][j] += lr * out_err[j] * hidden[i]

    # Update w1
    for i in range(len(w1)):
        for j in range(len(w1[0])):
            w1[i][j] += lr * hid_err[j] * X[i]

# === One-hot encoding ===
def one_hot(label, num):
    v = [0.0] * num
    v[label] = 1.0
    return v

# === Data sederhana 5 fitur dan 3 kelas ===
data = [
    [0.51, 0.35, 0.14, 0.02, 0.51, 0],
    [0.49, 0.30, 0.14, 0.02, 0.49, 0],
    [0.47, 0.32, 0.13, 0.02, 0.47, 0],
    [0.63, 0.33, 0.60, 0.25, 0.63, 2],
    [0.58, 0.27, 0.51, 0.19, 0.58, 2],
    [0.67, 0.30, 0.52, 0.23, 0.67, 2],
    [0.55, 0.23, 0.40, 0.13, 0.55, 1],
    [0.65, 0.28, 0.46, 0.15, 0.65, 1],
    [0.57, 0.28, 0.45, 0.13, 0.57, 1]
]

# === Inisialisasi JST ===
input_size = 5
hidden_size = 6
output_size = 3
lr = 0.1
epochs = 500

w1, w2 = init_weights(input_size, hidden_size, output_size)

# === Training ===
for epoch in range(epochs):
    total_error = 0
    for row in data:
        X = row[:5]
        label = row[5]
        target = one_hot(label, 3)
        hidden, out = forward(X, w1, w2)
        backward(X, hidden, out, target, w1, w2, lr)
        total_error += sum([(target[i]-out[i])**2 for i in range(3)])
    if epoch % 100 == 0:
        print("Epoch:", epoch, "Error:", total_error)

# === Uji hasil ===
print("\n=== Uji Data ===")
for row in data:
    X = row[:5]
    label = row[5]
    _, out = forward(X, w1, w2)
    pred = 0
    maxval = out[0]
    for i in range(1, 3):
        if out[i] > maxval:
            maxval = out[i]
            pred = i
    print("Label:", label, "Prediksi:", pred, "Output:", [round(o,2) for o in out])
