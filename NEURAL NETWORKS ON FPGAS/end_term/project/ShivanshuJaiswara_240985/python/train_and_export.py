"""
train_and_export.py
-------------------
Trains a small 2-layer neural network on the Iris dataset, then exports
the learned weights, biases, and 10 test vectors as hex .mem files that
Verilog can load with $readmemh.

Run:
    pip install numpy scikit-learn tensorflow
    python train_and_export.py

Outputs (all in ../weights/):
    weights.mem   – 32 hidden-layer weights  (8 neurons × 4 inputs)
                    + 24 output-layer weights (3 neurons × 8 inputs)
    biases.mem    – 8 hidden biases + 3 output biases
    test_data.mem – 10 input vectors (4 values each) + expected label
"""

import os
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras

# ── reproducibility ──────────────────────────────────────────────────────────
np.random.seed(42)
tf.random.set_seed(42)

# ── output directory ─────────────────────────────────────────────────────────
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "weights")
os.makedirs(OUT_DIR, exist_ok=True)

# ── load and pre-process dataset ─────────────────────────────────────────────
iris        = load_iris()
X, y        = iris.data.astype(np.float32), iris.target

scaler      = StandardScaler()
X_scaled    = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# ── define and train model ────────────────────────────────────────────────────
model = keras.Sequential([
    keras.layers.Dense(8,  activation="relu",    input_shape=(4,),
                        name="hidden"),
    keras.layers.Dense(3,  activation="softmax", name="output"),
], name="iris_nn")

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.fit(X_train, y_train,
          epochs=200, batch_size=16,
          validation_split=0.1,
          verbose=0)

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {acc*100:.1f}%  (loss={loss:.4f})")

# ── extract weights and biases ────────────────────────────────────────────────
W1, b1 = model.get_layer("hidden").get_weights()   # (4,8), (8,)
W2, b2 = model.get_layer("output").get_weights()   # (8,3), (3,)

# ── quantisation helper (Q8 fixed-point: multiply by 256, clamp to 16-bit) ───
def quantise(fp_val: float) -> int:
    q = int(round(fp_val * 256))
    return max(-32768, min(32767, q))

def to_hex16(val: int) -> str:
    """Return a 4-digit two's-complement hex string for a 16-bit integer."""
    return format(val & 0xFFFF, "04X")

# ── write weights.mem ─────────────────────────────────────────────────────────
# Layout:
#   lines 0-31  : hidden layer  – neuron0_w0..w3, neuron1_w0..w3, …
#   lines 32-55 : output layer  – neuron0_w0..w7, neuron1_w0..w7, neuron2_w0..w7
weights_path = os.path.join(OUT_DIR, "weights.mem")
with open(weights_path, "w") as fh:
    # hidden layer: W1 shape (4 inputs, 8 neurons) → iterate by neuron
    for neuron_idx in range(8):
        for inp_idx in range(4):
            fh.write(to_hex16(quantise(W1[inp_idx, neuron_idx])) + "\n")
    # output layer: W2 shape (8 inputs, 3 neurons)
    for neuron_idx in range(3):
        for inp_idx in range(8):
            fh.write(to_hex16(quantise(W2[inp_idx, neuron_idx])) + "\n")
print(f"Wrote {weights_path}")

# ── write biases.mem ──────────────────────────────────────────────────────────
biases_path = os.path.join(OUT_DIR, "biases.mem")
with open(biases_path, "w") as fh:
    for v in b1:                        # 8 hidden biases
        fh.write(to_hex16(quantise(v)) + "\n")
    for v in b2:                        # 3 output biases
        fh.write(to_hex16(quantise(v)) + "\n")
print(f"Wrote {biases_path}")

# ── write test_data.mem ───────────────────────────────────────────────────────
# Format (one line each):
#   4 input values  →  expected label
# We interleave: inp0, inp1, inp2, inp3, label  (5 lines per test case)
test_path = os.path.join(OUT_DIR, "test_data.mem")
with open(test_path, "w") as fh:
    for i in range(10):
        for feat in X_test[i]:
            fh.write(to_hex16(quantise(float(feat))) + "\n")
        fh.write(format(int(y_test[i]), "04X") + "\n")   # expected label
print(f"Wrote {test_path}")

# ── quick sanity check: run a forward pass in quantised integer arithmetic ───
print("\n--- Quantised forward-pass sanity check (first 5 test samples) ---")
for i in range(5):
    x_q = np.array([quantise(float(v)) for v in X_test[i]], dtype=np.int32)

    # hidden layer
    h = np.zeros(8, dtype=np.int32)
    for n in range(8):
        acc = 0
        for j in range(4):
            acc += x_q[j] * quantise(float(W1[j, n]))
        acc = (acc >> 8) + quantise(float(b1[n]))   # Q8 rescale
        h[n] = max(0, acc)                           # ReLU

    # output layer
    o = np.zeros(3, dtype=np.int32)
    for n in range(3):
        acc = 0
        for j in range(8):
            acc += h[j] * quantise(float(W2[j, n]))
        acc = (acc >> 8) + quantise(float(b2[n]))
        o[n] = acc

    predicted = int(np.argmax(o))
    expected  = int(y_test[i])
    match     = "✓" if predicted == expected else "✗"
    print(f"  Sample {i}: expected={expected}  predicted={predicted}  {match}")
