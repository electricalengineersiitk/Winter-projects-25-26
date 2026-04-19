"""
train_and_export.py
Trains a 2-layer MLP on the Iris dataset and exports all weights,
biases, and test vectors as .mem files for the FPGA neural network.

Network architecture:
  Input:  4 features (scaled)
  Hidden: 8 neurons, ReLU
  Output: 3 neurons, argmax -> class 0/1/2

Quantisation: Q8 fixed-point  =>  int(round(float * 256)), clipped to int16

Output files (hex, one value per line):
  weights/weights.mem      -- W1 (8x4 = 32 values, row-major)
  weights/w1_bias.mem      -- b1 (8 values)
  weights/w2_weights.mem   -- W2 (3x8 = 24 values, row-major)
  weights/w2_bias.mem      -- b2 (3 values)
  weights/test_data.mem    -- 10 test samples, each: 4 inputs + 1 expected label (5 lines)

Usage:
  python train_and_export.py
"""

import os
import numpy as np
from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# ── reproducibility ───────────────────────────────────────────────────────────
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ── load & scale data ─────────────────────────────────────────────────────────
iris   = load_iris()
X, y   = iris.data, iris.target
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── train ─────────────────────────────────────────────────────────────────────
model = MLPClassifier(
    hidden_layer_sizes=(8,),
    activation='relu',
    max_iter=2000,
    random_state=RANDOM_STATE
)
model.fit(X_scaled, y)
print(f"Training accuracy: {model.score(X_scaled, y)*100:.1f}%")

# ── quantise ──────────────────────────────────────────────────────────────────
def quantise(x):
    q = int(round(float(x) * 256))
    return max(-32768, min(32767, q))

vquantise = np.vectorize(quantise)

W1_q = vquantise(model.coefs_[0]).T        # (8, 4)  — transposed for FPGA
b1_q = vquantise(model.intercepts_[0])     # (8,)
W2_q = vquantise(model.coefs_[1]).T        # (3, 8)
b2_q = vquantise(model.intercepts_[1])     # (3,)

# ── helper ────────────────────────────────────────────────────────────────────
def to_hex(v):
    return f"{int(v) & 0xFFFF:04X}"

def save_mem(path, values):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for v in np.array(values).flatten():
            f.write(to_hex(v) + "\n")
    print(f"  Wrote {path}  ({int(np.array(values).size)} values)")

# ── save weight/bias files ────────────────────────────────────────────────────
print("\nExporting weight files...")
save_mem("weights/weights.mem",     W1_q)
save_mem("weights/w1_bias.mem",     b1_q)
save_mem("weights/w2_weights.mem",  W2_q)
save_mem("weights/w2_bias.mem",     b2_q)

# ── forward pass (quantised) for verification ─────────────────────────────────
def forward(x_raw):
    x   = scaler.transform([x_raw])[0]
    xq  = np.array([quantise(v) for v in x], dtype=float)
    h   = np.maximum(0, (W1_q @ xq + b1_q * 256) / 256)
    o   = (W2_q @ h + b2_q * 256) / 256
    return int(np.argmax(o))

# ── test data ─────────────────────────────────────────────────────────────────
# 10 samples: 2 from each class + a few extras
indices  = [0, 1, 50, 51, 100, 101, 10, 60, 110, 25]
X_test   = X[indices]
y_test   = y[indices]

print("\nExporting test_data.mem...")
os.makedirs("weights", exist_ok=True)
with open("weights/test_data.mem", "w") as f:
    for i, (x_raw, label) in enumerate(zip(X_test, y_test)):
        x_sc  = scaler.transform([x_raw])[0]
        pred  = forward(x_raw)
        for v in x_sc:
            f.write(to_hex(quantise(v)) + "\n")
        f.write(to_hex(pred) + "\n")
        status = "OK" if pred == label else "MISMATCH"
        print(f"  Sample {i+1}: true={label}  predicted={pred}  {status}")

print(f"\nDone. All files written to weights/")
