"""
Iris Classifier - Training and Export
For the FPGA implementation, I'm exporting weights as Q8 fixed-point (16-bit signed).
"""

import numpy as np
import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras

# Setting seeds for random number generators so results are consistent
np.random.seed(404)
tf.random.set_seed(404)

# Load the Iris dataset
iris = load_iris()
X = iris.data.astype(np.float32)   # shape: (150, 4)
y = iris.target                     # 0, 1, or 2

# Normalise inputs (mean=0, std=1) — same scaler used for test export
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Build and train the model (4 -> 8 -> 3)
model = keras.Sequential([
    keras.layers.Dense(8, activation='relu', input_shape=(4,),
                        kernel_initializer='glorot_uniform', name='hidden'),
    keras.layers.Dense(3, activation='softmax', name='output')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train,
          epochs=400, batch_size=32,
          validation_split=0.1,
          verbose=0)

# Quick evaluation to see if accuracy is good enough
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Accuracy on test set: {acc*100:.1f}%")
assert acc >= 0.89, "Accuracy is too low. Try a different seed."

# Grab weights and biases from the layers
hidden_layer = model.get_layer('hidden')
output_layer = model.get_layer('output')

W_hidden = hidden_layer.get_weights()[0]   # shape: (4, 8)
b_hidden  = hidden_layer.get_weights()[1]  # shape: (8,)
W_output  = output_layer.get_weights()[0]  # shape: (8, 3)
b_output  = output_layer.get_weights()[1]  # shape: (3,)

print(f"\nWeight shapes:")
print(f"  W_hidden:  {W_hidden.shape}  (4 inputs × 8 neurons)")
print(f"  b_hidden:  {b_hidden.shape}")
print(f"  W_output:  {W_output.shape}  (8 inputs × 3 neurons)")
print(f"  b_output:  {b_output.shape}")

# Q8 conversion: multiply by 256 and round to the nearest integer
def to_q8(val):
    """Convert float to 16-bit signed Q8 integer."""
    q = int(round(float(val) * 256))
    q = max(-32768, min(32767, q))
    return q

def to_hex16(val):
    """Convert integer to 4-char uppercase hex (two's complement for negatives)."""
    q = to_q8(val)
    if q < 0:
        q = q + 65536  # two's complement 16-bit
    return f"{q:04X}"

# Save everything to .mem files in the weights directory
weights_dir = os.path.join(os.path.dirname(__file__), '..', 'weights')
os.makedirs(weights_dir, exist_ok=True)

print("\nSaving memory files...")

# weights_hidden.mem: 32 values
# Layout: neuron 0 weights (inputs 0,1,2,3), neuron 1 weights, ..., neuron 7
# W_hidden shape is (4, 8) so W_hidden[:,n] = weights for neuron n
with open(os.path.join(weights_dir, 'weights_hidden.mem'), 'w') as f:
    for neuron in range(8):
        for inp in range(4):
            f.write(to_hex16(W_hidden[inp, neuron]) + '\n')
print(f"Wrote weights_hidden.mem — {8*4} values")

# weights_output.mem: 24 values
# W_output shape is (8, 3), W_output[:,n] = weights for output neuron n
with open(os.path.join(weights_dir, 'weights_output.mem'), 'w') as f:
    for neuron in range(3):
        for inp in range(8):
            f.write(to_hex16(W_output[inp, neuron]) + '\n')
print(f"Wrote weights_output.mem — {3*8} values")

# biases_hidden.mem: 8 values
with open(os.path.join(weights_dir, 'biases_hidden.mem'), 'w') as f:
    for n in range(8):
        f.write(to_hex16(b_hidden[n]) + '\n')
print(f"Wrote biases_hidden.mem — 8 values")

print("Wrote output weights and all biases.")

# Save some test samples to verify in hardware
with open(os.path.join(weights_dir, 'test_data.mem'), 'w') as f:
    for i in range(10):
        for feat in range(4):
            f.write(to_hex16(X_test[i, feat]) + '\n')
        f.write(f"{y_test[i]:04X}" + '\n')  # label as hex
print("Wrote test_data.mem with 10 samples.")

# Double check predictions to make sure everything looks right
print("\nVerifying first 5 test predictions:")
preds = np.argmax(model.predict(X_test[:5], verbose=0), axis=1)
for i in range(5):
    status = "OK" if preds[i] == y_test[i] else "MISMATCH"
    print(f"  Sample {i}: expected {y_test[i]}, predicted {preds[i]} ({status})")

# Check for Q8 overflows just in case
all_weights = np.concatenate([W_hidden.flatten(), W_output.flatten(),
                               b_hidden.flatten(), b_output.flatten()])
max_abs = np.max(np.abs(all_weights))
print(f"\nMax weight: {max_abs:.4f} (Q8 value: {to_q8(max_abs)})")
if to_q8(max_abs) < 32767:
    print("Everything fits in 16 bits.")
else:
    print("Warning: some values are too large and might be clipped.")

print("\nDone. All .mem files are ready.")
