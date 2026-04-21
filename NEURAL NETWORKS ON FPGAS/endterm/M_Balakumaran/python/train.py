import os
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# ==========================================
# 1. Load Dataset and Prepare Test Data
# ==========================================
iris = load_iris()
X, y = iris.data, iris.target

# Split the data to ensure we have a test set to draw 10 samples from
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Isolate exactly 10 test samples for the final .mem file
X_test_10 = X_test[:10]
y_test_10 = y_test[:10]

# ==========================================
# 2. Build and Train the Model
# ==========================================
model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, activation='relu', input_shape = X_train.shape[1:]),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
print("Training model...")
model.fit(X_train, y_train, epochs=100, verbose=2)
print("Training complete.\n")


# ==========================================
# 3. Quantization & Hex Conversion Helpers
# ==========================================
def quantize_q8(float_val):
    """Converts a float to a 16-bit Q8 fixed-point integer."""
    quantised = int(np.round(float_val * 256))
    # Keep it in 16-bit signed integer range
    quantised = max(-32768, min(32767, quantised))
    return quantised

def to_hex16(int_val):
    """Converts an integer to a 16-bit two's complement hex string."""
    # The & 0xFFFF ensures negative numbers format correctly for Verilog
    return f"{int_val & 0xFFFF:04x}"

# ==========================================
# 4. Extract and Save Weights/Biases
# ==========================================
os.makedirs('weights', exist_ok=True)

# Extract weights and biases from the layers
# Layer 0: Hidden Layer | Layer 1: Output Layer
w1, b1 = model.layers[0].get_weights()
w2, b2 = model.layers[1].get_weights()

print(w1.shape, b1.shape)
print(w2.shape, b2.shape)  # Should be (4, 8) and (8,)

print("Saving weights.mem...")
with open('weights/weights.mem', 'w') as f:
    # Flatten arrays so they are saved one number per line
    for w_matrix in [w1, w2]:
        for val in w_matrix.flatten():
            q_val = quantize_q8(val)
            f.write(to_hex16(q_val) + '\n')

print("Saving biases.mem...")
with open('weights/biases.mem', 'w') as f:
    for b_array in [b1, b2]:
        for val in b_array.flatten():
            q_val = quantize_q8(val)
            f.write(to_hex16(q_val) + '\n')

# ==========================================
# 5. Save Test Data
# ==========================================
print("Saving test_data.mem...")
with open('weights/test_data.mem', 'w') as f:
    for i in range(10):
        # Quantize and write the 4 input features
        for feature in X_test_10[i]:
            q_feat = quantize_q8(feature)
            f.write(to_hex16(q_feat) + '\n')
            
        # Write the expected output label
        # Labels are already integers (0, 1, or 2), so we skip the *256 quantization
        f.write(to_hex16(y_test_10[i]) + '\n')

print("All files saved successfully in the 'weights/' directory.")