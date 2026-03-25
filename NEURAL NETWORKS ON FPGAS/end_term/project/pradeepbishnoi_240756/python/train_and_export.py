# Importing Modules
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Loading the Iris dataset and creating variables for training
iris = load_iris()
X = iris.data
y = iris.target

# Scaling the inputs to avoid Weight imbalance and also so that all inputs lie in the same range
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Model Training
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(4,)),
    tf.keras.layers.Dense(8, activation='relu', name='hidden_layer'),
    tf.keras.layers.Dense(3, activation='softmax', name='output_layer'),
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print('Training the model....')

model.fit(X_train, y_train, epochs=25, batch_size=4, verbose=1)

# Performance Evaluation
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

print('Test loss:', loss)
print(f"Test Accuracy: {accuracy*100:.2f}%")

# Extracting Weights
weights = np.concatenate([
    model.layers[0].get_weights()[0].flatten(),
    model.layers[1].get_weights()[0].flatten()
])

# Extracting Biases
biases = np.concatenate([
    model.layers[0].get_weights()[1],
    model.layers[1].get_weights()[1]
])

# Quantising
def quantize_to_16bit(values):
    # Multiply by 256 (Q8 format) and round
    quantized = np.round(values * 256).astype(int)
    # Clip to 16-bit signed integer range [-32768, 32767]
    return np.clip(quantized, -32768, 32767)

weights = quantize_to_16bit(weights)
biases = quantize_to_16bit(biases)

# Converting 16-bit digits into 4 char long hex format
hex_weights = [format(int(w) & 0xFFFF, '04x') for w in weights]
hex_biases = [format(int(b) & 0xFFFF, '04x') for b in biases]

# Storing Weights in weights.mem
weights_file = 'weights/weights.mem'
with open('weights.mem', 'w') as f:
    for h_w in hex_weights:
        f.write(f"{h_w}\n")

# Storing Biases in biases.mem
biases_file = 'weights/biases.mem'
with open('biases.mem', 'w') as f:
    for h_b in hex_biases:
        f.write(f"{h_b}\n")

print("Successfully stored weights.mem and biases.mem")

# Taking 10 test inputs
total_test = X_test.shape[0]
random_indices = np.random.choice(total_test, size=10, replace=False)

x_raw = X_test[random_indices]
y_raw = y_test[random_indices]

# Quantising test inputs
x_raw = quantize_to_16bit(x_raw)

# Reshaping labels to stack them with the inputs
y_raw = y_raw.reshape(-1, 1).astype(int)

# Stacking test inputs and labels
test_matrix = np.hstack((x_raw, y_raw))

# Storing test inputs in test_data.mem
test_file = 'weights/test_data.mem'
with open(test_file, 'w') as f:
    for row in test_matrix:
        # Convert each value in the row to a 4-digit hex string
        # format(val & 0xFFFF, '04x') ensures:
        # - Negative numbers use Two's Complement (e.g., -1 -> ffff)
        # - Positive numbers are padded with zeros (e.g., 2 -> 0002)
        hex_row = [format(int(val) & 0xFFFF, '04x') for val in row]

        # Join the 5 hex values with a space and add a newline
        f.write(" ".join(hex_row) + "\n")

print(f"Successfully created {test_file}")

