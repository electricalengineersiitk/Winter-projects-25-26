
import numpy as np

# -----------------------------
# XOR Dataset
# -----------------------------
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([[0], [1], [1], [0]])

# -----------------------------
# Activation functions
# -----------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

# -----------------------------
# Initialize weights and biases
# -----------------------------
np.random.seed(42)

W1 = np.random.randn(2, 2)   # input -> hidden
b1 = np.zeros((1, 2))

W2 = np.random.randn(2, 1)   # hidden -> output
b2 = np.zeros((1, 1))

# -----------------------------
# Training parameters
# -----------------------------
learning_rate = 0.1
epochs = 10000

# -----------------------------
# Training loop
# -----------------------------
for epoch in range(epochs):

    # ---- Forward pass ----
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)

    z2 = np.dot(a1, W2) + b2
    y_hat = sigmoid(z2)

    # ---- Loss (Mean Squared Error) ----
    loss = np.mean((y - y_hat) ** 2)

    # ---- Backpropagation ----
    error_output = y_hat - y
    delta_output = error_output * sigmoid_derivative(y_hat)

    error_hidden = np.dot(delta_output, W2.T)
    delta_hidden = error_hidden * sigmoid_derivative(a1)

    # ---- Update weights and biases ----
    W2 -= learning_rate * np.dot(a1.T, delta_output)
    b2 -= learning_rate * np.sum(delta_output, axis=0, keepdims=True)

    W1 -= learning_rate * np.dot(X.T, delta_hidden)
    b1 -= learning_rate * np.sum(delta_hidden, axis=0, keepdims=True)

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# -----------------------------
# Testing
# -----------------------------
print("\nFinal Predictions:")
print(np.round(y_hat))
print("\nActual Labels:")
print(y)
