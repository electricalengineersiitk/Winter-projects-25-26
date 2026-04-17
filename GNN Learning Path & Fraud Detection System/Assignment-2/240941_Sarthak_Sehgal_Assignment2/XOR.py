import numpy as np


# Activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# XOR Dataset
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([[0], [1], [1], [0]])


# Initialize weights & biases
np.random.seed(0)

# 2 input neurons -> 4 hidden neurons -> 1 output neuron
W1 = np.random.randn(2, 4)
b1 = np.zeros((1, 4))

W2 = np.random.randn(4, 1)
b2 = np.zeros((1, 1))

learning_rate = 0.5
epochs = 20000


# Training loop
for epoch in range(epochs):

    # Forward pass
    z1 = X @ W1 + b1
    a1 = sigmoid(z1)

    z2 = a1 @ W2 + b2
    y_hat = sigmoid(z2)

    # Binary Cross-Entropy Loss 
    loss = -np.mean(
        y * np.log(y_hat + 1e-8) +
        (1 - y) * np.log(1 - y_hat + 1e-8)
    )

    # Backpropagation
    d_z2 = y_hat - y
    d_W2 = a1.T @ d_z2
    d_b2 = np.sum(d_z2, axis=0, keepdims=True)

    d_a1 = d_z2 @ W2.T
    d_z1 = d_a1 * a1 * (1 - a1)
    d_W1 = X.T @ d_z1
    d_b1 = np.sum(d_z1, axis=0, keepdims=True)

    # Update parameters
    W2 -= learning_rate * d_W2
    b2 -= learning_rate * d_b2
    W1 -= learning_rate * d_W1
    b1 -= learning_rate * d_b1

    if epoch % 5000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")


# Final Predictions
print("\nFinal Predictions:")
for i in range(len(X)):
    print(f"{X[i]} -> {y_hat[i][0]:.4f}")
