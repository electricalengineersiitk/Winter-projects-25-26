import numpy as np

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def sigmoid_derivative(A):
    return A * (1 - A)

def compute_loss(A2, Y):
    # Binary Cross-Entropy Loss
    m = Y.shape[0]
    log_probs = np.multiply(Y, np.log(A2 + 1e-15)) + np.multiply((1 - Y), np.log(1 - A2 + 1e-15))
    return - (1 / m) * np.sum(log_probs)

# Data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([0, 1, 1, 0]).reshape(-1, 1)

# Architecture
N_INPUT = 2
N_HIDDEN = 4
N_OUTPUT = 1
LEARNING_RATE = 0.05
N_ITERS = 50000
m = X.shape[0]

W1 = np.random.randn(N_INPUT, N_HIDDEN) * 0.5
b1 = np.zeros((1, N_HIDDEN))
W2 = np.random.randn(N_HIDDEN, N_OUTPUT) * 0.5
b2 = np.zeros((1, N_OUTPUT))

print(f"Training XOR MLP ({N_ITERS} iterations)")

for i in range(N_ITERS):
    # FORWARD PROPAGATION
    # Layer 1
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)  # Hidden Layer Activation

    # Layer 2
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)  # Output Layer Activation (Prediction)

    # BACKPROPAGATION
    # Output Layer Gradients (dL/dZ2)
    dZ2 = A2 - Y
    dW2 = (1 / m) * np.dot(A1.T, dZ2)
    db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)

    # Hidden Layer Gradients (dL/dZ1)
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * sigmoid_derivative(A1)
    dW1 = (1 / m) * np.dot(X.T, dZ1)
    db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)

    # Parameter update
    W1 -= LEARNING_RATE * dW1
    b1 -= LEARNING_RATE * db1
    W2 -= LEARNING_RATE * dW2
    b2 -= LEARNING_RATE * db2

# Final prediction run
Z1 = np.dot(X, W1) + b1
A1 = sigmoid(Z1)
Z2 = np.dot(A1, W2) + b2
A2 = sigmoid(Z2)

# Convert probabilities to binary
Y_prob = A2.copy()
Y_pred = (A2 > 0.5).astype(int).flatten()
accuracy = np.mean(Y_pred == Y.flatten())

print("Final Results")
print(f"Input (X):\n{X.tolist()}")
print(f"True Labels (Y): {Y.flatten().tolist()}")
print(f"Predicted probs: {Y_prob.flatten().tolist()}")
print(f"Predictions:     {Y_pred.tolist()}")
print(f"Final Accuracy: {accuracy * 100:.2f}%")