import numpy as np

# XOR DATASET
# Inputs- x1, x2
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]])

# Targets- XOR output
y = np.array([
    [0],
    [1],
    [1],
    [0]])

# ACTIVATION FUNCTION
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(a):
    return a * (1 - a)

# INITIALIZE PARAMETERS
np.random.seed(42)
# Input (2) -> Hidden (2)
W1 = np.random.randn(2, 2)
b1 = np.zeros((1, 2))

# Hidden (2) -> Output (1)
W2 = np.random.randn(2, 1)
b2 = np.zeros((1, 1))

learning_rate = 0.1 # neta
epochs = 10000 # iterations 

# itreation LOOP
for epoch in range(epochs):

    # Forward Pass
    z1 = np.dot(X, W1) + b1      # input to hidden
    a1 = sigmoid(z1)             # hidden output

    z2 = np.dot(a1, W2) + b2     # input to output
    y_hat = sigmoid(z2)          # predicted output

    # Loss (MSE)
    loss = np.mean((y - y_hat) ** 2)

    # Backpropagation
    # Output layer gradients
    d_out = (y_hat - y) * sigmoid_derivative(y_hat)

    dW2 = np.dot(a1.T, d_out)
    db2 = np.sum(d_out, axis=0, keepdims=True)

    # Hidden layer gradients
    d_hidden = np.dot(d_out, W2.T) * sigmoid_derivative(a1)

    dW1 = np.dot(X.T, d_hidden)
    db1 = np.sum(d_hidden, axis=0, keepdims=True)

    # Update Weights
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    # Print loss occasionally
    if epoch % 2000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# FINAL OUTPUT
print("\nFinal Predictions:")
for i in range(len(X)):
    print(f"Input: {X[i]} -> Predicted: {y_hat[i][0]:.4f}")
