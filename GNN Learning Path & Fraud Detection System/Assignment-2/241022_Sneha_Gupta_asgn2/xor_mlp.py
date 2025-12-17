import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([[0], [1], [1], [0]])

np.random.seed(42)
W1 = np.random.randn(2, 2)
b1 = np.zeros((1, 2))
W2 = np.random.randn(2, 1)
b2 = np.zeros((1, 1))

learning_rate = 0.1
epochs = 10000


for _ in range(epochs):
   
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    y_hat = sigmoid(z2)

    
    loss = np.mean((y - y_hat) ** 2)

    
    d_y_hat = (y_hat - y) * sigmoid_derivative(y_hat)
    d_W2 = np.dot(a1.T, d_y_hat)
    d_b2 = np.sum(d_y_hat, axis=0, keepdims=True)

    d_a1 = np.dot(d_y_hat, W2.T)
    d_z1 = d_a1 * sigmoid_derivative(a1)
    d_W1 = np.dot(X.T, d_z1)
    d_b1 = np.sum(d_z1, axis=0, keepdims=True)

    
    W2 -= learning_rate * d_W2
    b2 -= learning_rate * d_b2
    W1 -= learning_rate * d_W1
    b1 -= learning_rate * d_b1


print("Final Predictions:")
print(np.round(y_hat))
