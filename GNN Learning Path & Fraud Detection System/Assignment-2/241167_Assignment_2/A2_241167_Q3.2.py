import numpy as np

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
y = np.array([[0], [1], [1], [0]])

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

np.random.seed(0)
W1 = np.random.randn(2, 2)
b1 = np.zeros((1, 2))
W2 = np.random.randn(2, 1)
b2 = np.zeros((1, 1))

learning_rate = 0.1
epochs = 10000

for _ in range(epochs):
    z1 = X.dot(W1) + b1
    a1 = sigmoid(z1)
    z2 = a1.dot(W2) + b2
    y_hat = sigmoid(z2)

    d_out = (y_hat - y) * sigmoid_derivative(y_hat)
    d_W2 = a1.T.dot(d_out)
    d_b2 = np.sum(d_out, axis=0, keepdims=True)

    d_hidden = d_out.dot(W2.T) * sigmoid_derivative(a1)
    d_W1 = X.T.dot(d_hidden)
    d_b1 = np.sum(d_hidden, axis=0, keepdims=True)

    W2 -= learning_rate * d_W2
    b2 -= learning_rate * d_b2
    W1 -= learning_rate * d_W1
    b1 -= learning_rate * d_b1

print("Final Predictions:")
print((y_hat > 0.5).astype(int))
