import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train(X, y, lr=0.1, epochs=10000):
    np.random.seed(50)
    w1 = np.random.randn(2, 2)
    b1 = np.random.randn(1, 2)
    w2 = np.random.randn(2, 1)
    b2 = np.random.randn(1, 1)

    for epoch in range(epochs):
        
        z1 = np.dot(X, w1) + b1
        a1 = sigmoid(z1)
        z2 = np.dot(a1, w2) + b2
        y_pred = sigmoid(z2)

        loss = np.mean((y - y_pred) ** 2)

        dl_dy = 2 * (y_pred - y) / y.shape[0]
        dl_z2 = dl_dy * (y_pred * (1 - y_pred))

        dl_w2 = np.dot(a1.T, dl_z2)
        dl_b2 = np.sum(dl_z2, axis=0, keepdims=True)

        dl_a1 = np.dot(dl_z2, w2.T)
        dl_z1 = dl_a1 * (a1 * (1 - a1))

        dl_w1 = np.dot(X.T, dl_z1)
        dl_b1 = np.sum(dl_z1, axis=0, keepdims=True)

        w2 -= lr * dl_w2
        b2 -= lr * dl_b2
        w1 -= lr * dl_w1
        b1 -= lr * dl_b1

    return w1, b1, w2, b2

X = np.array([[0, 0],[0, 1],[1, 0],[1, 1]], dtype=float)
y = np.array([[0], [1], [1], [0]], dtype=float)
W1, b1, W2, b2 = train(X, y)
