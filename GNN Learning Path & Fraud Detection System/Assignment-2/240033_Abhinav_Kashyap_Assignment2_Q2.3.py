import numpy as np

class LinearSVM:
    def __init__(self, lr=0.001, lam=0.01, iters=1000):
        self.lr = lr
        self.lam = lam
        self.iters = iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)
        
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.iters):
            for i, x in enumerate(X):
                condition = y_[i] * (np.dot(x, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lam * self.w)
                else:
                    self.w -= self.lr * (2 * self.lam * self.w - np.dot(x, y_[i]))
                    self.b -= self.lr * y_[i]

    def predict(self, X):
        linear = np.dot(X, self.w) - self.b
        return np.sign(linear)