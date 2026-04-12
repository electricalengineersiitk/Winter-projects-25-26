import numpy as np

class LinearSVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters

    def fit(self, X, y):
        samples_count, features_count = X.shape

        y = np.where(y == 0, -1, 1)

        self.w = np.zeros(features_count)
        self.b = 0

        for epoch in range(self.n_iters):
            for i in range(samples_count):
                if y[i] * (np.dot(self.w, X[i]) - self.b) >= 1:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - y[i] * X[i])
                    self.b -= self.lr * y[i]

    def predict(self, X):
        linear_output = np.dot(X, self.w) - self.b
        return np.sign(linear_output)
