import numpy as np


class LinearSVM:
    def _init_(self, learning_rate=0.001, lmd=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.lmd = lmd
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        # Convert labels to {-1, 1} if they are {0, 1}
        if set(np.unique(y)) == {0, 1}:
            y = np.where(y == 0, -1, 1)

        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for i in range(n_samples):
                x_i = X[i]
                y_i = y[i]

                # Check margin condition
                if y_i * (np.dot(self.w, x_i) - self.b) >= 1:
                    # Update only based on regularization
                    self.w -= self.learning_rate * (2 * self.lmd * self.w)
                else:
                    # Update based on regularization + hinge loss
                    self.w -= self.learning_rate * (2 * self.lmd * self.w - y_i * x_i)
                    self.b -= self.learning_rate * y_i

        return self

    def predict(self, X):
        return np.sign(np.dot(X, self.w) - self.b)