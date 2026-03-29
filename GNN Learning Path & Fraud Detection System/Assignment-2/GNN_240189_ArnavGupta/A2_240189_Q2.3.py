import numpy as np

class LinearSVM:
    def __init__(self, n=0.001, lambda_param=0.01, itr=1000):
        self.lr = n
        self.lambda_param = lambda_param
        self.itr = itr
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        y_ = np.where(y <= 0, -1, 1)

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.itr):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(self.w, x_i) - self.b) >= 1

                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (
                        2 * self.lambda_param * self.w - y_[idx] * x_i
                    )
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        linear_output = np.dot(X, self.w) - self.b
        return np.sign(linear_output)
