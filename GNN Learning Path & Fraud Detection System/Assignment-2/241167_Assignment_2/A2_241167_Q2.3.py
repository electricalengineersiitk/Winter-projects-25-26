import numpy as np

class LinearSVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lam = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        _, n_features = X.shape
        y = np.where(y <= 0, -1, 1)

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for i in range(len(X)):
                if y[i] * (np.dot(X[i], self.w) - self.b) >= 1:
                    self.w -= self.lr * (2 * self.lam * self.w)
                else:
                    self.w -= self.lr * (2 * self.lam * self.w - y[i] * X[i])
                    self.b -= self.lr * y[i]

    def predict(self, X):
        return np.sign(np.dot(X, self.w) - self.b)


X = np.array([[1, 2],
              [2, 3],
              [3, 3],
              [2, 1],
              [3, 2]])

y = np.array([1, 1, 1, -1, -1])

svm = LinearSVM()
svm.fit(X, y)
print("Predictions:", svm.predict(X))
