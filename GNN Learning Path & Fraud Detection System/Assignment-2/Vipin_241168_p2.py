import numpy as np

class LinearSVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        y = np.where(y <= 0, -1, 1)

        # Initialize weights and bias
        self.w = np.zeros(n_features)
        self.b = 0

        # Gradient Descent optimization
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):

                margin = y[idx] * (np.dot(x_i, self.w) - self.b)

                # classification and outside margin
                if margin >= 1:
                    self.w -= self.learning_rate * (
                        2 * self.lambda_param * self.w
                    )

                # Misclassified or inside margin
                else:
                    self.w -= self.learning_rate * (
                        2 * self.lambda_param * self.w - y[idx] * x_i
                    )
                    self.b -= self.learning_rate * y[idx]

    def predict(self, X):
        """
        Predict class labels (-1 or +1)
        """
        linear_output = np.dot(X, self.w) - self.b
        return np.sign(linear_output)
# Example usage
X = np.array([
    [2, 3],
    [1, 1],
    [2, 1],
    [3, 2]
])

y = np.array([1, -1, -1, 1])

svm = LinearSVM()
svm.fit(X, y)

predictions = svm.predict(X)
print("Predictions:", predictions)
