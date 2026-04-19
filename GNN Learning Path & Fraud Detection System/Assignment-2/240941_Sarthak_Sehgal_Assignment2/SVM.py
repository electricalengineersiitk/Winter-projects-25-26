import numpy as np

class LinearSVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        # Train  
        n_samples, n_features = X.shape

        # Convert labels {0,1} -> {-1,1}
        y_ = np.where(y <= 0, -1, 1)

        # Initialize weights and bias
        self.w = np.zeros(n_features)
        self.b = 0

        # Gradient Descent
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1

                if condition:
                    # Correctly classified & outside margin
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    # Misclassified or inside margin
                    self.w -= self.lr * (
                        2 * self.lambda_param * self.w - y_[idx] * x_i
                    )
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        """
        Predict class labels {-1,1}
        """
        linear_output = np.dot(X, self.w) - self.b
        return np.sign(linear_output)



# Example usage (can be kept or removed)

if __name__ == "__main__":
    # Simple linearly separable dataset
    X = np.array([
        [1, 2],
        [2, 3],
        [3, 3],
        [2, 1],
        [3, 2],
        [4, 3]
    ])

    y = np.array([1, 1, 1, 0, 0, 0])

    svm = LinearSVM(
        learning_rate=0.001,
        lambda_param=0.01,
        n_iters=1000
    )

    svm.fit(X, y)
    predictions = svm.predict(X)

    print("Predictions:", predictions)
