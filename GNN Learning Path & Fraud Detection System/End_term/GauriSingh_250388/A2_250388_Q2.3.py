import numpy as np


class LinearSVM:
    def __init__(self, learning_rate=0.001, lmd=0.01, n_iters=1000):
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
    
from sklearn.model_selection import train_test_split
from sklearn import datasets

# Generate some synthetic data for demonstration
X, y = datasets.make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Linear SVM
svm = LinearSVM(learning_rate=0.01, lmd=0.001, n_iters=1000)
svm.fit(X_train, y_train)

# Make predictions on the test set
predictions = svm.predict(X_test)

print("Predictions:", predictions)
print("True labels:", y_test)

# Evaluate the model 
from sklearn.metrics import accuracy_score

# Convert y_test to {-1, 1} if it's {0, 1} to match SVM's output
y_test_converted = np.where(y_test == 0, -1, 1)

accuracy = accuracy_score(y_test_converted, predictions)
print(f"Model Accuracy: {accuracy:.2f}")