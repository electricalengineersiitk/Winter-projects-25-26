# zday_logistic_regression.py

import numpy as np
import matplotlib.pyplot as plt

def load_data(filepath):
    X = []
    y = []

    with open(filepath, 'r') as f:
        header = f.readline()  # skip header

        for line in f:
            if line.strip() == "":
                continue
            speed, ammo, label = line.strip().split(',')
            X.append([float(speed), float(ammo)])
            y.append(int(label))

    return np.array(X), np.array(y)



# 2. Feature Normalization

def normalize_features(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    X_norm = (X - mean) / std
    return X_norm, mean, std



# 3. Sigmoid Function

def sigmoid(z):
    return 1 / (1 + np.exp(-z))



# 4. Cost Function

def compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(X @ theta)

    cost = - (1/m) * np.sum(
        y * np.log(h + 1e-9) +
        (1 - y) * np.log(1 - h + 1e-9)
    )
    return cost



# 5. Gradient Descent

def gradient_descent(X, y, alpha=0.1, iterations=2000):
    m, n = X.shape
    theta = np.zeros(n)
    cost_history = []

    for i in range(iterations):
        h = sigmoid(X @ theta)
        gradient = (1/m) * (X.T @ (h - y))
        theta -= alpha * gradient

        cost = compute_cost(X, y, theta)
        cost_history.append(cost)

    return theta, cost_history



# 6. Prediction Function

def predict(sample, theta, mean, std):
    sample_norm = (sample - mean) / std
    sample_norm = np.insert(sample_norm, 0, 1)  # bias
    prob = sigmoid(sample_norm @ theta)
    return prob



# 7. Plot Cost vs Iterations

def plot_cost(cost_history):
    plt.figure()
    plt.plot(cost_history)
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title("Cost Reduction During Training")
    plt.grid()
    plt.show()



# 8. Plot Decision Boundary

def plot_decision_boundary(X, y, theta, mean, std):
    plt.figure()

    # Plot data points
    for i in range(len(y)):
        if y[i] == 1:
            plt.scatter(X[i, 0], X[i, 1], c='green', marker='o')
        else:
            plt.scatter(X[i, 0], X[i, 1], c='red', marker='x')

    # Decision boundary
    x1_vals = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
    x1_vals_norm = (x1_vals - mean[0]) / std[0]

    x2_vals_norm = (-theta[0] - theta[1] * x1_vals_norm) / theta[2]
    x2_vals = x2_vals_norm * std[1] + mean[1]

    plt.plot(x1_vals, x2_vals, 'b', label="Decision Boundary")

    plt.xlabel("Sprint Speed (km/h)")
    plt.ylabel("Ammo Clips")
    plt.title("Z-Day Survival Decision Boundary")
    plt.legend()
    plt.grid()
    plt.show()



if __name__ == "__main__":

    # Load and normalize data
    X, y = load_data("zombie_survival_data.csv")
    X_norm, mean, std = normalize_features(X)

    # Add bias term
    X_bias = np.c_[np.ones(X_norm.shape[0]), X_norm]

    # Train model
    theta, cost_history = gradient_descent(
        X_bias, y,
        alpha=0.1,
        iterations=2000
    )

    print("Learned Parameters (theta):", theta)

    # Test prediction
    test_runner = np.array([25, 1])
    probability = predict(test_runner, theta, mean, std)

    print(f"\nTest Runner: 25 km/h, 1 Ammo Clip")
    print(f"Survival Probability: {probability:.4f}")
    print("Prediction:", "SURVIVES" if probability >= 0.5 else "INFECTED")

    # Visualizations
    plot_cost(cost_history)
    plot_decision_boundary(X, y, theta, mean, std)
