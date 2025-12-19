import numpy as np
import matplotlib.pyplot as plt
import os


CSV_FILE_PATH = '/Users/ibrahim900/PycharmProjects/EEA_GNN/zombies_data.csv'


def read_csv_manually(filepath):
    """
    Manually parses the CSV to extract features (Speed, Ammo) and labels (Survival).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Could not find file: {filepath}. "
                                f"Please make sure you downloaded the dataset and renamed it to '{os.path.basename(filepath)}'.")

    speed_list = []
    ammo_list = []
    labels_list = []

    with open(filepath, 'r') as f:
        lines = f.readlines()
        # Skip header (index 0)
        for line in lines[1:]:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                try:
                    # Column 0: Sprint Speed, Column 1: Ammo Clips, Column 2: Survive
                    speed_list.append(float(parts[0]))
                    ammo_list.append(float(parts[1]))
                    labels_list.append(float(parts[2]))
                except ValueError:
                    continue

    X = np.column_stack((speed_list, ammo_list))
    Y = np.array(labels_list)
    return X, Y


def normalize_features(X):
    """
    Min-Max normalization.
    Crucial for Gradient Descent when features have different scales (Speed ~20, Ammo ~1-5).
    """
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1

    X_norm = (X - min_vals) / ranges
    return X_norm, min_vals, max_vals


def sigmoid(z):
    """The activation function that squishes output between 0 and 1."""
    return 1 / (1 + np.exp(-z))


def compute_cost(X, Y, w, b):
    """
    Computes Binary Cross Entropy Loss (Log Loss).
    Cost = -(1/m) * sum( y*log(pred) + (1-y)*log(1-pred) )
    """
    m = len(Y)
    z = np.dot(X, w) + b
    predictions = sigmoid(z)

    epsilon = 1e-15
    cost = (-1 / m) * np.sum(Y * np.log(predictions + epsilon) + (1 - Y) * np.log(1 - predictions + epsilon))
    return cost


def train_logistic_regression(X, Y, learning_rate, epochs):
    """
    Performs Gradient Descent to find the best weights (w) and bias (b).
    """
    m, n_features = X.shape
    # Initialize weights to zeros
    w = np.zeros(n_features)
    b = 0.0
    cost_history = []

    for i in range(epochs):
        # Forward Pass
        z = np.dot(X, w) + b
        predictions = sigmoid(z)

        # Calculate Gradients
        # derivative of Cost w.r.t weights = (1/m) * X.T * (Predictions - Y)
        dw = (1 / m) * np.dot(X.T, (predictions - Y))
        db = (1 / m) * np.sum(predictions - Y)

        # Update Parameters
        w -= learning_rate * dw
        b -= learning_rate * db

        # Record Cost
        if i % 100 == 0:
            cost = compute_cost(X, Y, w, b)
            cost_history.append(cost)

    return w, b, cost_history


def predict(speed, ammo, w, b, min_vals, max_vals):
    """
    Predicts probability for a new single input.
    Must normalize the input using the *original training data's* stats.
    """
    input_features = np.array([speed, ammo])
    # Normalize
    input_norm = (input_features - min_vals) / (max_vals - min_vals)

    # Calculate Logits
    z = np.dot(input_norm, w) + b
    probability = sigmoid(z)

    return probability


def main():
    print("--- Z-Day Survival Algorithm ---")

    # Load Data
    print(f"Loading data from {CSV_FILE_PATH}...")
    X_raw, Y = read_csv_manually(CSV_FILE_PATH)

    # Normalize
    X_norm, min_vals, max_vals = normalize_features(X_raw)

    # Train Model
    LEARNING_RATE = 0.1
    EPOCHS = 5000
    print(f"Training Logistic Regression ({EPOCHS} epochs)...")

    w, b, cost_history = train_logistic_regression(X_norm, Y, LEARNING_RATE, EPOCHS)

    print("Training Complete.")
    print(f"Weights: {w}")
    print(f"Bias: {b:.4f}")

    # Test Prediction (Task Requirement)
    # Runner: 25 km/h, 1 Ammo Clip
    test_speed = 25
    test_ammo = 1
    prob = predict(test_speed, test_ammo, w, b, min_vals, max_vals)

    print("-" * 40)
    print(f"TEST PREDICTION (Speed: {test_speed} km/h, Ammo: {test_ammo})")
    print(f"Survival Probability: {prob:.4f} ({prob * 100:.2f}%)")
    status = "SURVIVOR" if prob >= 0.5 else "INFECTED"
    print(f"Prediction: {status}")
    print("-" * 40)

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Cost History
    ax1.plot(cost_history)
    ax1.set_title("Training Cost (Loss) over Time")
    ax1.set_xlabel("Iterations (x100)")
    ax1.set_ylabel("Log Loss")
    ax1.grid(True)

    # Plot 2: Decision Boundary
    survivors = X_raw[Y == 1]
    infected = X_raw[Y == 0]
    ax2.scatter(survivors[:, 0], survivors[:, 1], color='green', label='Survived (1)')
    ax2.scatter(infected[:, 0], infected[:, 1], color='red', label='Infected (0)')

    # Decision Boundary Line
    # The boundary is where sigmoid(z) = 0.5, which means z = 0.
    # w1*x1_norm + w2*x2_norm + b = 0
    # We solve for x2_norm (Ammo) based on x1_norm (Speed) to draw the line.

    x1_values_norm = np.array([0, 1])  # Min and Max normalized speed
    x2_values_norm = -(w[0] * x1_values_norm + b) / w[1]  # Solve for x2

    # Convert these boundary points back to Real Scale for plotting
    x1_real = x1_values_norm * (max_vals[0] - min_vals[0]) + min_vals[0]
    x2_real = x2_values_norm * (max_vals[1] - min_vals[1]) + min_vals[1]

    ax2.plot(x1_real, x2_real, color='blue', linewidth=2, linestyle='--', label='Decision Boundary')

    ax2.set_title("Survival Decision Boundary")
    ax2.set_xlabel("Sprint Speed (km/h)")
    ax2.set_ylabel("Ammo Clips")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()