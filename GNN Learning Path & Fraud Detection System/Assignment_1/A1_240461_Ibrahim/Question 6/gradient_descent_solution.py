import numpy as np
import matplotlib.pyplot as plt
import os

CSV_FILE_PATH = '/Users/ibrahim900/PycharmProjects/EEA_GNN/housing_prices.csv'

def read_csv_manually(filepath):
    """Parses the CSV file into X and Y arrays"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Could not find the file: {filepath}. "
                                f"Please check the CSV_FILE_PATH variable.")
    x_list = []
    y_list = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                try:
                    x_list.append(float(parts[0]))
                    y_list.append(float(parts[1]))
                except ValueError:
                    continue
    return np.array(x_list), np.array(y_list)


def normalize_data(data):
    """Scales data to range [0, 1]. Returns scaled data, min, and max."""
    min_val = np.min(data)
    max_val = np.max(data)
    scaled = (data - min_val) / (max_val - min_val)
    return scaled, min_val, max_val


def train_gradient_descent(x, y, learning_rate, epochs):
    """Iteratively updates m and b to minimize error."""
    m = 0.0
    b = 0.0
    n = len(x)
    loss_history = []

    for i in range(epochs):
        y_pred = m * x + b

        # Calculate Gradients (Derivatives)
        dm = (-2 / n) * np.sum(x * (y - y_pred))
        db = (-2 / n) * np.sum(y - y_pred)

        # Update Parameters
        m -= learning_rate * dm
        b -= learning_rate * db

        # Track loss for debugging
        loss = np.mean((y - y_pred) ** 2)
        loss_history.append(loss)

    return m, b, loss_history


def main():
    # Load Data
    print(f"Loading data from: {CSV_FILE_PATH}...")
    x_raw, y_raw = read_csv_manually(CSV_FILE_PATH)

    # Normalize Data (CRITICAL STEP)
    x_norm, x_min, x_max = normalize_data(x_raw)
    y_norm, y_min, y_max = normalize_data(y_raw)

    # Hyperparameters
    LEARNING_RATE = 0.01
    EPOCHS = 5000

    # 4. Train Model
    print(f"Training Gradient Descent for {EPOCHS} epochs...")
    m_norm, b_norm, history = train_gradient_descent(x_norm, y_norm, LEARNING_RATE, EPOCHS)

    # Prediction Logic
    target_sq_ft = 2500

    # Normalize input
    target_norm = (target_sq_ft - x_min) / (x_max - x_min)

    # Predict using normalized model
    price_pred_norm = m_norm * target_norm + b_norm

    # Denormalize output (Convert back to Dollars)
    price_predicted = price_pred_norm * (y_max - y_min) + y_min

    # Output Results
    print("-" * 30)
    print(f"GD Slope (Normalized): {m_norm:.4f}")
    print(f"GD Intercept (Normalized): {b_norm:.4f}")
    print(f"Predicted Price for 2,500 sq ft: ${price_predicted:.2f}")
    print("-" * 30)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(x_raw, y_raw, color='blue', label='Actual Data')

    line_x_norm = np.array([0, 1])  # Represents min and max
    line_y_norm = m_norm * line_x_norm + b_norm

    line_x_real = line_x_norm * (x_max - x_min) + x_min
    line_y_real = line_y_norm * (y_max - y_min) + y_min

    plt.plot(line_x_real, line_y_real, color='green', linewidth=2, label='Gradient Descent Fit')
    plt.title('Housing Price Prediction (Gradient Descent)')
    plt.xlabel('Square Feet')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()


if __name__ == "__main__":
    main()