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

def main():
    print(f"Loading data from: {CSV_FILE_PATH}...")
    x, y = read_csv_manually(CSV_FILE_PATH)

    # m = (N * sum(xy) - sum(x)sum(y)) / (N * sum(x^2) - (sum(x))^2)
    # b = (sum(y) - m * sum(x)) / N

    N = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x_sq = np.sum(x ** 2)

    numerator = (N * sum_xy) - (sum_x * sum_y)
    denominator = (N * sum_x_sq) - (sum_x ** 2)

    m = numerator / denominator
    b = (sum_y - (m * sum_x)) / N

    # Output
    print("-" * 30)
    print(f"OLS Slope (m): {m:.4f}")
    print(f"OLS Intercept (b): {b:.4f}")

    # Prediction for 2,500 sq ft
    target_sq_ft = 2500
    predicted_price = (m * target_sq_ft) + b
    print(f"Predicted Price for 2,500 sq ft: ${predicted_price:.2f}")
    print("-" * 30)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='blue', label='Actual Data')
    plt.plot(x, m * x + b, color='red', linewidth=2, label='Line of Best Fit (OLS)')
    plt.title('Housing Price Prediction (OLS Method)')
    plt.xlabel('Square Feet')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

if __name__ == "__main__":
    main()
