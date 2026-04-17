# ols_price_prediction.py

import matplotlib.pyplot as plt

def load_data(filepath):
    
    x_values = []
    y_values = []

    with open(filepath, 'r') as f:
        # Skip header
        header = f.readline()

        for line in f:
            line = line.strip()
            if not line:  # skip empty lines
                continue
            parts = line.split(',')
            # Assuming exactly 2 columns: [SquareFootage, Price]
            sqft = float(parts[0])
            price = float(parts[1])

            x_values.append(sqft)
            y_values.append(price)

    return x_values, y_values


def ols_linear_regression(x, y):
    """
    Computing slope (m) and intercept (b) using Ordinary Least Squares (closed form):

    m = Σ (x - x̄)(y - ȳ) / Σ (x - x̄)²
    b = ȳ - m * x̄
    """
    n = len(x)
    if n == 0:
        raise ValueError("Empty dataset")

    x_mean = sum(x) / n
    y_mean = sum(y) / n

    numerator = 0.0
    denominator = 0.0
    for xi, yi in zip(x, y):
        numerator += (xi - x_mean) * (yi - y_mean)
        denominator += (xi - x_mean) ** 2

    if denominator == 0:
        raise ValueError("Cannot compute slope because denominator is zero.")

    m = numerator / denominator
    b = y_mean - m * x_mean
    return m, b


def predict(x, m, b):
    """Prediction function: price = m * x + b"""
    return m * x + b


def plot_best_fit_line(x, y, m, b, title="Best Fit Line (OLS)"):
    """
    Plotting scatter of data and the best fit line.
    """
    plt.figure()

    # Scatter plot of data points
    plt.scatter(x, y, label="Data Points")

    # Line of best fit
    x_min, x_max = min(x), max(x)
    line_x = [x_min, x_max]
    line_y = [m * x_min + b, m * x_max + b]
    plt.plot(line_x, line_y, label="Best Fit Line", linewidth=2)

    plt.xlabel("Square Footage")
    plt.ylabel("Price")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 1. Load data
    x_values, y_values = load_data("housing_data.csv")

    # 2. Compute OLS parameters
    m, b = ols_linear_regression(x_values, y_values)
    print(f"OLS slope (m): {m}")
    print(f"OLS intercept (b): {b}")

    # 3. Predict price for 2,500 sq ft
    sqft_to_predict = 2500
    predicted_price = predict(sqft_to_predict, m, b)
    print(f"Predicted price for a {sqft_to_predict} sq ft house: {predicted_price:.2f}")

    # 4. Plot best fit line
    plot_best_fit_line(x_values, y_values, m, b, title="Best Fit Line (OLS)")
