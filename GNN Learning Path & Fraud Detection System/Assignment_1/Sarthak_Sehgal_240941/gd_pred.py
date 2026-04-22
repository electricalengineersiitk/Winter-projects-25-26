# gd_price_prediction.py

import matplotlib.pyplot as plt

def load_data(filepath):
   
    x_values = []
    y_values = []

    with open(filepath, 'r') as f:
        # Skip header
        header = f.readline()

        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            sqft = float(parts[0])
            price = float(parts[1])

            x_values.append(sqft)
            y_values.append(price)

    return x_values, y_values


def compute_cost(x, y, m, b):
    """
    Mean Squared Error (cost function):
    J(m, b) = (1/n) * Σ (y_i - (m*x_i + b))^2
    """
    n = len(x)
    total_error = 0.0
    for xi, yi in zip(x, y):
        prediction = m * xi + b
        total_error += (yi - prediction) ** 2
    return total_error / n


def gradient_descent(x, y, learning_rate=1e-8, epochs=100000):
    """
    Performing Gradient Descent to learn m and b.

    Update rules:
      m := m - α * (dJ/dm)
      b := b - α * (dJ/db)

    where:
      dJ/dm = (-2/n) * Σ x_i * (y_i - (m*x_i + b))
      dJ/db = (-2/n) * Σ (y_i - (m*x_i + b))
    """
    m = 0.0  # initial slope
    b = 0.0  # initial intercept
    n = len(x)

    for epoch in range(epochs):
        dm = 0.0
        db = 0.0

        for xi, yi in zip(x, y):
            prediction = m * xi + b
            error = yi - prediction
            dm += -2 * xi * error
            db += -2 * error

        dm /= n
        db /= n

        m = m - learning_rate * dm
        b = b - learning_rate * db

        # (Optional) print progress every some epochs
        if (epoch + 1) % 20000 == 0:
            cost = compute_cost(x, y, m, b)
            print(f"Epoch {epoch + 1}/{epochs}, Cost: {cost:.2f}, m: {m:.4f}, b: {b:.4f}")

    return m, b


def predict(x, m, b):
    """Prediction function: price = m * x + b"""
    return m * x + b


def plot_best_fit_line(x, y, m, b, title="Best Fit Line (Gradient Descent)"):
    """
    Plot scatter of data and the best fit line using m, b from gradient descent.
    """
    plt.figure()

    plt.scatter(x, y, label="Data Points")

    x_min, x_max = min(x), max(x)
    line_x = [x_min, x_max]
    line_y = [m * x_min + b, m * x_max + b]
    plt.plot(line_x, line_y, label="Best Fit Line (GD)", linewidth=2)

    plt.xlabel("Square Footage")
    plt.ylabel("Price")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    
    x_values, y_values = load_data("housing_data.csv")

    # 2. Learn parameters using gradient descent
    m, b = gradient_descent(
        x_values,
        y_values,
        learning_rate=1e-8,  # tuned for this data scale
        epochs=100000
    )
    print(f"GD slope (m): {m}")
    print(f"GD intercept (b): {b}")

    # 3. Predict price for 2,500 sq ft
    sqft_to_predict = 2500
    predicted_price = predict(sqft_to_predict, m, b)
    print(f"Predicted price for a {sqft_to_predict} sq ft house: {predicted_price:.2f}")

    # 4. Plot best fit line
    plot_best_fit_line(x_values, y_values, m, b, title="Best Fit Line (Gradient Descent)")
