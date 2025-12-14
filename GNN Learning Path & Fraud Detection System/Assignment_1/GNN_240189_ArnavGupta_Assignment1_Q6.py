import numpy as np
import matplotlib.pyplot as plt

file_path = "/content/housing_prices - housing_prices.csv"

SquareFootage = []
Price = []

with open(file_path, "r") as f:
    next(f)
    for line in f:
        parts = line.strip().split(",")
        if len(parts) < 2:
            continue
        SquareFootage.append(float(parts[0]))
        Price.append(float(parts[1]))

x = np.array(SquareFootage)
y = np.array(Price)

x_mean = np.mean(x)
y_mean = np.mean(y)

num = np.sum((x - x_mean) * (y - y_mean))
den = np.sum((x - x_mean)**2)
m = num / den

b = y_mean - m * x_mean

print("Slope (m):", m)
print("Intercept (b):", b)

x_input = 2500
pred_Price = m * x_input + b

print(f"Predicted Price for 2500 sq ft: {pred_Price:.2f}")

plt.scatter(x, y, label="Data Points")
plt.plot(x, m*x + b, label="Best Fit Line", linewidth=2)
plt.xlabel("Square Feet")
plt.ylabel("Price")
plt.title("Best Fit Line")
plt.legend()
plt.show()