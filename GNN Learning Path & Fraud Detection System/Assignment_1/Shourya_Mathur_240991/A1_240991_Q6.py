import numpy as np
import matplotlib.pyplot as plt

file_path = "/content/housing_prices - housing_prices.csv"

square_footage = []
price = []

with open(file_path, "r") as f:
    next(f)
    for line in f:
        parts = line.strip().split(",")
        if len(parts) < 2:
            continue
        square_footage.append(float(parts[0]))
        price.append(float(parts[1]))

x = np.array(square_footage)
y = np.array(price)

x_avg = np.mean(x)
y_avg = np.mean(y)

num = np.sum((x - x_avg) * (y - y_avg))
den = np.sum((x - x_avg)**2)
m = num / den

b = y_avg - m * x_avg

print("Slope:", m)
print("Intercept:", b)

x_input = 2500
pred_Price = m * x_input + b

print(f"Predicted Price for 2500 sq ft: {pred_Price:.2f}")

plt.scatter(x, y, label="Data Points")
plt.plot(x, m*x + b, label="Best Fit Line", linewidth=2)
plt.xlabel("Square Feet")
plt.ylabel("Price")
plt.legend()
plt.show()