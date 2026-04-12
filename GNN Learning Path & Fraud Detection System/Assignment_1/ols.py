import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("prices.csv", delimiter=",", skiprows=1)

x = data[:, 0]
y = data[:, 1]

n = len(x)

sum_x = np.sum(x)
sum_y = np.sum(y)
sum_xy = np.sum(x * y)
sum_x2 = np.sum(x * x)

m = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
b = (sum_y - m * sum_x) / n

test_house_size = 2500
predicted_price = m * test_house_size + b

print("Slope (m):", m)
print("Intercept (b):", b)
print(f"Predicted price for 2500 sq ft: {predicted_price:.2f}")

plt.scatter(x, y)
plt.plot(x, m * x + b)
plt.xlabel("House Size (sq ft)")
plt.ylabel("House Price")
plt.title("OLS Linear Regression")
plt.show()
