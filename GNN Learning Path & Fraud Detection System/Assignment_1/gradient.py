import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("prices.csv", delimiter=",", skiprows=1)

x = data[:, 0]
y = data[:, 1]

m = 0
b = 0

learning_rate = 0.00000001
iterations = 200000
n = len(x)

for _ in range(iterations):
    y_pred = m * x + b

    dm = (-2/n) * np.sum(x * (y - y_pred))
    db = (-2/n) * np.sum(y - y_pred)

    m -= learning_rate * dm
    b -= learning_rate * db

test_house_size = 2500
predicted_price = m * test_house_size + b

print("Slope (m):", m)
print("Intercept (b):", b)
print(f"Predicted price for 2500 sq ft: {predicted_price:.2f}")

plt.scatter(x, y)
plt.plot(x, m * x + b)
plt.xlabel("House Size (sq ft)")
plt.ylabel("House Price")
plt.title("Gradient Descent Linear Regression")
plt.show()
