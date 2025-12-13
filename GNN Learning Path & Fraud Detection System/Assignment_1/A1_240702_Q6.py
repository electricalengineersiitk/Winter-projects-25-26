import numpy as np
import matplotlib.pyplot as plt

xs = [1100, 1400, 1600, 1700, 1500, 1800, 2000, 2200, 2400, 2600, 2800, 3000, 3200, 2100, 1900, 1750, 1550, 1350, 2500, 2900]
ys = [190000, 240000, 215000, 295000, 230000, 345000, 360000, 385000, 420000, 455000, 465000, 510000, 545000, 350000, 320000, 315000, 280000, 235000, 435000, 490000]

xs = np.array(xs)
ys = np.array(ys)

m = 0.0
b = 0.0
lr = 0.0000001
epochs = 50000
n = len(xs)

for i in range(epochs):
    y_pred = m * xs + b
    dj_dm = (-2/n) * np.sum(xs * (ys - y_pred))
    dj_db = (-2/n) * np.sum(ys - y_pred)
    m -= lr * dj_dm
    b -= lr * dj_db

pred = m * 2500 + b
print(f"Predicted price: {pred:.2f}")

plt.scatter(xs, ys, label="Data Points")
plt.plot(xs, m*xs + b,  color='red' , label="Best Fit Line", linewidth=2)
plt.plot(2500, pred, marker='*', color='green', label="Prediction for 2500 square footage",markersize=10)
plt.xlabel("Square Footage")
plt.ylabel("Price")
plt.title("Best Fit Line")
plt.legend()
plt.grid(True)
plt.show()