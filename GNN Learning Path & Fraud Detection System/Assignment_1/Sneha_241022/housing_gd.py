import csv
import numpy as np
import matplotlib.pyplot as plt


areas = []
prices = []

with open("housing_prices.csv", "r") as f:
    reader = csv.reader(f)
    header = next(reader)

    area_idx = header.index("SquareFootage")
    price_idx = header.index("Price")

    for row in reader:
        if not row:
            continue
        areas.append(float(row[area_idx]))
        prices.append(float(row[price_idx]))

areas = np.array(areas)
prices = np.array(prices)

m_samples = len(areas)
print("Loaded", m_samples, "data points.")


m = 0.0
b = 0.0


alpha = 0.0000001     
num_iters = 5000

cost_history = []

for i in range(num_iters):
   
    y_pred = m * areas + b

  
    error = y_pred - prices

    dm = (1/m_samples) * np.sum(error * areas)
    db = (1/m_samples) * np.sum(error)

    m -= alpha * dm
    b -= alpha * db

    cost = (1/(2*m_samples)) * np.sum(error**2)
    cost_history.append(cost)

    if (i+1) % 500 == 0:
        print(f"Iteration {i+1}: Cost = {cost:.4f}")

print("\nFinal slope m:", m)
print("Final intercept b:", b)


area_new = 2500
price_pred = m * area_new + b
print(f"\nPredicted price for 2500 sq ft: {price_pred:.2f}")


plt.scatter(areas, prices, label="Data points")

x_line = np.linspace(min(areas), max(areas), 100)
y_line = m * x_line + b

plt.plot(x_line, y_line, 'r', label="Gradient Descent Fit")

plt.xlabel("SquareFootage")
plt.ylabel("Price")
plt.title("Linear Regression Using Gradient Descent")
plt.legend()
plt.grid(True)
plt.show()


plt.plot(cost_history)
plt.xlabel("Iteration")
plt.ylabel("Cost (MSE)")
plt.title("Gradient Descent Cost Curve")
plt.grid(True)
plt.show()
