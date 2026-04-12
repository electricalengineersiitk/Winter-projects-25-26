import numpy as np
import matplotlib.pyplot as plt

csv_content = """SquareFootage,Price
1100,199000
1400,245000
1425,230000
1550,215000
1600,280000
1700,295000
1750,345000
1800,315000
1875,325000
2000,360000
2100,350000
2250,385000
2300,390000
2400,425000
2450,415000
2600,455000
2800,465000
2900,495000
3000,510000
3150,545000
3300,570000"""

lines = csv_content.strip().split('\n')[1:]
x_data = []
y_data = []
for line in lines:
    sq, price = line.split(',')
    x_data.append(float(sq))
    y_data.append(float(price))
X_raw = np.array(x_data)
Y = np.array(y_data)
X = np.column_stack((np.ones(len(X_raw)), X_raw))
theta = np.linalg.inv(X.T @ X) @ X.T @ Y
b, m = theta[0], theta[1]
print(f"Slope (m): {m:.4f}")
print(f"Intercept (b): {b:.4f}")
print(f"Model: y = {m:.2f}x + {b:.2f}")
query_sqft = 2500
prediction = m * query_sqft + b
print(f"Predicted price for 2,500 sq ft: ${prediction:.2f}")
plt.scatter(X_raw, Y, color='blue', label='Data')
plt.plot(X_raw, m * X_raw + b, color='red', label='OLS Fit')
plt.scatter(query_sqft, prediction, color='green', zorder=5, label='Prediction')
plt.xlabel('Square Footage')
plt.ylabel('Price')
plt.legend()
plt.show()