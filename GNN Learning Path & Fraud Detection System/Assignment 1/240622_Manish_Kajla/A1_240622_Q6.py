import numpy as np
import matplotlib.pyplot as plt


# 1. The Data
X = np.array([1100, 1400, 1425, 1550, 1600, 1700, 1750, 1800, 1875, 2000,2100, 2250, 2300, 2400, 2450, 2600, 2800, 2900, 3000, 3150, 3300])
Y = np.array([199000, 245000, 230000, 215000, 280000, 295000, 345000, 315000, 325000, 360000,350000, 385000, 390000, 425000, 415000, 455000, 465000, 495000, 510000, 545000, 570000])


# 2. Scaling
x_mean = np.mean(X)
x_std = np.std(X)
y_mean = np.mean(Y)
y_std = np.std(Y)

X_scaled = (X - x_mean) / x_std
Y_scaled = (Y - y_mean) / y_std


# 3. Gradient Descent
m = 0  # Slope
b = 0  # Intercept
L = 0.01  # Learning Rate
epochs = 1000  # Standard amount
n = len(X)

for i in range(epochs):
    # Current guess
    y_pred = m * X_scaled + b
    
    # Calculate errors
    d_m = (-2/n) * sum(X_scaled * (Y_scaled - y_pred))
    d_b = (-2/n) * sum(Y_scaled - y_pred)
    
    # Update weights
    m = m - L * d_m
    b = b - L * d_b


# 4. Predict
target = 2500

target_scaled = (target - x_mean) / x_std
pred_scaled = m * target_scaled + b
final_price = pred_scaled * y_std + y_mean
print(f"Prediction for 2,500 sqft: {final_price:.2f}")


# 5. Plotting
plt.scatter(X, Y, color='blue')

line_x = np.linspace(min(X), max(X), 100)

line_x_scaled = (line_x - x_mean) / x_std
line_y_scaled = m * line_x_scaled + b
line_y = line_y_scaled * y_std + y_mean

plt.plot(line_x, line_y, color='red')
plt.xlabel('Square Feet')
plt.ylabel('Price')
plt.show()