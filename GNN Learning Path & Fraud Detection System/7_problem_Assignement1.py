import numpy as np
import matplotlib.pyplot as plt
import math

# Sprint Speed (km/h)
X1 = np.array([12, 14.5, 10, 18, 8.5, 15, 22, 11, 13, 20.5, 24, 16, 12.5, 28, 9, 25, 14, 19, 10.5, 26.5, 15.5, 17])

# Ammo Clips
X2 = np.array([0, 1, 2, 0, 4, 1, 0, 5, 2, 1, 2, 3, 0, 0, 6, 1, 4, 2, 2, 2, 5, 3])

# Result 0 = Infected, 1 = Survived;
Y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1])

m = len(Y)

# NORMALIZATION;
X1_norm = (X1 - X1.mean()) / X1.std()
X2_norm = (X2 - X2.mean()) / X2.std()

# Add Bias Column;
X = np.column_stack((np.ones(m), X1_norm, X2_norm))

# SIGMOID FUNCTION
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# COST FUNCTION
def cost_function(X, Y, theta):
    h = sigmoid(X @ theta)
    cost = (-1/m) * np.sum(Y*np.log(h) + (1-Y)*np.log(1-h))
    return cost

# GRADIENT DESCENT
theta = np.zeros(3)
alpha = 0.1
iterations = 2000
cost_history = []

for _ in range(iterations):
    h = sigmoid(X @ theta)
    gradient = (1/m) * (X.T @ (h - Y))
    theta = theta - alpha * gradient
    cost_history.append(cost_function(X, Y, theta))

# TEST PREDICTION
test_speed = 25
test_ammo = 1

test_speed_norm = (test_speed - X1.mean()) / X1.std()
test_ammo_norm = (test_ammo - X2.mean()) / X2.std()

test_input = np.array([1, test_speed_norm, test_ammo_norm])
probability = sigmoid(test_input @ theta)

print("Survival Probability for Runner (25 km/h, 1 Ammo):", round(probability, 4))

# COST FUNCTION PLOT
plt.figure()
plt.plot(cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost Function Decreasing")
plt.show()

# DECISION BOUNDARY PLOT
plt.figure() 
plt.scatter(X1, X2)

x_values = np.linspace(min(X1), max(X1), 50)
x_values_norm = (x_values - X1.mean()) / X1.std()

y_values_norm = -(theta[0] + theta[1]*x_values_norm) / theta[2]
y_values = y_values_norm * X2.std() + X2.mean()

plt.plot(x_values, y_values)
plt.xlabel("Sprint Speed")
plt.ylabel("Ammo Clips")
plt.title("Decision Boundary")
plt.show()
