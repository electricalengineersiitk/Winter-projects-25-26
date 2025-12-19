import numpy as np
import matplotlib.pyplot as plt

data = [
    [12, 0, 0], [14.5, 1, 0], [10, 2, 0], [18, 0, 0], [8.5, 4, 0],
    [15, 1, 0], [22, 0, 1], [11, 5, 1], [13, 2, 0], [20.5, 1, 1],
    [24, 2, 1], [16, 3, 1], [12.5, 0, 0], [28, 0, 1], [9, 6, 1],
    [25, 1, 1], [14, 4, 1], [19, 2, 1], [10.5, 2, 0], [26.5, 2, 1],
    [15.5, 5, 1], [17, 3, 1]
]

X = np.array([row[:2] for row in data])
y = np.array([row[2] for row in data]).reshape(-1, 1)

min_speed = np.min(X[:, 0])
max_speed = np.max(X[:, 0])
range_speed = max_speed - min_speed
min_ammo = np.min(X[:, 1])
max_ammo = np.max(X[:, 1])
range_ammo = max_ammo - min_ammo

X_norm = np.zeros_like(X)
X_norm[:, 0] = (X[:, 0] - min_speed) / range_speed
X_norm[:, 1] = (X[:, 1] - min_ammo) / range_ammo

ones = np.ones((len(X), 1))
X_with_bias = np.hstack([ones, X_norm])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

theta = np.zeros(3)
learning_rate = 0.5
epochs = 1000
costs = []

for epoch in range(epochs):
    total_cost = 0
    grad_theta0 = 0
    grad_theta1 = 0
    grad_theta2 = 0
    
    for i in range(len(X)):
        x1 = X_with_bias[i, 1]
        x2 = X_with_bias[i, 2]
        y_true = y[i, 0]
        
        z = theta[0] + theta[1]*x1 + theta[2]*x2
        prediction = sigmoid(z)
        
        if y_true == 1:
            cost = -np.log(prediction + 1e-8)
        else:
            cost = -np.log(1 - prediction + 1e-8)
        total_cost += cost
        
        error = prediction - y_true
        grad_theta0 += error
        grad_theta1 += error * x1
        grad_theta2 += error * x2
    
    avg_cost = total_cost / len(X)
    if epoch % 100 == 0:
        costs.append(avg_cost)
    
    theta[0] -= learning_rate * (grad_theta0 / len(X))
    theta[1] -= learning_rate * (grad_theta1 / len(X))
    theta[2] -= learning_rate * (grad_theta2 / len(X))

plt.figure(figsize=(8, 3))
plt.plot(range(0, epochs, 100), costs, 'b-')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Training Cost Over Time')
plt.grid(True, alpha=0.3)
plt.show()

def predict(speed, ammo):
    speed_norm = (speed - min_speed) / range_speed
    ammo_norm = (ammo - min_ammo) / range_ammo
    z = theta[0] + theta[1]*speed_norm + theta[2]*ammo_norm
    probability = sigmoid(z)
    prediction = 1 if probability >= 0.5 else 0
    return probability, prediction

test_speed = 25
test_ammo = 1
prob, pred = predict(test_speed, test_ammo)
print(f"Test: {test_speed} km/h, {test_ammo} clip")
print(f"Probability: {prob:.4f}")
print(f"Prediction: {'SURVIVE' if pred == 1 else 'INFECTED'}")

speed_grid = np.linspace(5, 30, 100)
ammo_grid = np.linspace(-1, 7, 100)
speed_mesh, ammo_mesh = np.meshgrid(speed_grid, ammo_grid)
prob_grid = np.zeros_like(speed_mesh)

for i in range(speed_mesh.shape[0]):
    for j in range(speed_mesh.shape[1]):
        prob_grid[i, j], _ = predict(speed_mesh[i, j], ammo_mesh[i, j])

plt.figure(figsize=(8, 5))
plt.contourf(speed_mesh, ammo_mesh, prob_grid, alpha=0.3, levels=[0, 0.5, 1], colors=['#FF9999', '#99FF99'])
plt.contour(speed_mesh, ammo_mesh, prob_grid, levels=[0.5], colors='black', linewidths=2)

for i in range(len(X)):
    color = 'green' if y[i, 0] == 1 else 'red'
    marker = 'o' if y[i, 0] == 1 else 'x'
    plt.scatter(X[i, 0], X[i, 1], c=color, marker=marker, s=60, label='Survived' if y[i, 0] == 1 and i == 0 else 'Infected' if i == 0 else '')

plt.scatter(test_speed, test_ammo, c='blue', s=150, marker='*', label='Test Case')
plt.xlabel('Speed (km/h)')
plt.ylabel('Ammo Clips')
plt.title('Zombie Survival Decision Boundary')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()