import numpy as np
import matplotlib.pyplot as plt

data_str = """SprintSpeed,AmmoClips,Result
12,0,0
14.5,1,0
10,2,0
18,0,0
8.5,4,0
15,1,0
22,0,1
11,5,1
13,2,0
20.5,1,1
24,2,1
16,3,1
12.5,0,0
28,0,1
9,6,1
25,1,1
14,4,1
19,2,1
10.5,2,0
26.5,2,1
15.5,5,1
17,3,1"""

lines = data_str.strip().split('\n')[1:]
x_list = []
y_list = []

for line in lines:
    s, a, r = line.split(',')
    x_list.append([float(s), float(a)])
    y_list.append(float(r))

X = np.array(x_list)
y = np.array(y_list)

mu = np.mean(X, axis=0)
sigma = np.std(X, axis=0)
X_norm = (X - mu) / sigma

X_bias = np.column_stack((np.ones(len(y)), X_norm))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

alpha = 0.1
iterations = 2000
m, n = X_bias.shape
theta = np.zeros(n)
cost_history = []

for i in range(iterations):
    z = np.dot(X_bias, theta)
    h = sigmoid(z)
    gradient = np.dot(X_bias.T, (h - y)) / m
    theta -= alpha * gradient
    
    cost = (-y * np.log(h + 1e-5) - (1 - y) * np.log(1 - h + 1e-5)).mean()
    cost_history.append(cost)

test_runner = np.array([25, 1])
test_norm = (test_runner - mu) / sigma
test_bias = np.hstack(([1], test_norm))

prob = sigmoid(np.dot(test_bias, theta))
pred = 1 if prob >= 0.5 else 0

print(f"Weights: {theta}")
print(f"Test Prediction (25 km/h, 1 Clip): {prob:.4f}")
print(f"Class: {pred}")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(cost_history)
plt.title('Cost Function')
plt.xlabel('Iterations')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
colors = ['red' if val == 0 else 'green' for val in y]
plt.scatter(X[:, 0], X[:, 1], c=colors)
plt.scatter(test_runner[0], test_runner[1], c='blue', s=100, marker='*')

x_bound = np.array([X[:, 0].min(), X[:, 0].max()])
x_bound_norm = (x_bound - mu[0]) / sigma[0]
y_bound_norm = -(theta[0] + theta[1] * x_bound_norm) / theta[2]
y_bound = y_bound_norm * sigma[1] + mu[1]

plt.plot(x_bound, y_bound, 'k--')
plt.xlabel('Sprint Speed')
plt.ylabel('Ammo Clips')
plt.title('Decision Boundary')

plt.show()