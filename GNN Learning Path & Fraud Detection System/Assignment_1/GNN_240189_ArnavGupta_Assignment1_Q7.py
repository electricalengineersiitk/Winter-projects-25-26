import numpy as np
import matplotlib.pyplot as plt
import csv

file_path = "/content/zombies_data - Sheet1.csv"

SprintSpeed = []
AmmoClips = []
Result = []

with open(file_path, "r") as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        SprintSpeed.append(float(row[0]))
        AmmoClips.append(float(row[1]))
        Result.append(int(row[2]))

SprintSpeed = np.array(SprintSpeed)
AmmoClips = np.array(AmmoClips)
Result = np.array(Result)

def normalize(x):
    return (x - np.mean(x)) / np.std(x)

SprintSpeed_n = normalize(SprintSpeed)
AmmoClips_n = normalize(AmmoClips)

X = np.c_[np.ones(len(SprintSpeed)), SprintSpeed_n, AmmoClips_n]
y = Result.reshape(-1, 1)

m, n = X.shape

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(X @ theta)
    eps = 1e-6
    cost = -(1/m) * np.sum(y*np.log(h+eps) + (1-y)*np.log(1-h+eps))
    return cost

def gradient_descent(X, y, theta, lr, iters):
    cost_history = []
    for i in range(iters):
        h = sigmoid(X @ theta)
        gradient = (1/m) * (X.T @ (h - y))
        theta -= lr * gradient
        cost_history.append(compute_cost(X, y, theta))
    return theta, cost_history

theta = np.zeros((n, 1))
learning_rate = 0.1
iterations = 4000

theta, cost_history = gradient_descent(X, y, theta, learning_rate, iterations)

runner_SprintSpeed = 25
runner_AmmoClips = 1

runner_SprintSpeed_n = (runner_SprintSpeed - np.mean(SprintSpeed)) / np.std(SprintSpeed)
runner_AmmoClips_n = (runner_AmmoClips - np.mean(AmmoClips)) / np.std(AmmoClips)

x_test = np.array([1, runner_SprintSpeed_n, runner_AmmoClips_n]).reshape(1, 3)
prob = sigmoid(x_test @ theta)[0][0]

print("\nPrediction for Runner (25 km/h, 1 ammo clip):")
print("Survival Probability =", prob)
print("Predicted Class =", 1 if prob >= 0.5 else 0)

plt.figure(figsize=(7,5))
plt.plot(cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost (Loss)")
plt.title("Cost Dropping")
plt.grid(True)
plt.show()

plt.figure(figsize=(7,5))

for i in range(len(y)):
    if y[i] == 1:
        plt.scatter(SprintSpeed[i], AmmoClips[i], color="green")
    else:
        plt.scatter(SprintSpeed[i], AmmoClips[i], color="red")

x_vals = np.linspace(min(SprintSpeed), max(SprintSpeed), 100)
x_vals_n = (x_vals - np.mean(SprintSpeed)) / np.std(SprintSpeed)

y_boundary_n = -(theta[0] + theta[1]*x_vals_n) / theta[2]

y_boundary = y_boundary_n * np.std(AmmoClips) + np.mean(AmmoClips)

plt.plot(x_vals, y_boundary, label="Decision Boundary")
plt.xlabel("SprintSpeed")
plt.ylabel("AmmoClips")
plt.title("Logistic Regression Decision Boundary")
plt.grid(True)
plt.legend()
plt.show()
