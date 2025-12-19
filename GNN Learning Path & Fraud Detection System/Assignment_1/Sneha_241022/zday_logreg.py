import csv
import numpy as np
import matplotlib.pyplot as plt


speeds = []  
ammos  = []  
labels = []   

with open("zday_data.csv", "r") as f:
    reader = csv.reader(f)
    header = next(reader)
    print("Header:", header)

   
    speed_col_name  = "SprintSpeed"
    ammo_col_name   = "AmmoClips"
    label_col_name  = "Result"

    speed_idx = header.index(speed_col_name)
    ammo_idx  = header.index(ammo_col_name)
    label_idx = header.index(label_col_name)

    for row in reader:
        if not row:
            continue
        speeds.append(float(row[speed_idx]))
        ammos.append(float(row[ammo_idx]))
        labels.append(float(row[label_idx]))

print(f"Loaded {len(speeds)} data points.")

X_raw = np.column_stack([speeds, ammos])  
y = np.array(labels).reshape(-1, 1)       
m = X_raw.shape[0]


mu = X_raw.mean(axis=0)
sigma = X_raw.std(axis=0)

X_norm = (X_raw - mu) / sigma


X = np.column_stack([np.ones((m, 1)), X_norm])


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(theta, X, y):
    m = X.shape[0]
    h = sigmoid(X @ theta)
    eps = 1e-15
    return -(1/m) * np.sum(
        y * np.log(h + eps) + (1 - y) * np.log(1 - h + eps)
    )

def gradient_descent(X, y, alpha=0.1, num_iters=1000):
    m, n = X.shape
    theta = np.zeros((n, 1))
    J_history = []

    for i in range(num_iters):
        h = sigmoid(X @ theta)
        grad = (1/m) * (X.T @ (h - y))
        theta -= alpha * grad

        cost = compute_cost(theta, X, y)
        J_history.append(cost)

        if (i+1) % 100 == 0:
            print(f"Iteration {i+1}, Cost = {cost:.4f}")

    return theta, J_history

theta, J_history = gradient_descent(X, y, alpha=0.1, num_iters=2000)

print("\nLearned parameters theta:")
print(theta)


runner_speed = 25.0
runner_ammo  = 1.0

runner_features = np.array([[runner_speed, runner_ammo]])
runner_norm = (runner_features - mu) / sigma
runner_X = np.column_stack([np.ones((1, 1)), runner_norm])

survival_prob = float(sigmoid(runner_X @ theta))
prediction = 1 if survival_prob >= 0.5 else 0

print(f"\nPrediction for runner (25 km/h, 1 ammo):")
print(f"Survival probability = {survival_prob:.4f}")
print(f"Predicted class (1 = Survive, 0 = Infected): {prediction}")


plt.figure()
plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Training Loss for Logistic Regression")
plt.grid(True)
plt.tight_layout()
plt.show()


plt.figure()


pos = (y.flatten() == 1)
neg = (y.flatten() == 0)

plt.scatter(X_raw[pos, 0], X_raw[pos, 1], label="Survived (1)")
plt.scatter(X_raw[neg, 0], X_raw[neg, 1], label="Infected (0)")


theta0, theta1, theta2 = theta.flatten()

speed_vals = np.linspace(X_raw[:,0].min(), X_raw[:,0].max(), 100)
ammo_vals = []

for s in speed_vals:
    z1 = (s - mu[0]) / sigma[0]
    if abs(theta2) < 1e-8:
        ammo_vals.append(np.nan)
    else:
        x2 = mu[1] - (sigma[1]/theta2) * (theta0 + theta1*z1)
        ammo_vals.append(x2)

plt.plot(speed_vals, ammo_vals, 'r-', label="Decision Boundary")

plt.xlabel("Sprint Speed (km/h)")
plt.ylabel("Ammo Clips")
plt.title("Logistic Regression: Decision Boundary")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
