import csv
import numpy as np
import matplotlib.pyplot as plt

file = "zombies_data.csv"

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

speed = []
ammo = []
survive = []

with open(file, "r") as f:
    reader = csv.reader(f)
    next(reader)     
    for row in reader:
        speed.append(float(row[0]))
        ammo.append(float(row[1]))
        survive.append(float(row[2]))

speed = np.array(speed)
ammo = np.array(ammo)
survive = np.array(survive)

speed_mean = speed.mean()
speed_std = speed.std()
speed_n = (speed - speed_mean) / speed_std

ammo_mean = ammo.mean()
ammo_std = ammo.std()
ammo_n = (ammo - ammo_mean) / ammo_std

n = len(speed)
X = []
for i in range(n):
    X.append([speed_n[i], ammo_n[i], 1])   
X = np.array(X)
y = survive

theta = np.zeros(3)     

lr = 0.01
iterations = 3000
costs = []

for i in range(iterations):
    z = X.dot(theta)
    a = sigmoid(z)
    
    cost = -(1/n) * np.sum(y*np.log(a + 1e-12) + (1-y)*np.log(1 - a + 1e-12))
    costs.append(cost)

    grad = (1/n) * (X.T).dot(a - y)
    theta -= lr * grad

# 5. Test prediction

test_s = 25
test_a = 1

test_s_n = (test_s - speed_mean) / speed_std
test_a_n = (test_a - ammo_mean) / ammo_std

test_x = np.array([test_s_n, test_a_n, 1])

prob = sigmoid(test_x.dot(theta))
pred_class = 1 if prob >= 0.5 else 0

print("Test Survival Probability:", prob)
print("Predicted Class:", pred_class)

# Plot cost (loss) dropping

plt.figure(figsize=(7,5))
plt.plot(costs)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Cost Drop During Gradient Descent")
plt.grid(True)
plt.show()

# Decision boundary plot

plt.figure(figsize=(7,7))
plt.scatter(speed[y==0], ammo[y==0], marker='x', label="Infected (0)")
plt.scatter(speed[y==1], ammo[y==1], marker='o', label="Survive (1)")

s_vals = np.linspace(speed.min()-5, speed.max()+5, 200)
a_vals = np.linspace(ammo.min()-1, ammo.max()+1, 200)

S, A = np.meshgrid(s_vals, a_vals)

S_n = (S - speed_mean) / speed_std
A_n = (A - ammo_mean) / ammo_std

grid_X = np.column_stack((S_n.flatten(), A_n.flatten(), np.ones(S.size)))
probs = sigmoid(grid_X.dot(theta)).reshape(S.shape)

cs = plt.contour(S, A, probs, levels=[0.5], colors='red', linewidths=2)
plt.clabel(cs, fmt={0.5: "Decision Boundary"}, inline=True)

plt.xlabel("Speed (km/h)")
plt.ylabel("Ammo Clips")
plt.title("Logistic Regression Decision Boundary")
plt.legend()
plt.grid(True)
plt.show()
