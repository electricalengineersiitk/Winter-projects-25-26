import numpy as np
import matplotlib.pyplot as plt
import csv

data = []

with open("data.csv","r") as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        data.append([float(row[0]), float(row[1]), int(row[2])])

data = np.array(data)

X = data[:,0:2]
y = data[:,2]

mean = X.mean(axis=0)
std = X.std(axis=0)
X = (X - mean) / std

X = np.c_[np.ones(X.shape[0]), X]

def sigmoid(z):
    return 1/(1 + np.exp(-z))

theta = np.zeros(3)
alpha = 0.1
iterations = 8000

loss = []

for _ in range(iterations):
    z = X @ theta
    h = sigmoid(z)

    error = h - y
    grad = (1/len(y)) * (X.T @ error)

    theta -= alpha * grad

    cost = -(1/len(y)) * np.sum(y*np.log(h+1e-10) + (1-y)*np.log(1-h+1e-10))
    loss.append(cost)

test = np.array([25,1])
test = (test - mean) / std
test = np.array([1,test[0],test[1]])

p = sigmoid(test @ theta)

print("Survival probability:", round(p,4))
print("Prediction:", 1 if p>=0.5 else 0)

plt.figure()
plt.plot(loss)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()

plt.figure()

plt.scatter(X[y==1,1]*std[0]+mean[0], X[y==1,2]*std[1]+mean[1], label="Survive (1)")
plt.scatter(X[y==0,1]*std[0]+mean[0], X[y==0,2]*std[1]+mean[1], label="Infected (0)")

x_vals = np.linspace(min(data[:,0]),max(data[:,0]),100)

x_norm = (x_vals - mean[0]) / std[0]
y_norm = -(theta[0] + theta[1]*x_norm) / theta[2]
y_vals = y_norm * std[1] + mean[1]

plt.plot(x_vals, y_vals, color="black", label="Decision Boundary")

plt.xlabel("Sprint Speed")
plt.ylabel("Ammo Clips")
plt.legend()
plt.title("Logistic Regression Classifier")
plt.show()
