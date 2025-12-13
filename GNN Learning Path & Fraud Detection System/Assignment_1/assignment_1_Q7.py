import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

A = np.array([
    [12 ,0,1],
    [14.5 ,1,1],
    [10 ,2,1],
    [18 ,0,1],
    [8.5,4,1],
    [15 ,1,1],
    [22 ,0,1],
    [11 ,5,1],
    [13 ,2,1],
    [20.5,1,1],
    [24 ,2,1],
    [16 ,3,1],
    [12.5,0,1],
    [28 ,0,1],
    [9  ,6,1],
    [25 ,1,1],
    [14 ,4,1],
    [19 ,2,1],
    [10.5   ,2,1],
    [26.5   ,2,1],
    [15.5   ,5,1],
    [17 ,3,1]
])

X = np.array([
    [0.0],
    [0.0],
    [0.0],
])

B = np.array([
    [0],
    [0],
    [0],
    [0],
    [0],
    [0],
    [1],
    [1],
    [0],
    [1],
    [1],
    [1],
    [0],
    [1],
    [1],
    [1],
    [1],
    [1],
    [0],
    [1],
    [1],
    [1],
])

A_T = np.transpose(A)
L_R = 1e-2

costs = []

for i in range(1000000):
    h = A @ X
    H = 1/(1+np.exp(-h))
    diff = H-B
    
    cost = -np.mean(B * np.log(H + 1e-5) + (1 - B) * np.log(1 - H + 1e-5))
    costs.append(cost)
    
    gradient = 1/22 *(A_T @ diff)
    X = X - (L_R * gradient)

print("X is:")
print(X)

given = np.array([
    [25,1,1]
])

linear_output = given @ X
Required_answer = 1/(1+np.exp(-linear_output))

print(f"required answer:{Required_answer}")

answer_1 = np.where(Required_answer>0.5,1,0)
print(f"final answer : {answer_1}")

if answer_1[0][0] == 1:
    print("true")
else:
    print("false")

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(costs)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost dropping')

plt.subplot(1, 2, 2)
survivors = A[B.flatten() == 1]
infected = A[B.flatten() == 0]
plt.scatter(survivors[:, 0], survivors[:, 1], c='green', label='Survivor')
plt.scatter(infected[:, 0], infected[:, 1], c='red', label='Infected')

x_values = np.linspace(np.min(A[:, 0]), np.max(A[:, 0]), 100)
y_values = -(x_values * X[0][0] + X[2][0]) / X[1][0]

plt.plot(x_values, y_values, c='blue', label='Decision Boundary')
plt.xlabel('Sprint Speed')
plt.ylabel('Ammo Clips')
plt.legend()
plt.tight_layout()
plt.show()