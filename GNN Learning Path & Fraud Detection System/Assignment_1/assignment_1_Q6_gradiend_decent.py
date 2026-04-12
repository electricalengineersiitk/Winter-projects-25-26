import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

A = np.array([
    [1100,1],
[1400,1],
[1425,1],
[1550,1],
[1600,1],
[1700,1],
[1750,1],
[1800,1],
[1875,1],
[2000,1],
[2100,1],
[2250,1],
[2300,1],
[2400,1],
[2450,1],
[2600,1],
[2800,1],
[2900,1],
[3000,1],
[3150,1],
[3300,1]
])
A_T = np.transpose(A)
B = np.array([
    [199000],
[245000],
[230000],
[215000],
[280000],
[295000],
[345000],
[315000],
[325000],
[360000],
[350000],
[385000],
[390000],
[425000],
[415000],
[455000],
[465000],
[495000],
[510000],
[545000],
[570000]
])

X= np.array([
    [0],
    [0]
])

L_R = 1e-7 #learning rate
for i in range(1000000):
    h = A @ X
    H = h
    diff = H-B
    gradient = 1/21 *(A_T @ diff)
    X = X - L_R * gradient

print(f"slope:{X[0]}")
print(f"intersept:{X[1]}")

print(f"shape of the matrix : {X.shape}")
Y = A @ X


X_axis = A[:,0]

plt.scatter(X_axis, B,color='blue', label='Actual Data')

plt.plot(X_axis,Y,color='red', label='Best Fit Line')

plt.xlabel('SquareFootage') 
plt.ylabel('Prices')  
plt.show()