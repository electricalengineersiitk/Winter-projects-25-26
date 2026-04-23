# import numpy as np

# def sigmoid(z): return 1/(1+np.exp(-z))
# def sp(a): return a*(1-a)

# W1=np.array([[0.5,0.3],[-0.2,0.4],[0.1,-0.3]])
# b1=np.array([[0.1],[-0.2],[0.3]])
# W2=np.array([[0.6,-0.4,0.2]])
# b2=np.array([[0.1]])
# eta=0.5

# def step(x,y,W1,b1,W2,b2):
#     z1=W1@x+b1; a1=sigmoid(z1)
#     z2=W2@a1+b2; a2=sigmoid(z2)
#     C=0.5*(y-a2)**2
#     d2=(a2-y)*sp(a2)
#     dW2=d2*a1.T
#     db2=d2
#     d1=(W2.T*d2)*sp(a1)
#     dW1=d1@x.T
#     db1=d1
#     W2n=W2-eta*dW2
#     b2n=b2-eta*db2
#     W1n=W1-eta*dW1
#     b1n=b1-eta*db1
#     return W1n,b1n,W2n,b2n,C,a2,z1,a1,z2

# # Iterations
# data=[(np.array([[0.0],[1.0]]),1.0),
#       (np.array([[1.0],[0.0]]),1.0),
#       (np.array([[1.0],[1.0]]),0.0)]
# out=[]
# for x,y in data:
#     res=step(x,y,W1,b1,W2,b2)
#     W1,b1,W2,b2=res[:4]
#     out.append(res)

# out







# # problem 6 for Assignment 1
# import matplotlib.pyplot as plt

# # GIVEN DATA
# sqft = [1100, 1400, 1425, 1550, 1600, 1700, 1750, 1800, 1875, 2000, 2100, 2250, 2300, 2400, 2450, 2600, 2800, 2900, 3000, 3150, 3300]

# price = [199000, 245000, 230000, 215000, 280000, 295000, 345000, 315000, 325000, 360000, 350000, 385000, 390000, 425000, 415000, 455000,465000, 495000, 510000, 545000, 570000]

# # TOTAL NUMBER OF POINTS
# n = len(sqft)

# # MEAN OF X AND Y
# # mean_x = (Σx) / n
# # mean_y = (Σy) / n
# mean_x = sum(sqft) / n
# mean_y = sum(price) / n

# # SLOPE (m) USING OLS FORMULA
# # m = Σ(x - mean_x)(y - mean_y) / Σ(x - mean_x)²
# num = 0   # numerator
# den = 0   # denominator

# for i in range(n):
#     num += (sqft[i] - mean_x) * (price[i] - mean_y)
#     den += (sqft[i] - mean_x) ** 2

# m = num / den
# # INTERCEPT (b) USING OLS FORMULA
# # b = mean_y - m * mean_x
# b = mean_y - m * mean_x

# # PREDICTION FOR 2500 SQ FT
# # y = m*x + b
# x_new = 2500
# y_new = m * x_new + b

# print("Mean of Square Feet:", mean_x)
# print("Mean of Price:", mean_y)
# print("Slope (m):", m)
# print("Intercept (b):", b)
# print("Predicted Price for 2500 sqft = ₹", format(y_new, ".2f"))

# # DRAW BEST FIT LINE
# line_y = []
# for x in sqft:
#     line_y.append(m * x + b)

# plt.scatter(sqft, price)
# plt.plot(sqft, line_y)
# plt.xlabel("Square Feet")
# plt.ylabel("House Price")
# plt.title("OLS Best Fit Line (Manual Calculation)")
# plt.show()


import matplotlib.pyplot as plt

# GIVEN DATA
sqft = [1100, 1400, 1425, 1550, 1600, 1700, 1750, 1800, 1875, 2000, 2100, 2250, 2300, 2400, 2450, 2600, 2800, 2900, 3000, 3150, 3300]

price = [199000, 245000, 230000, 215000, 280000, 295000, 345000, 315000, 325000, 360000, 350000, 385000, 390000, 425000, 415000, 455000, 465000, 495000, 510000, 545000, 570000]

n = len(sqft)
#  INITIAL VALUES
m = 0      # initial slope
b = 0      # initial intercept
alpha = 0.00000001   # learning rate
epochs = 2000        # iterations

loss_history = []

# GRADIENT DESCENT
for _ in range(epochs):
    dm = 0
    db = 0

    for i in range(n):
        y_pred = m * sqft[i] + b      # prediction
        error = y_pred - price[i]    # error

        dm += sqft[i] * error
        db += error

    dm = (2/n) * dm
    db = (2/n) * db

    m = m - alpha * dm
    b = b - alpha * db

    # loss calculation
    loss = 0
    for i in range(n):
        loss += (m * sqft[i] + b - price[i])**2
    loss = loss / n
    loss_history.append(loss)

# PREDICTION FOR 2500 SQFT
x_new = 2500
y_new = m * x_new + b

print("Final Slope (m):", m)
print("Final Intercept (b):", b)
print("Predicted Price for 2500 sqft =", format(y_new, ".2f"))

# BEST FIT LINE
line_y = []
for x in sqft:
    line_y.append(m * x + b)

plt.scatter(sqft, price)
plt.plot(sqft, line_y)
plt.xlabel("Square Feet")
plt.ylabel("House Price")
plt.title("Linear Regression Using Gradient Descent")
plt.show()

# LOSS GRAPH
plt.plot(loss_history)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Loss Reduction Using Gradient Descent")
plt.show()
