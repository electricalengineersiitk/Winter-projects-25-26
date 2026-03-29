import csv
import math
import matplotlib.pyplot as plt

file = "housing_prices.csv"

x = []
y = []

with open(file, "r") as f:
    reader = csv.reader(f)
    next(reader)   
    for row in reader:
        x.append(float(row[0]))
        y.append(float(row[1]))

n = len(x)

w0 = 0.0   
w1 = 0.0   

lr0 = 1e-6   
lr1 = 1e-10  
iters = 20000

for it in range(1, iters + 1):

    preds = [w0 + w1 * x[i] for i in range(n)]
    resid = [preds[i] - y[i] for i in range(n)]   

    grad0 = (2/n) * sum(resid[i] for i in range(n))    
    grad1 = (2/n) * sum(resid[i] * x[i] for i in range(n))  

    w0 -= lr0 * grad0
    w1 -= lr1 * grad1

pred = w1 * 2500 + w0
print(f"{pred:.2f}")

plt.scatter(x, y, label="Data")
x1 = min(x)
x2 = max(x)
y1 = w1 * x1 + w0
y2 = w1 * x2 + w0
plt.plot([x1, x2], [y1, y2], "", label="", linewidth=2)
plt.xlabel("Square Footage")
plt.ylabel("Price")
plt.title("Gradient Descent")
plt.grid()
plt.show()
