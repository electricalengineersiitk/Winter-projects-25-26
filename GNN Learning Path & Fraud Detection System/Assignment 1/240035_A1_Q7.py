import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

url = 'https://docs.google.com/spreadsheets/d/1n80MKN7eWNrhfpJA2L9McdWCfb6H18Pn7LA8uiZO0-U/edit?usp=sharing'

df = pd.read_csv(url.replace('/edit?usp=sharing', '/export?format=csv'))

data = df.values.tolist()

#print(data)

def mat_mul(A, B):
    result = np.zeros((len(A),len(B[0])))

    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]
    return result
       
def transpose(A):
    result = np.zeros((len(A[0]),len(A)))

    for i in range (len(A[0])):
        for j in range (len(A)):
            result[i][j] = A[j][i]

    return result

def sigmoid(A):
    result = 1/(1 + np.exp(-A))
    return result

X = np.zeros((len(data),3))
beta = np.zeros((len(data[0]),1))
y = np.zeros((len(data),1))

for i in range(len(data)):
    y[i] = data[i][2]

for i in range(len(data)):
    X[i][0] = data[i][0]
    X[i][1] = data[i][1]
    X[i][2] = 1

#Normalization of features
for j in range(len(X[0])-1):
    mean = np.mean(X[:,j])
    std = np.std(X[:,j])
    for i in range(len(X)):
        X[i][j] = (X[i][j] - mean)/std
               

# print(X)
# print(beta)
# print(y)

Loss = np.zeros(100000)

for epoch in range(100000):
    z = mat_mul(X,beta)
    h = sigmoid(z)
    loss = -np.mean(y * np.log(h) + (1 - y) * np.log(1 - h))
    Loss[epoch] = loss
    Gradient = mat_mul(transpose(X), (h - y))/len(y)
    beta = beta - 1e-2 * Gradient
    
h = sigmoid(mat_mul(X,beta))

for i in range(len(h)):
    if h[i]>=0.5:
        h[i]=1
    else:
        h[i]=0

error = y - h
accuracy = (1 - (np.count_nonzero(error)/len(y))) * 100
print(accuracy)

x = np.linspace(1,100000,100000)
# print(x.shape)
# print(Loss.shape)ok
plt.scatter(x,Loss,color ='green', label = 'Loss Function Visualizer')
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.title('Loss Function Visulaizer')
plt.show()
#print("Accuracy: {:.2f}%".format(accuracy))
#print("Predicted values:\n",h)
#print("Actual values:\n",y)

# Plotting the decision boundary

b0 = beta[0][0]      
b1 = beta[1][0]      
b_bias = beta[2][0]  

x_line = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
y_line = - (b0 / b1) * x_line - (b_bias / b1)

plt.figure(figsize=(8, 6))

mask_0 = (y.flatten() == 0)
mask_1 = (y.flatten() == 1)

plt.scatter(X[mask_0, 0], X[mask_0, 1], c='blue', label='Class 0')

plt.scatter(X[mask_1, 0], X[mask_1, 1], c='red', label='Class 1')

plt.plot(x_line, y_line, c='black', linewidth=2, label='Decision Boundary')

plt.xlabel('Normalized Feature 1')
plt.ylabel('Normalized Feature 2')
plt.title('Logistic Regression Decision Boundary')
plt.legend()
plt.show()