import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

url =  'https://docs.google.com/spreadsheets/d/1au3HI9TCGs8dy_YIpGVPJizmI_-eeyXB0jdVcGCEXss/edit?usp=sharing'

df = pd.read_csv(url.replace('/edit?usp=sharing', '/export?format=csv'))

housing_price = df.values.tolist()

#print(housing_price)

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

def inv_2x2(A):
    result = np.zeros((2,2))
    det_A = A[0][0]*A[1][1] - A[1][0]*A[0][1]
    m = 1/det_A
    result[0][0] = m*(A[1][1])
    result[0][1] = -m*(A[0][1])
    result[1][0] = -m*(A[1][0])
    result[1][1] = m*(A[0][0])

    return result

X = np.zeros((len(housing_price),2))

for i in range(len(housing_price)):
    X[i][0] = housing_price[i][0]
    X[i][1] = 1

y = np.zeros((len(housing_price),1))

for i in range(len(housing_price)):
    y[i] = housing_price[i][1]

beta = np.zeros((2,1))

for i in range(100000):
    prediction = mat_mul(X,beta)
    error = prediction - y
    Gradient = mat_mul(transpose(X), error)/len(y)
    beta = beta - 1e-8 * Gradient

print('Beta:\n', beta)

x = 2500
p = beta[0]*x + beta[1]

plt.scatter(df['SquareFootage'], df['Price'])
plt.plot(x,p, 'ro',label='Predicted')
x_line = np.linspace(500,5000,1000)
p_line = beta[0]*x_line + beta[1]
plt.plot(x_line,p_line, color='orange', label='Best Fit Line')
plt.xlabel('Size (sq ft)')
plt.ylabel('Price (Rs)')
plt.title('Housing Price vs Size')
plt.legend()
plt.show()