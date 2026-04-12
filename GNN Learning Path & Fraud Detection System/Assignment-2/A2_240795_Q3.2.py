import numpy as np

A = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

B = np.array([
    [0],
    [1],
    [1],
    [0]
])


W1 = np.random.uniform(size=(2, 2))
b1 = np.random.uniform(size=(1, 2))

W2 = np.random.uniform(size=(2, 1))
b2 = np.random.uniform(size=(1, 1))

L_R = 0.1   
epochs = 10000

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)
for i in range(epochs):
    z1 =A@W1+b1
    H1 = sigmoid(z1)

    z2 = H1@W2 +b2
    H2 = sigmoid(z2)

    diff2 = H2-B
   

    H1_transpose = np.transpose(H1)
    A_transpose = np.transpose(A)

    gradient2 = H1_transpose@(diff2*sigmoid_derivative(H2))
    error = (diff2*sigmoid_derivative(H2))@W2.T
    gradient1 = A_transpose@(error*sigmoid_derivative(H1))



    W1 -= L_R*gradient1
    b1 = b1 - L_R * np.sum(error*sigmoid_derivative(H1), axis=0, keepdims=True)


    W2 -= L_R*gradient2
    b2 = b2 - L_R * np.sum(diff2*sigmoid_derivative(H2), axis=0, keepdims=True)

print(f"final W1:{W1}")
print(f"final W2:{W2}")
finalz1 = A@W1+b1
finalH1 = sigmoid(finalz1)
finalZ2 = finalH1@W2 +b2
output = sigmoid(finalZ2)

print("\nFinal Predictions:")
print(output)
final_output = np.where(output>0.5,1,0)
print(final_output)