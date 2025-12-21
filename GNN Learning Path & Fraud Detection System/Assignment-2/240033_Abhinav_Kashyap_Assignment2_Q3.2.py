import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivative(x):
    return x * (1 - x)

X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

w1 = np.random.uniform(size=(2, 2))
b1 = np.random.uniform(size=(1, 2))
w2 = np.random.uniform(size=(2, 1))
b2 = np.random.uniform(size=(1, 1))
lr = 0.5

for i in range(10000):
    layer1 = np.dot(X, w1) + b1
    act1 = sigmoid(layer1)
    
    layer2 = np.dot(act1, w2) + b2
    act2 = sigmoid(layer2)

    err = y - act2
    d2 = err * derivative(act2)
    
    err_h = d2.dot(w2.T)
    d1 = err_h * derivative(act1)

    w2 += act1.T.dot(d2) * lr
    b2 += np.sum(d2, axis=0, keepdims=True) * lr
    w1 += X.T.dot(d1) * lr
    b1 += np.sum(d1, axis=0, keepdims=True) * lr

print(act2)