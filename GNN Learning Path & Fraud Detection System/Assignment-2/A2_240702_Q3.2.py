import numpy as np

X=np.array([[0,0],[0,1],[1,0],[1,1]])
y=np.array([[0],[1],[1],[0]])

np.random.seed(42)

W1=np.random.uniform(size=(2,2))
b1=np.random.uniform(size=(1,2))
W2=np.random.uniform(size=(2,1))
b2=np.random.uniform(size=(1,1))

lr=0.1
epochs=10000

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return x*(1-x)

for epoch in range(epochs):
    hidden_output=sigmoid(X.dot(W1)+b1)
    output=sigmoid(hidden_output.dot(W2)+b2)
    error=y-output
    delta_output=error*sigmoid_derivative(output)
    delta_hidden=delta_output.dot(W2.T)*sigmoid_derivative(hidden_output)
    W2+=hidden_output.T.dot(delta_output)*lr
    b2+=np.sum(delta_output,0,keepdims=1)*lr
    W1+=X.T.dot(delta_hidden)*lr
    b1+=np.sum(delta_hidden,0,keepdims=1)*lr
    if epoch%1000==0:
        loss=np.mean(np.square(error))
        print(f"Epoch {epoch}, Loss: {loss}")

for xi,yi in zip(X,y):
    hidden_output=sigmoid(xi.dot(W1)+b1)
    output=sigmoid(hidden_output.dot(W2)+b2)
    print(xi,output.round()[0],yi[0])