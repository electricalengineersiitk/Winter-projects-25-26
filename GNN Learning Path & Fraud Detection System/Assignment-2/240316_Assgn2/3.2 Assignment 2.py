import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])


y = np.array([[0],
              [1],
              [1],
              [0]])

np.random.seed(10)


input_layer_neurons = 2
hidden_layer_neurons = 2
output_layer_neurons = 1


wh = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons)) 
bh = np.random.uniform(size=(1, hidden_layer_neurons)) 


w_out = np.random.uniform(size=(hidden_layer_neurons, output_layer_neurons))
b_out = np.random.uniform(size=(1, output_layer_neurons))

learning_rate = 0.5
epochs = 20000

for i in range(epochs):

    hidden_layer_input = np.dot(X, wh) + bh
    hidden_layer_output = sigmoid(hidden_layer_input)
    
    output_layer_input = np.dot(hidden_layer_output, w_out) + b_out
    predicted_output = sigmoid(output_layer_input)
    
  
    error = y - predicted_output
    

    d_output = error * sigmoid_derivative(predicted_output)
    
    error_hidden_layer = d_output.dot(w_out.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)
    
    
    w_out += hidden_layer_output.T.dot(d_output) * learning_rate
    b_out += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    
    wh += X.T.dot(d_hidden_layer) * learning_rate
    bh += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate


print("Training complete.")
print("\n(x1) | (x2) | Predicted")
print("---- | ---- | ---------")
for i in range(len(X)):
    print(f" {X[i][0]}   |  {X[i][1]}   | {predicted_output[i][0]:.4f}")