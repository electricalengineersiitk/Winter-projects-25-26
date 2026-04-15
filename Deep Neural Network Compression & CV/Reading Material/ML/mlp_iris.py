import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

# ==============================
# 1. Data Loading & Preprocessing
# ==============================

# Load the Iris dataset
dataset = load_iris()

# Extract features (x) and targets (y)
x = dataset.data   # Shape: (150, 4) -> 150 samples, 4 features each (sepal len, width, etc.)
y = dataset.target # Shape: (150,) -> 150 labels (0, 1, or 2)

print("Input shape:", x.shape)
print("Target shape:", y.shape)

# Split data into training (80%) and testing (20%) sets
# random_state=42 ensures the split is the same every time we run the code
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Standardize the features (Mean = 0, Std Dev = 1)
# This helps the neural network converge faster and prevents values from exploding
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train) # Fit scaler on train data & transform
x_test = scaler.transform(x_test)       # Transform test data using train stats

# Convert numpy arrays to PyTorch Tensors
# Features need to be Float32 for matrix multiplication
# Targets need to be Long (int64) for CrossEntropyLoss
x_train = torch.tensor(x_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# Check the distribution of classes in the test set
print("Test set class distribution:\n", pd.Series(y_test).value_counts())


# ==============================
# 2. Neural Network Initialization
# ==============================

# Architecture:
# Input Layer: 4 neurons (matches feature count)
# Hidden Layer 1: 16 neurons
# Hidden Layer 2: 16 neurons
# Output Layer: 3 neurons (matches number of classes)

torch.manual_seed(42) # Set seed for reproducible random weights

# Initialize Weights and Biases manually
# requires_grad=True tells PyTorch to track operations on these tensors for backpropagation

# Layer 1 weights: Connects 4 inputs to 16 hidden neurons
w1 = torch.randn(4, 16, requires_grad=True)
b1 = torch.randn(16, requires_grad=True)

# Layer 2 weights: Connects 16 hidden neurons to 16 hidden neurons
w2 = torch.randn(16, 16, requires_grad=True)
b2 = torch.randn(16, requires_grad=True)

# Layer 3 weights: Connects 16 hidden neurons to 3 output classes
w3 = torch.randn(16, 3, requires_grad=True)
b3 = torch.randn(3, requires_grad=True)


# ==============================
# 3. Training Loop
# ==============================

learning_rate = 0.01

for epoch in range(200):
    
    # --- Forward Pass ---
    # Layer 1: Matrix multiplication (Input * W1) + Bias
    z1 = torch.matmul(x_train, w1) + b1
    a1 = torch.relu(z1) # Activation: ReLU (turns negative values to 0)
    
    # Layer 2: Matrix multiplication (Layer1_Output * W2) + Bias
    z2 = torch.matmul(a1, w2) + b2
    a2 = torch.relu(z2) # Activation: ReLU
    
    # Layer 3: Matrix multiplication (Layer2_Output * W3) + Bias
    z3 = torch.matmul(a2, w3) + b3
    
    # Apply Softmax to output layer to get probabilities summing to 1
    # dim=1 means apply across the columns (classes) for each row (sample)
    a3 = torch.softmax(z3, dim=1)  

    # --- Loss Calculation ---
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # Note: CrossEntropyLoss usually expects raw logits (z3), not softmax (a3).
    # However, since you are passing 'z3' into the loss function below, it is correct.
    output_loss = loss_fn(z3, y_train)

    # --- Backward Pass ---
    # Calculates gradients (dLoss/dw) for all tensors with requires_grad=True
    output_loss.backward()

    # --- Optimizer Step (Manual Gradient Descent) ---
    # We use 'with torch.no_grad()' because we don't want these update operations
    # to be added to the computation graph (we only track gradients during Forward Pass)
    with torch.no_grad():
        # Update weights: w_new = w_old - (learning_rate * gradient)
        w1 -= learning_rate * w1.grad
        b1 -= learning_rate * b1.grad
        w2 -= learning_rate * w2.grad
        b2 -= learning_rate * b2.grad
        w3 -= learning_rate * w3.grad
        b3 -= learning_rate * b3.grad
        
        # Reset gradients to zero after updating.
        # Otherwise, PyTorch accumulates (adds) gradients in the next loop.
        w1.grad.zero_()
        b1.grad.zero_()
        w2.grad.zero_()
        b2.grad.zero_()
        w3.grad.zero_()
        b3.grad.zero_()


# ==============================
# 4. Testing / Validation
# ==============================

# Perform a forward pass on the test data (unseen data)
z1 = torch.matmul(x_test, w1) + b1
a1 = torch.relu(z1)

z2 = torch.matmul(a1, w2) + b2
a2 = torch.relu(z2)

z3 = torch.matmul(a2, w3) + b3
a3 = torch.softmax(z3, dim=1) # Get final probabilities

# Get predictions:
# torch.max(a3, 1) returns (max_values, indices_of_max_values)
# We only care about indices (the predicted class ID: 0, 1, or 2)
_, predictions = torch.max(a3, 1)

# Calculate Accuracy:
# (predictions == y_test) creates a boolean tensor [True, False, True...]
# .sum() counts the Trues. .item() gets the number as a python float.
accuracy = (predictions == y_test).sum().item() / y_test.size(0)

# Calculate final loss on test set for reference
val_loss = loss_fn(z3, y_test)

print(f"accuracy is : {accuracy}, training loss is : {output_loss}, val loss is : {val_loss}")