# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt


# 1. DATA 

data = np.array([
    [12, 0, 0],   [14.5, 1, 0], [10, 2, 0],   [18, 0, 0], 
    [8.5, 4, 0],  [15, 1, 0],   [22, 0, 1],   [11, 5, 1], 
    [13, 2, 0],   [20.5, 1, 1], [24, 2, 1],   [16, 3, 1], 
    [12.5, 0, 0], [28, 0, 1],   [9, 6, 1],    [25, 1, 1], 
    [14, 4, 1],   [19, 2, 1],   [10.5, 2, 0], [26.5, 2, 1], 
    [15.5, 5, 1], [17, 3, 1]
])

X = data[:, :2] # Features: Speed, Ammo
y = data[:, 2]  # Labels: 0 or 1


# 2. NORMALIZATION 

X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)

X_norm = (X - X_mean) / X_std


# 3. LOGISTIC REGRESSION FUNCTIONS
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost_function(X, y, w, b):
    m = len(y)
    z = np.dot(X, w) + b
    h = sigmoid(z)
    # Epsilon to prevent log(0) errors
    epsilon = 1e-15
    cost = (-1/m) * np.sum(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))
    return cost

def gradient_descent(X, y, w, b, learning_rate, epochs):
    m = len(y)
    cost_history = []
    
    for i in range(epochs):
        # 1. Forward pass (Prediction)
        z = np.dot(X, w) + b
        h = sigmoid(z)
        
        # 2. Calculate Gradients
        dw = (1/m) * np.dot(X.T, (h - y))
        db = (1/m) * np.sum(h - y)
        
        # 3. Update Weights
        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        # 4. Record Cost
        cost_history.append(cost_function(X, y, w, b))
        
    return w, b, cost_history


# 4. TRAIN THE MODEL


w_init = np.zeros(X_norm.shape[1])
b_init = 0


learning_rate = 0.1
epochs = 2000

w_final, b_final, costs = gradient_descent(X_norm, y, w_init, b_init, learning_rate, epochs)

print(f"Training Complete.")
print(f"Weights: {w_final}")
print(f"Bias: {b_final}")
print("-" * 30)


# 5. PREDICTION (Task 1)

runner_speed = 25
runner_ammo = 1
runner_features = np.array([runner_speed, runner_ammo])

runner_norm = (runner_features - X_mean) / X_std

z_runner = np.dot(runner_norm, w_final) + b_final
prob_runner = sigmoid(z_runner)

print(f"Prediction for Runner (25 km/h, 1 Ammo):")
print(f"Survival Probability: {prob_runner:.4f}")
print(f"Class: {'SURVIVE' if prob_runner >= 0.5 else 'INFECTED'}")
print("-" * 30)


# 6. VISUALIZATION 

plt.figure(figsize=(12, 5))


plt.subplot(1, 2, 1)
plt.plot(costs)
plt.title("Cost (Loss) over Iterations")
plt.xlabel("Epochs")
plt.ylabel("Cost")


plt.subplot(1, 2, 2)


mask_survived = y == 1
mask_infected = y == 0

plt.scatter(X_norm[mask_survived, 0], X_norm[mask_survived, 1], c='green', label='Survive')
plt.scatter(X_norm[mask_infected, 0], X_norm[mask_infected, 1], c='red', label='Infected')


# The boundary is where z = 0  =>  w1*x1 + w2*x2 + b = 0
# x2 = -(w1*x1 + b) / w2
x1_boundary = np.linspace(np.min(X_norm[:, 0]), np.max(X_norm[:, 0]), 100)
x2_boundary = -(w_final[0] * x1_boundary + b_final) / w_final[1]

plt.plot(x1_boundary, x2_boundary, color='blue', label='Decision Boundary')

plt.title("Decision Boundary (Normalized Data)")
plt.xlabel("Normalized Speed")
plt.ylabel("Normalized Ammo")
plt.legend()

plt.show()