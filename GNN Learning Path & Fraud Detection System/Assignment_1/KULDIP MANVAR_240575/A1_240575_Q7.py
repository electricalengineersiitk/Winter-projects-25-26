import numpy as np
import matplotlib.pyplot as plt

def run_z_day_algorithm():
    filename = 'zombies_data.csv'
    learning_rate = 0.01
    iterations = 5000

    data = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            clean_line = line.strip()
            if clean_line:
                str_values = clean_line.split(',')
                data.append([float(x) for x in str_values])

    data = np.array(data)
    X_raw = data[:, :2] # SprintSpeed, AmmoClips
    y = data[:, 2]      # Result

    mean_X = np.mean(X_raw, axis=0)
    std_X = np.std(X_raw, axis=0)
    X_norm = (X_raw - mean_X) / std_X

    m = len(y)
    X = np.hstack((np.ones((m, 1)), X_norm))

    # logistic regression
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def compute_cost(X, y, theta):
        m = len(y)
        h = sigmoid(np.dot(X, theta))
        epsilon = 1e-15 
        return -(1/m) * np.sum(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))

    # gradient descent
    theta = np.zeros(X.shape[1])
    cost_history = []
    
    for i in range(iterations):
        h = sigmoid(np.dot(X, theta))
        gradient = (1/m) * np.dot(X.T, (h - y))
        theta -= learning_rate * gradient
        
        if i % 100 == 0:
            cost_history.append(compute_cost(X, y, theta))

    print(f"Final Weights: {theta}")

    #Test Prediction
    test_features = np.array([25, 1]) 
    test_norm = (test_features - mean_X) / std_X
    test_input = np.concatenate(([1], test_norm))
    
    prob = sigmoid(np.dot(test_input, theta))
    prediction = 1 if prob >= 0.5 else 0
    
    print(f"TEST PREDICTION (25 km/h, 1 Ammo): Probability {prob:.4f}, Class {prediction}")

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(cost_history)
    plt.title("Cost Reduction")
    plt.xlabel("Iterations (x100)")
    plt.ylabel("Cost")

    plt.subplot(1, 2, 2)
    pos = X_raw[y == 1]
    neg = X_raw[y == 0]
    plt.scatter(pos[:, 0], pos[:, 1], c='green', marker='o', label='Survive')
    plt.scatter(neg[:, 0], neg[:, 1], c='red', marker='x', label='Infected')


    x_min, x_max = X_raw[:, 0].min(), X_raw[:, 0].max()
    x_plot = np.linspace(x_min, x_max, 100)
    x_plot_norm = (x_plot - mean_X[0]) / std_X[0]

    y_plot_norm = - (theta[0] + theta[1] * x_plot_norm) / theta[2]
    y_plot = (y_plot_norm * std_X[1]) + mean_X[1]

    plt.plot(x_plot, y_plot, 'b-', label='Decision Boundary')
    plt.xlabel('Sprint Speed')
    plt.ylabel('Ammo Clips')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_z_day_algorithm()