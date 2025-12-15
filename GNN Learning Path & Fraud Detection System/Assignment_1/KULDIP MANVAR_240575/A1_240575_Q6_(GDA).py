import matplotlib.pyplot as plt
import numpy as np

def parse_csv(filename):
    x_data = []
    y_data = []

    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                x_data.append(float(parts[0]))
                y_data.append(float(parts[1]))
                
    return x_data, y_data

def gradient_descent(x, y, learning_rate=0.01, epochs=1000):
    m = 0
    b = 0
    n = len(x)
    
    for _ in range(epochs):
        y_pred = m * x + b

        # dJ/dm = (1/n) * sum((y_pred - y) * x)
        dm = (1/n) * np.sum((y_pred - y) * x)
        
        # dJ/db = (1/n) * sum(y_pred - y)
        db = (1/n) * np.sum(y_pred - y)

        m = m - learning_rate * dm
        b = b - learning_rate * db
        
    return m, b

def main():
    x_list, y_list = parse_csv('housing_prices.csv')
    x = np.array(x_list)
    y = np.array(y_list)
    x_mean = np.mean(x)
    x_std = np.std(x)
    y_mean = np.mean(y)
    y_std = np.std(y)

    x_norm = (x - x_mean) / x_std
    y_norm = (y - y_mean) / y_std

    #Gradient Descent
    m_norm, b_norm = gradient_descent(x_norm, y_norm, learning_rate=0.01, epochs=2000)

    #Denormalize Coefficients to get actual m and b
    m = m_norm * (y_std / x_std)
    b = (b_norm * y_std) + y_mean - (m * x_mean)

    #Predict
    prediction_sqft = 2500
    predicted_price = m * prediction_sqft + b

    print(f"Gradient Descent Solution")
    print(f"Slope (m): {m:.4f}")
    print(f"Intercept (b): {b:.4f}")
    print(f"Prediction for 2,500 sqft: ${predicted_price:,.2f}")

    #Plotting
    plt.scatter(x, y, color='blue', label='Actual Data')
    
    line_x = np.linspace(min(x), max(x), 100)
    line_y = m * line_x + b
    
    plt.plot(line_x, line_y, color='green', linestyle='--', label='GD Best Fit')
    plt.xlabel('Square Footage')
    plt.ylabel('Price')
    plt.title('Housing Prices - Gradient Descent')
    plt.legend()
    plt.show()
if __name__ == "__main__":
    main()