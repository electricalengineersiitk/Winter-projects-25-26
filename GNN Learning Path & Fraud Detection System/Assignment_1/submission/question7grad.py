# %%
import matplotlib.pyplot as plt

def get_data():
    """Returns the PropTech Solutions dataset."""
    return [
        [1100, 199000], [1400, 245000], [1425, 230000], [1550, 215000],
        [1600, 280000], [1700, 295000], [1750, 345000], [1800, 315000],
        [1875, 325000], [2000, 360000], [2100, 350000], [2250, 385000],
        [2300, 390000], [2400, 425000], [2450, 415000], [2600, 455000],
        [2800, 465000], [2900, 495000], [3000, 510000], [3150, 545000],
        [3300, 570000]
    ]

def run_gradient_descent(x_values, y_values, learning_rate=0.1, epochs=50000):
    """
    Performs Gradient Descent to find the line of best fit.
    Includes normalization to ensure convergence.
    """
    n = len(x_values)
    
    # 1. SCALING (Crucial for Gradient Descent)
    # We divide by the max value to get numbers between 0 and 1
    x_max = max(x_values)
    y_max = max(y_values)
    
    x_scaled = [x / x_max for x in x_values]
    y_scaled = [y / y_max for y in y_values]
    
    # 2. INITIALIZATION
    m = 0  # Start slope at 0
    b = 0  # Start intercept at 0
    
    # 3. TRAINING LOOP
    for _ in range(epochs):
        # Calculate Predictions
        y_pred = [(m * x + b) for x in x_scaled]
        
        # Calculate Gradients (Partial Derivatives)
        # d_m = (-2/n) * sum(x * (y - y_pred))
        d_m = (-2/n) * sum(x * (y - yp) for x, y, yp in zip(x_scaled, y_scaled, y_pred))
        
        # d_b = (-2/n) * sum(y - y_pred)
        d_b = (-2/n) * sum(y - yp for y, yp in zip(y_scaled, y_pred))
        
        # Update Weights
        m = m - (learning_rate * d_m)
        b = b - (learning_rate * d_b)
        
    # 4. UNSCALING (Convert back to real world units)
    # Real m = Scaled m * (y_max / x_max)
    real_m = m * (y_max / x_max)
    
    # Real b = Scaled b * y_max
    real_b = b * y_max
    
    return real_m, real_b

def save_plot(x_values, y_values, m, b):
    """Generates and saves the regression plot."""
    try:
        plt.figure(figsize=(10, 6))
        
        # Plot Scatter
        plt.scatter(x_values, y_values, color='#2980b9', label='Actual Sales')
        
        # Plot Line
        line_x = [min(x_values), max(x_values)]
        line_y = [(m * x) + b for x in line_x]
        plt.plot(line_x, line_y, color='#e74c3c', linewidth=2, label='Gradient Descent Fit')
        
        plt.title('PropTech Solutions: Gradient Descent Model')
        plt.xlabel('Square Footage')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        
        # Save to file
        plt.savefig('gradient_descent_result.png')
        print("Graph saved as 'gradient_descent_result.png'")
        
    except Exception as e:
        print(f"Plotting error: {e}")

def main():
    print("--- Gradient Descent Engine Starting ---")
    
    # Load Data
    data = get_data()
    x = [row[0] for row in data]
    y = [row[1] for row in data]
    
    # Train
    print("Training model (this may take a split second)...")
    m, b = run_gradient_descent(x, y)
    
    print(f"Training Complete.")
    print(f"Slope (m): {m:.4f}")
    print(f"Intercept (b): {b:.4f}")
    
    # Predict
    target = 2500
    prediction = m * target + b
    print(f"\nPrediction for {target} sq ft: ${prediction:,.2f}")
    
    # Plot
    save_plot(x, y, m, b)

if __name__ == "__main__":
    main()


