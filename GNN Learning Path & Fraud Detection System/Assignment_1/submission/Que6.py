# %%
import matplotlib.pyplot as plt

def get_housing_data():
    """
    Returns the manually parsed dataset for PropTech Solutions.
    Format: [Square_Feet, Price]
    """
    return [
        [1100, 199000], [1400, 245000], [1425, 230000], [1550, 215000],
        [1600, 280000], [1700, 295000], [1750, 345000], [1800, 315000],
        [1875, 325000], [2000, 360000], [2100, 350000], [2250, 385000],
        [2300, 390000], [2400, 425000], [2450, 415000], [2600, 455000],
        [2800, 465000], [2900, 495000], [3000, 510000], [3150, 545000],
        [3300, 570000]
    ]
def calculate_ols_regression(x_values, y_values):
    """
     the Slope (m) and Y-Intercept (b) using the 
    Ordinary Least Squares (OLS) method.
    """
    n = len(x_values)
    
    sum_x = sum(x_values)
    sum_y = sum(y_values)
    
    # Calculate Sum of XY and Sum of X^2
    sum_xy = sum(x * y for x, y in zip(x_values, y_values))
    sum_x_squared = sum(x ** 2 for x in x_values)
    
    # Calculate Slope (m)
    numerator_m = (n * sum_xy) - (sum_x * sum_y)
    denominator_m = (n * sum_x_squared) - (sum_x ** 2)
    m = numerator_m / denominator_m
    
    # Calculate Intercept (b)
    b = (sum_y - (m * sum_x)) / n
    
    return m, b

def plot_regression_line(x_values, y_values, m, b):
    """
    Plots the actual data points vs the Line of Best Fit.
    """
    plt.figure(figsize=(10, 6))
    
    # Plot actual data
    plt.scatter(x_values, y_values, color='#2c3e50', label='Actual Sales Data')
    
    # Generate line of best fit points
    line_x = [min(x_values), max(x_values)]
    line_y = [(m * x) + b for x in line_x]
    
    # Plot regression line
    plt.plot(line_x, line_y, color='#e74c3c', linewidth=2, label='OLS Regression Line')
    
    # Formatting
    plt.title('PropTech Solutions: Housing Price Model', fontsize=14)
    plt.xlabel('Square Footage', fontsize=12)
    plt.ylabel('Price ($)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

def main():
    print("--- PropTech Solutions Price Engine ---")
    
    # 1. Load Data
    raw_data = get_housing_data()
    
    # Separate columns into lists
    sq_footage = [row[0] for row in raw_data]
    prices = [row[1] for row in raw_data]
    
    # 2. Train Model (Calculate OLS)
    slope, intercept = calculate_ols_regression(sq_footage, prices)
    
    print(f"Model Trained Successfully.")
    print(f"Slope (m): {slope:.4f}")
    print(f"Intercept (b): {intercept:.4f}")
    
    # 3. Predict for specific case
    target_sqft = 2500
    predicted_price = (slope * target_sqft) + intercept
    
    print(f"\nPrediction for {target_sqft} sq. ft:")
    print(f"${predicted_price:,.2f}")
    
    # 4. Visualize
    plot_regression_line(sq_footage, prices, slope, intercept)

if __name__ == "__main__":
    main()


