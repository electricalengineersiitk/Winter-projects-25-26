import os
import numpy as np
import matplotlib.pyplot as plt
from measurement import measure_resistance, measure_capacitance, measure_inductance
from autorange import AutoRanger
np.random.seed(42)
def run_simulation():
    # 50 values spread across a 10^5 range (10 to 1M Ohms)
    true_values = np.logspace(1, 6, 50) 
    auto_ranger = AutoRanger()
    
    auto_errors = []
    fixed_errors = []
    ranges = []

    # Map the active range (1-5) to its actual hardware Reference Resistor
    range_r_refs = {1: 100.0, 2: 1000.0, 3: 10000.0, 4: 100000.0, 5: 1000000.0}

    for val in true_values:
        # We must take multiple readings per test value so the engine can settle
        for _ in range(4):
            # 1. reference resistor 
            current_r_ref = range_r_refs[auto_ranger.current_range]
            
            # measurement
            measured_val, actual_error = measure_resistance(val, r_ref=current_r_ref)
            
            # the auto-ranger 
            settled_val, current_range = auto_ranger.process_reading(measured_val)
            
        # true error
        # massive errors
        _, fixed_error_real = measure_resistance(val, r_ref=1000.0)
        
        auto_errors.append(actual_error)
        fixed_errors.append(fixed_error_real)
        ranges.append(current_range)

    # Results Table
    avg_auto = np.mean(auto_errors)
    avg_fixed = np.mean(fixed_errors)
    print("\nMethod\t\t\tAverage Error")
    print("-" * 40)
    print(f"Fixed-range (no auto)\t{avg_fixed:.2f}%")
    print(f"Your auto-ranging sim\t{avg_auto:.2f}%")
    # C and L errors 
    c_errors = []
    # Capacitance sweep: 10 nF (1e-8) to 100 uF (1e-4)
    for val in np.logspace(-8, -4, 50): 
        _, err_c = measure_capacitance(val)
        c_errors.append(err_c)

    l_errors = []
    # Inductance sweep: 10 uH (1e-5) to 100 mH (1e-1)
    for val in np.logspace(-5, -1, 50): 
        _, err_l = measure_inductance(val)
        l_errors.append(err_l)

    avg_c = np.mean(c_errors)
    avg_l = np.mean(l_errors)
    
    print("-" * 40)
    print(f"Capacitance sim \t{avg_c:.2f}%")
    print(f"Inductance sim \t\t{avg_l:.2f}%")
    print("-" * 40)
    # Plots
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(true_values, auto_errors, label='Auto-Ranging Sim', color='blue')
    plt.plot(true_values, fixed_errors, label='Fixed Baseline', color='red', linestyle='--')
    plt.xscale('log')
    plt.xlabel('True Component Value (Ohms)')
    plt.ylabel('% Measurement Error')
    plt.title('Accuracy vs Input Value')
    plt.legend()
    plt.savefig(os.path.join(results_dir, "plot_accuracy.png"))
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.step(range(1, 51), ranges, where='post', color='green', linewidth=2)
    plt.xlabel('Test Sample Index (1 to 50)')
    plt.ylabel('Active Range (1 to 5)')
    plt.title('Auto-Range State Over Time')
    plt.yticks([1, 2, 3, 4, 5])
    plt.grid(True, axis='y')
    plt.savefig(os.path.join(results_dir, "plot_autorange.png"))
    plt.close()

if __name__ == "__main__":
    run_simulation()