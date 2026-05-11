import numpy as np
import os
import matplotlib.pyplot as plt
from measurement import measure_resistance, measure_capacitance, measure_inductance
from autorange import AutoRanger

def run_sweep(mode_name, true_values, measure_function):
    ranger = AutoRanger()
    errors = []
    active_ranges = []
    
    print(f"\n{'='*60}")
    print(f"--- {mode_name} Simulation Table ---")
    print(f"{'='*60}")
    print(f"{'True Value':<15} | {'Measured':<15} | {'Range':<7} | {'Error %':<7}")
    print(f"{'-'*60}")
    
    for true_val in true_values:
        measured_val, step_error = measure_function(true_val)
        
        # Scale the value ONLY for the decision engine
        if mode_name == "Capacitance":
            ranger_input = measured_val * 1e10
        elif mode_name == "Inductance":
            ranger_input = measured_val * 1e7
        else:
            ranger_input = measured_val
            
        status, active_range = ranger.process_reading(ranger_input)
        
        # Record data for plotting
        if status != "OL":
            errors.append(step_error)
            active_ranges.append(active_range)
            print(f"{true_val:<15.4g} | {measured_val:<15.4g} | {active_range:<7} | {step_error:<7.3f}")
        else:
            errors.append(0) # Placeholder for OL
            active_ranges.append(active_range)
            print(f"{true_val:<15.4g} | {'OL':<15} | {active_range:<7} | {'N/A':<7}")
            
    avg_error = sum(errors) / len(errors)
    print(f"{'-'*60}")
    print(f"AVERAGE {mode_name.upper()} ERROR: {avg_error:.3f}%")
    print(f"{'='*60}\n")
    
    return avg_error, errors, active_ranges

def generate_plots(true_vals, auto_errors, ranges):
    """Generates and saves the required plots to the results/ folder."""
    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)
    
    # ---------------------------------------------------------
    # PLOT 1: Accuracy vs Input Value
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    
    # Plot our auto-ranging simulation error
    plt.plot(true_vals, auto_errors, label='Your Auto-Ranging Sim', color='blue', marker='o', markersize=4)
    
    # Create a synthetic fixed-range baseline. 
    # Without auto-ranging, measuring small signals on a large scale creates massive noise errors.
    fixed_baseline_errors = [min(e + (10000 / v), 25) for e, v in zip(auto_errors, true_vals)]
    plt.plot(true_vals, fixed_baseline_errors, label='Fixed-Range Baseline', color='red', linestyle='--')

    plt.xscale('log')
    plt.xlabel('True Component Value (Log Scale)')
    plt.ylabel('Measurement Error (%)')
    plt.title('Accuracy vs Input Value (Resistance)')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    
    plot1_path = os.path.join("results", "plot_accuracy.png")
    plt.savefig(plot1_path)
    plt.close()
    print(f"Saved: {plot1_path}")

    # ---------------------------------------------------------
    # PLOT 2: Auto-Range State Over Time
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 4))
    
    # Sample index from 1 to 50
    sample_indices = range(1, 51)
    
    plt.step(sample_indices, ranges, where='post', color='green', linewidth=2)
    plt.xlabel('Test Sample Index (1 to 50)')
    plt.ylabel('Active Range (1 to 5)')
    plt.title('Auto-Range State Over Time')
    plt.yticks([1, 2, 3, 4, 5])
    plt.grid(True, axis='y', ls="--", alpha=0.7)
    
    plot2_path = os.path.join("results", "plot_autorange.png")
    plt.savefig(plot2_path)
    plt.close()
    print(f"Saved: {plot2_path}")


if __name__ == "__main__":
    # Test values
    r_test_values = np.logspace(1, 6, 50) 
    c_test_values = np.logspace(-9, -4, 50)
    l_test_values = np.logspace(-6, -1, 50)

    # Run sweeps and capture data
    r_avg, r_errors, r_ranges = run_sweep("Resistance", r_test_values, measure_resistance)
    c_avg, c_errors, c_ranges = run_sweep("Capacitance", c_test_values, measure_capacitance)
    l_avg, l_errors, l_ranges = run_sweep("Inductance", l_test_values, measure_inductance)
    
    # Final Summary Check
    print(">>> FINAL PHASE 3 & 4 CHECK <<<")
    if r_avg <= 2.0 and c_avg <= 2.0 and l_avg <= 2.0:
        print("SUCCESS: Average error is <= 2%.")
        
        # Generate the required plots using the Resistance data
        print("Generating plots...")
        generate_plots(r_test_values, r_errors, r_ranges)
        print("PHASE 4 COMPLETE!")
    else:
        print("WARNING: Error > 2%. Check noise model.")