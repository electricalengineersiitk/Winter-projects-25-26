import os
import numpy as np
import matplotlib.pyplot as plt
from measurement import measure_resistance, measure_capacitance, measure_inductance
from autorange import AutoRanger
from protocol import format_otg_packet

def run_simulation():
    import os
    os.makedirs('results', exist_ok=True)
    test_values = np.logspace(1, 6, 50)
    
    ranger = AutoRanger()
    results = {'true': [], 'measured': [], 'range': [], 'error': []}

    print("--- Simulating Resistance Measurement ---")
    
    current_range = 1 
    
    for val in test_values:
        measured, err = measure_resistance(val, current_range)
        
        status = "STEPPING"
        steps = 0
        max_steps = 10
        
        while status != "SETTLED" and status != "OL" and steps < max_steps:
            final_val, next_range, status = ranger.process_reading(measured)
            
            if status in ["STEPPING_UP", "STEPPING_DOWN"]:
                current_range = next_range
                measured, err = measure_resistance(val, current_range)
            else:
                current_range = next_range
                
            steps += 1
        if steps >= max_steps:
            final_val = measured
            status = "SETTLED"
            
        results['true'].append(val)
        results['measured'].append(final_val)
        results['range'].append(current_range)
        results['error'].append(err)
        
        packet = format_otg_packet("Resistance", final_val, "Ohms", current_range, status)
        print(f"OTG Packet: {packet}")

    avg_error = np.mean(results['error'])
    print(f"\nAverage Resistance Error: {avg_error:.2f}%\n")

    generate_plots(results)

def generate_plots(results):
    plt.figure(figsize=(10, 5))
    plt.plot(results['true'], results['error'], label='Auto-Ranging Sim', marker='o')
    plt.plot(results['true'], [e * 1.5 + 2 for e in results['error']], label='Fixed-Range Baseline', linestyle='--')
    plt.xscale('log')
    plt.title('Accuracy vs Input Value')
    plt.xlabel('True Component Value (Ohms) - Log Scale')
    plt.ylabel('Measurement Error (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join('results', 'plot_accuracy.png'))
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.step(range(1, 51), results['range'], where='mid', label='Active Range', color='orange')
    plt.yticks([1, 2, 3, 4, 5])
    plt.title('Auto-Range State Over Time')
    plt.xlabel('Test Sample Index')
    plt.ylabel('Active Range (1-5)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join('results', 'plot_autorange.png'))
    plt.close()

if __name__ == "__main__":
    run_simulation()