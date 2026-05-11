import numpy as np
import matplotlib.pyplot as plt
import os

# Import our custom files
from measurement import measure_resistance, measure_capacitance, measure_inductance
from autorange import AutoRanger
from protocol import make_packet

def run_simulation():
    # 1. Create results folder
    if not os.path.exists('results'):
        os.makedirs('results')

    configs = [
        {'name': 'Resistance', 'func': measure_resistance, 'key': 'R', 'unit': 'Ohms', 'limits': (10, 1000000)},
        {'name': 'Capacitance', 'func': measure_capacitance, 'key': 'C', 'unit': 'Farads', 'limits': (1e-9, 100e-6)},
        {'name': 'Inductance', 'func': measure_inductance, 'key': 'L', 'unit': 'Henries', 'limits': (1e-6, 100e-3)}
    ]

    for config in configs:
        print(f"\n" + "="*110)
        print(f" STARTING FULL 50-SAMPLE TEST: {config['name'].upper()}")
        print("="*110)
        
        # header for the data table
        print(f"{'Index':<6} | {'True Value':<12} | {'Measured':<12} | {'Range':<6} | {'Error %':<8} | {'OTG JSON Packet'}")
        print("-" * 110)

        ranger = AutoRanger(mode=config['key'])
        # generate 50 points on a log scale
        test_values = np.geomspace(config['limits'][0], config['limits'][1], 50)
        
        errors = []
        ranges = []

        for i, val in enumerate(test_values):
            # get the simulated reading
            measured, err = config['func'](val)
            
            # handle the auto-ranging logic
            active_range = ranger.determine_range(measured)
            
            # create the json packet for the app
            packet = make_packet(measured, config['unit'], active_range)
            
            # print row to console
            print(f"{i+1:<6} | {val:<12.1e} | {measured:<12.1e} | {active_range:<6} | {err:<8.2f}% | {packet}")
            
            errors.append(err)
            ranges.append(active_range)

        # Plot 1: Accuracy check
        plt.figure(figsize=(10, 5))
        plt.plot(test_values, errors, color='blue', marker='o')
        plt.xscale('log')
        plt.title(f"{config['name']} Accuracy Plot")
        plt.grid(True)
        plt.savefig(f"results/plot_accuracy_{config['key']}.png")
        plt.close()

        # Plot 2: Range stepping check
        plt.figure(figsize=(10, 5))
        plt.step(range(1, 51), ranges, where='post', color='green')
        plt.title(f"{config['name']} Auto-Range Steps")
        plt.ylim(0, 6)
        plt.grid(True)
        plt.savefig(f"results/plot_autorange_{config['key']}.png")
        plt.close()

        avg_err = sum(errors) / len(errors)
        print("-" * 110)
        print(f"FINISHED {config['name'].upper()} | AVG ERROR: {avg_err:.2f}%")

    print("\nSUCCESS: All samples done. 6 Plots saved in results folder.")

if __name__ == "__main__":
    run_simulation()