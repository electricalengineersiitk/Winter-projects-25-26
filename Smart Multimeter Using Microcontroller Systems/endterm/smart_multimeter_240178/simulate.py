import numpy as np
import matplotlib.pyplot as plt
import os

from measurement import (
    measure_resistance,
    measure_capacitance,
    measure_inductance,
)
from autorange import AutoRange

NUM_SAMPLES = 50

# Ranges
R_ranges = [100, 1e3, 1e4, 1e5, 1e6]

def generate_values(min_val, max_val):
    return np.logspace(np.log10(min_val), np.log10(max_val), NUM_SAMPLES)

def run_simulation(measure_func, ranges, values):
    auto = AutoRange(ranges)

    records = []

    auto_errors = []
    fixed_errors = []
    active_ranges = []

    fixed_range_max = ranges[-1]  # highest range

    for val in values:
        measured, error = measure_func(val)

        # AUTO-RANGE
        r = auto.update(measured)

        # FIXED RANGE (simulate worse accuracy)
        fixed_measured = np.random.normal(val, 0.02 * val)  # 2% noise baseline
        fixed_error = abs(fixed_measured - val) / val * 100

        # STORE
        records.append((val, measured, r + 1, error))

        auto_errors.append(error)
        fixed_errors.append(fixed_error)
        active_ranges.append(r + 1)

    return records, auto_errors, fixed_errors, active_ranges

def print_table(records, title):
    print(f"\n--- {title} Results ---")
    print(f"{'True':>12} {'Measured':>12} {'Range':>8} {'Error %':>10}")

    for row in records[:10]:  # print first 10 rows only (clean output)
        print(f"{row[0]:12.4g} {row[1]:12.4g} {row[2]:8} {row[3]:10.4f}")

def plot_accuracy(values, auto_err, fixed_err):
    plt.figure()
    plt.xscale("log")
    plt.plot(values, auto_err, label="Auto-ranging")
    plt.plot(values, fixed_err, label="Fixed-range (baseline)")
    plt.xlabel("True Value")
    plt.ylabel("% Error")
    plt.title("Accuracy vs Input Value")
    plt.legend()
    plt.grid()

    os.makedirs("results", exist_ok=True)
    plt.savefig("results/plot_accuracy.png")
    plt.close()

def plot_autorange(active_ranges):
    plt.figure()
    plt.plot(active_ranges)
    plt.xlabel("Sample Index (1–50)")
    plt.ylabel("Active Range (1–5)")
    plt.title("Auto-Range State Over Time")
    plt.grid()

    os.makedirs("results", exist_ok=True)
    plt.savefig("results/plot_autorange.png")
    plt.close()

def main():
    # Use resistance as primary mode (as required)
    values = generate_values(100, 1e6)

    records, auto_err, fixed_err, active_ranges = run_simulation(
        measure_resistance, R_ranges, values
    )

    # PRINT TABLE
    print_table(records, "Resistance")

    # PRINT AVERAGE ERRORS
    print("\n--- Average Errors ---")
    print(f"Auto-ranging Error: {np.mean(auto_err):.3f}%")
    print(f"Fixed-range Error: {np.mean(fixed_err):.3f}%")

    # PLOTS (ONLY 2)
    plot_accuracy(values, auto_err, fixed_err)
    plot_autorange(active_ranges)

if __name__ == "__main__":
    main()