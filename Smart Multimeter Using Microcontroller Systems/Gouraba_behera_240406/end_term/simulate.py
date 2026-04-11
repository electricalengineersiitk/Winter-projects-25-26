import numpy as np
import matplotlib.pyplot as plt
import os

from measurement import measure_resistance, measure_capacitance, measure_inductance
from autorange import AutoRanger
from protocol import create_packet, encode_packet, decode_packet


# CREATE RESULTS FOLDER
BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, "results")
os.makedirs(RESULTS, exist_ok=True)


# DEFINE RANGES (IMPORTANT)

ranges_R = [100, 1e3, 10e3, 100e3, 1e6]
ranges_C = [10e-9, 100e-9, 1e-6, 10e-6, 100e-6]
ranges_L = [10e-6, 100e-6, 1e-3, 10e-3, 100e-3]

# GENERATE VALUES

values_R = np.logspace(2, 6, 50)
values_C = np.logspace(-8, -4, 50)
values_L = np.logspace(-5, -1, 50)


# SIMULATION FUNCTION


def run(values, measure_func, ranger, name, use_range=False):
    errors = []
    ranges = []
    print(f"\n=== {name} ===")

    for v in values:
     
        state = ranger.update(v)
        current_max = state["max"]

        if use_range:
            measured, err = measure_func(v, current_max)
        else:
            measured, err = measure_func(v)

        pkt = create_packet(name[0], measured, err, state["range"])
        encoded = encode_packet(pkt)
        decoded = decode_packet(encoded)

        errors.append(err)
        ranges.append(state["range"])
        print(f"Value={v:.2e} | Error={err:.2f}% | Range={state['range']}")

    return errors, ranges

ranger_R = AutoRanger(ranges_R)
ranger_C = AutoRanger(ranges_C)
ranger_L = AutoRanger(ranges_L)

err_R, range_R = run(values_R, measure_resistance, ranger_R, "Resistance", use_range=True)
err_C, range_C = run(values_C, measure_capacitance, ranger_C, "Capacitance")
err_L, range_L = run(values_L, measure_inductance, ranger_L, "Inductance")


# PLOT 1
plt.figure()

plt.semilogx(values_R, err_R, label="R")
plt.semilogx(values_C, err_C, label="C")
plt.semilogx(values_L, err_L, label="L")

plt.axhline(2.0, linestyle="--", label="2%")

plt.xlabel("True Component Value")
plt.ylabel("Error (%)")
plt.legend()
plt.grid()
plt.title("Accuracy")

plt.savefig(os.path.join(RESULTS, "plot_accuracy.png"))

# PLOT 2
plt.figure()
plt.plot(np.array(range_R) + 0.05, marker='o', label="R")
plt.plot(np.array(range_C), marker='x', linestyle='--', label="C")
plt.plot(np.array(range_L) - 0.05, marker='s', linestyle=':', label="L")

plt.xlabel("Sample Index (1 to 50)")
plt.ylabel("Active Range (1 to 5)")
plt.legend()
plt.grid()
plt.title("Auto Range")

# Saving plots
plt.savefig(os.path.join(RESULTS, "plot_autorange.png"))


avg_R = sum(err_R) / len(err_R)
avg_C = sum(err_C) / len(err_C)
avg_L = sum(err_L) / len(err_L)

print("\n========== FINAL RESULTS ==========")
print(f"Total Error (Resistance): {avg_R:.3f}%")
print(f"Total Error (Capacitance): {avg_C:.3f}%")
print(f"Total Error (Inductance): {avg_L:.3f}%")
print("===================================")
