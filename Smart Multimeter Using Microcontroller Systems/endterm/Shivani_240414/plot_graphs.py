import matplotlib.pyplot as plt
from measurement import get_resistance

# test values
actual_values = [100, 1000, 10000, 100000, 1000000]

measured_values = []
errors = []

for val in actual_values:
    measured, error = get_resistance(val)
    measured_values.append(measured)
    errors.append(error)

# Graph 1
plt.figure()
plt.plot(actual_values, measured_values, marker='o')
plt.xlabel("Actual Resistance")
plt.ylabel("Measured Resistance")
plt.title("Actual vs Measured")
plt.grid()
plt.savefig("actual_vs_measured.png")

# Graph 2
plt.figure()
plt.plot(actual_values, errors, marker='o')
plt.xlabel("Actual Resistance")
plt.ylabel("Error (%)")
plt.title("Error vs Resistance")
plt.grid()
plt.savefig("error_vs_resistance.png")

plt.show()