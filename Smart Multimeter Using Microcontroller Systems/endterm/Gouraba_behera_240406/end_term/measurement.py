import numpy as np

# RESISTANCE MEASUREMENT
def measure_resistance(true_R, R_ref, V_ref=5.0):
    V_adc = V_ref * true_R / (R_ref + true_R)
    V_adc = np.random.normal(V_adc, 0.003 * V_adc)
    V_adc = np.clip(V_adc, 1e-6, V_ref - 1e-6)

    measured_R = R_ref * V_adc / (V_ref - V_adc)
    error = abs(measured_R - true_R) / true_R * 100

    return measured_R, error
# CAPACITANCE MEASUREMENT
def measure_capacitance(true_C, R_ref=10000.0):
    tau = true_C * R_ref
    tau = np.random.normal(tau, 0.003 * tau)
    tau = max(tau, 1e-15)

    measured_C = tau / R_ref
    error = abs(measured_C - true_C) / true_C * 100

    return measured_C, error


# INDUCTANCE MEASUREMENT
def measure_inductance(true_L, C_ref=1e-9):
    f = 1 / (2 * np.pi * np.sqrt(true_L * C_ref))
    f = np.random.normal(f, 0.003* f)
    f = max(f, 1e-6)

    measured_L = 1 / ((2 * np.pi * f) ** 2 * C_ref)
    error = abs(measured_L - true_L) / true_L * 100

    return measured_L, error


# TEST BLOCK 
if __name__ == "__main__":
    print("=== TESTING MEASUREMENT ===\n")

    # Resistance test
    print("Resistance:")
    for r in [100, 1e3, 10e3, 100e3, 1e6]:
        m, e = measure_resistance(r,100000)
        print(f"True={r:.0f} Ω | Measured={m:.2f} Ω | Error={e:.2f}%")

    # Capacitance test
    print("\nCapacitance:")
    for c in [10e-9, 100e-9, 1e-6, 10e-6, 100e-6]:
        m, e = measure_capacitance(c)
        print(f"True={c*1e9:.1f} nF | Measured={m*1e9:.2f} nF | Error={e:.2f}%")

    # Inductance test
    print("\nInductance:")
    for l in [10e-6, 100e-6, 1e-3, 10e-3, 100e-3]:
        m, e = measure_inductance(l)
        print(f"True={l*1e6:.1f} µH | Measured={m*1e6:.2f} µH | Error={e:.2f}%")
