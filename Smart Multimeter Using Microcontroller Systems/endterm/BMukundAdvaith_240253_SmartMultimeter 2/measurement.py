"""
measurement.py  —  R, C, L formulas + noise model
Smart Multimeter Simulation | End-Term Project

Primary focus: Resistance (voltage divider method)
Supporting:    Capacitance (RC time constant), Inductance (LC resonance)

Each function:
  - Takes a true component value
  - Adds Gaussian noise (sigma = 0.5% of true value) to simulate ADC imprecision
  - Returns (measured_value, error_%)
"""

import numpy as np

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

V_REF     = 5.0      # Reference voltage (volts)
R_REF_C   = 10_000   # Reference resistor for RC timing circuit (10 kOhm)
C_REF     = 1e-6     # Reference capacitor for LC resonance (1 uF)
NOISE_STD = 0.005    # 0.5% of true value — ADC imprecision model


def _add_noise(true_value):
    """
    Simulate ADC imprecision by adding Gaussian noise.
    sigma = 0.5% of the true value, mean = 0 (unbiased).
    """
    return np.random.normal(true_value, NOISE_STD * abs(true_value))


def _pick_r_ref(true_r):
    """
    Pick the best R_REF for the voltage divider based on the expected range.
    R_REF should be close to R_unknown for maximum ADC sensitivity.
    In real hardware this is done by switching in different reference resistors.
    """
    if true_r < 500:
        return 100
    elif true_r < 5_000:
        return 1_000
    elif true_r < 50_000:
        return 10_000
    elif true_r < 500_000:
        return 100_000
    else:
        return 1_000_000


# ─────────────────────────────────────────────
# 1. RESISTANCE  (PRIMARY MODE)
# ─────────────────────────────────────────────
# Method: Voltage divider
#   V_adc = V_ref * R_unknown / (R_ref + R_unknown)
#   Rearranged: R = R_ref * V_adc / (V_ref - V_adc)
# ─────────────────────────────────────────────

def measure_resistance(true_r):
    """
    Measure resistance using the voltage divider method.

    Args:
        true_r: True resistance in ohms

    Returns:
        (measured_r, error_percent)
    """
    r_ref = _pick_r_ref(true_r)

    # Ideal ADC voltage from voltage divider
    v_adc_ideal = V_REF * true_r / (r_ref + true_r)

    # Add noise to simulate ADC reading
    v_adc_noisy = _add_noise(v_adc_ideal)

    # Clamp to valid range to avoid division by zero
    v_adc_noisy = np.clip(v_adc_noisy, 1e-6, V_REF - 1e-6)

    # Derive resistance from noisy ADC voltage
    measured_r = r_ref * v_adc_noisy / (V_REF - v_adc_noisy)

    error_pct = abs(measured_r - true_r) / true_r * 100

    return measured_r, error_pct


# ─────────────────────────────────────────────
# 2. CAPACITANCE
# ─────────────────────────────────────────────
# Method: RC charge-discharge timing
#   Time for V to reach 63.2% of V_supply = tau = R_ref * C
#   So: C = tau / R_ref
# ─────────────────────────────────────────────

def measure_capacitance(true_c):
    """
    Measure capacitance using RC time constant method.

    Args:
        true_c: True capacitance in farads

    Returns:
        (measured_c, error_percent)
    """
    tau_true  = R_REF_C * true_c
    tau_noisy = _add_noise(tau_true)
    tau_noisy = max(tau_noisy, 1e-12)

    measured_c = tau_noisy / R_REF_C
    error_pct  = abs(measured_c - true_c) / true_c * 100

    return measured_c, error_pct


# ─────────────────────────────────────────────
# 3. INDUCTANCE
# ─────────────────────────────────────────────
# Method: LC resonance frequency
#   f = 1 / (2*pi * sqrt(L * C_ref))
#   Rearranged: L = 1 / ((2*pi*f)^2 * C_ref)
# ─────────────────────────────────────────────

def measure_inductance(true_l):
    """
    Measure inductance using LC resonance frequency method.

    Args:
        true_l: True inductance in henrys

    Returns:
        (measured_l, error_percent)
    """
    f_true  = 1 / (2 * np.pi * np.sqrt(true_l * C_REF))
    f_noisy = _add_noise(f_true)
    f_noisy = max(f_noisy, 1e-6)

    measured_l = 1 / ((2 * np.pi * f_noisy) ** 2 * C_REF)
    error_pct  = abs(measured_l - true_l) / true_l * 100

    return measured_l, error_pct


# ─────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────
'''
if __name__ == "__main__":
    print("=" * 55)
    print("  measurement.py - Self-Test")
    print("=" * 55)

    r_tests = [50, 500, 5_000, 50_000, 500_000]
    print("\n[ RESISTANCE ]")
    print(f"  {'True (Ohm)':<15} {'Measured (Ohm)':<18} {'Error %'}")
    print(f"  {'-'*48}")
    for r in r_tests:
        m, e = measure_resistance(r)
        print(f"  {r:<15,.0f} {m:<18,.2f} {e:.4f}%")

    c_tests = [5e-9, 50e-9, 500e-9, 5e-6, 50e-6]
    print("\n[ CAPACITANCE ]")
    print(f"  {'True (F)':<15} {'Measured (F)':<18} {'Error %'}")
    print(f"  {'-'*48}")
    for c in c_tests:
        m, e = measure_capacitance(c)
        print(f"  {c:<15.3e} {m:<18.3e} {e:.4f}%")

    l_tests = [5e-6, 50e-6, 500e-6, 5e-3, 50e-3]
    print("\n[ INDUCTANCE ]")
    print(f"  {'True (H)':<15} {'Measured (H)':<18} {'Error %'}")
    print(f"  {'-'*48}")
    for l in l_tests:
        m, e = measure_inductance(l)
        print(f"  {l:<15.3e} {m:<18.3e} {e:.4f}%")

    print("\n  All formulas returned sensible values.")
    print("=" * 55)
'''