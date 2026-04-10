"""
measurement.py — R, C, L formulas with Gaussian noise model
Smart Multimeter Simulation
"""

import numpy as np

# Constants
V_REF = 5.0       # Reference voltage (V)
V_SUPPLY = 5.0    # Supply voltage (V)
NOISE_STD = 0.005 # 0.5% of true value (σ)


def _select_r_ref(true_r_ohms: float) -> float:
    """Select the best reference resistor for the given resistance range."""
    if true_r_ohms <= 200:
        return 100.0
    elif true_r_ohms <= 2000:
        return 1000.0
    elif true_r_ohms <= 20000:
        return 10000.0
    elif true_r_ohms <= 200000:
        return 100000.0
    else:
        return 1000000.0


def measure_resistance(true_r_ohms: float, r_ref: float = None) -> tuple[float, float]:
    """
    Voltage divider method: R = R_ref × V_adc / (V_ref − V_adc)

    Args:
        true_r_ohms: True resistance value in ohms
        r_ref: Reference resistor in series (auto-selected if None)

    Returns:
        (measured_value, error_percent)
    """
    if r_ref is None:
        r_ref = _select_r_ref(true_r_ohms)
    # Ideal V_adc from voltage divider
    v_adc_ideal = V_REF * true_r_ohms / (r_ref + true_r_ohms)

    # Add Gaussian noise (0.5% of ideal V_adc)
    v_adc_noisy = np.random.normal(v_adc_ideal, NOISE_STD * v_adc_ideal)

    # Clamp to valid range
    v_adc_noisy = np.clip(v_adc_noisy, 1e-6, V_REF - 1e-6)

    # Recover R from noisy ADC reading
    measured_r = r_ref * v_adc_noisy / (V_REF - v_adc_noisy)

    error_pct = abs(measured_r - true_r_ohms) / true_r_ohms * 100.0
    return measured_r, error_pct


def measure_capacitance(true_c_farads: float, r_ref: float = 10000.0) -> tuple[float, float]:
    """
    RC charge-discharge method: C = t / R_ref
    where t = time for V to reach 63.2% of V_supply (one time constant τ)

    Args:
        true_c_farads: True capacitance in Farads
        r_ref: Charging resistor (default 10kΩ)

    Returns:
        (measured_value, error_percent)
    """
    # True time constant τ = R_ref × C
    true_tau = r_ref * true_c_farads

    # Add noise to the measured time constant
    noisy_tau = np.random.normal(true_tau, NOISE_STD * true_tau)

    # Derive C from time constant: C = τ / R_ref
    measured_c = noisy_tau / r_ref

    error_pct = abs(measured_c - true_c_farads) / true_c_farads * 100.0
    return measured_c, error_pct


def measure_inductance(true_l_henrys: float, c_ref: float = 1e-7) -> tuple[float, float]:
    """
    LC resonance method: L = 1 / ((2πf)² × C_ref)
    f = measured resonant frequency from LC tank circuit

    Args:
        true_l_henrys: True inductance in Henrys
        c_ref: Known reference capacitor (default 100 nF)

    Returns:
        (measured_value, error_percent)
    """
    # True resonant frequency: f = 1 / (2π√(LC))
    true_f = 1.0 / (2.0 * np.pi * np.sqrt(true_l_henrys * c_ref))

    # Add noise to the measured frequency
    noisy_f = np.random.normal(true_f, NOISE_STD * true_f)

    # Recover L from noisy frequency
    measured_l = 1.0 / ((2.0 * np.pi * noisy_f) ** 2 * c_ref)

    error_pct = abs(measured_l - true_l_henrys) / true_l_henrys * 100.0
    return measured_l, error_pct


if __name__ == "__main__":
    # Quick sanity check
    print("=== measurement.py sanity check ===\n")

    test_r = 4700.0
    r_val, r_err = measure_resistance(test_r)
    print(f"Resistance:   true={test_r:.1f} Ω  measured={r_val:.2f} Ω  error={r_err:.3f}%")

    test_c = 47e-9
    c_val, c_err = measure_capacitance(test_c)
    print(f"Capacitance:  true={test_c*1e9:.1f} nF  measured={c_val*1e9:.3f} nF  error={c_err:.3f}%")

    test_l = 1e-3
    l_val, l_err = measure_inductance(test_l)
    print(f"Inductance:   true={test_l*1e3:.1f} mH  measured={l_val*1e3:.3f} mH  error={l_err:.3f}%")
