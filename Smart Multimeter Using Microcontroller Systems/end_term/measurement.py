"""
measurement.py — R, C, L measurement formulas + Gaussian noise model
Smart Multimeter Simulation | End-Term Project
"""

import numpy as np

# Constants
V_REF = 5.0       # Reference voltage (V)
V_SUPPLY = 5.0    # Supply voltage for RC timing (V)
NOISE_SIGMA = 0.005  # 0.5% of true value (Gaussian noise std dev)

RNG = np.random.default_rng(seed=42)


def _add_noise(true_val: float) -> float:
    """Add Gaussian noise with σ = 0.5% of true value."""
    return true_val + RNG.normal(0, NOISE_SIGMA * abs(true_val))


def measure_resistance(true_r: float, r_ref: float = 10_000.0) -> tuple[float, float]:
    """
    Voltage-divider method.
    R = R_ref × V_adc / (V_ref − V_adc)

    Args:
        true_r:  True resistance (Ω)
        r_ref:   Known reference resistor (Ω), default 10 kΩ

    Returns:
        (measured_r, error_pct)
    """
    # True V_adc from ideal voltage divider
    v_adc_true = V_REF * true_r / (r_ref + true_r)
    v_adc = _add_noise(v_adc_true)

    # Clamp to valid range to avoid divide-by-zero
    v_adc = np.clip(v_adc, 1e-6, V_REF - 1e-6)

    measured_r = r_ref * v_adc / (V_REF - v_adc)
    error_pct = abs(measured_r - true_r) / true_r * 100.0
    return measured_r, error_pct


def measure_capacitance(true_c: float, r_ref: float = 10_000.0) -> tuple[float, float]:
    """
    RC charge-discharge timing method.
    C = t / R_ref  where t is time to reach 63.2% of V_supply.

    Args:
        true_c:  True capacitance (F)
        r_ref:   Charging resistor (Ω), default 10 kΩ

    Returns:
        (measured_c, error_pct)
    """
    # Time constant τ = R_ref × C  → add noise to τ
    tau_true = r_ref * true_c
    tau_measured = _add_noise(tau_true)
    tau_measured = max(tau_measured, 1e-15)  # physical guard

    measured_c = tau_measured / r_ref
    error_pct = abs(measured_c - true_c) / true_c * 100.0
    return measured_c, error_pct


def measure_inductance(true_l: float, c_ref: float = 100e-9) -> tuple[float, float]:
    """
    LC resonance frequency method.
    L = 1 / ((2πf)² × C_ref)  where f is measured resonant frequency.

    Args:
        true_l:  True inductance (H)
        c_ref:   Known reference capacitor (F), default 100 nF

    Returns:
        (measured_l, error_pct)
    """
    import math
    f_true = 1.0 / (2.0 * math.pi * math.sqrt(true_l * c_ref))
    f_measured = _add_noise(f_true)
    f_measured = max(f_measured, 1e-3)  # physical guard

    measured_l = 1.0 / ((2.0 * math.pi * f_measured) ** 2 * c_ref)
    error_pct = abs(measured_l - true_l) / true_l * 100.0
    return measured_l, error_pct


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== measurement.py self-test ===\n")

    r_val, r_err = measure_resistance(4_700)
    print(f"Resistance  | true=4700 Ω    | measured={r_val:.2f} Ω   | error={r_err:.3f}%")

    c_val, c_err = measure_capacitance(470e-9)
    print(f"Capacitance | true=470 nF    | measured={c_val*1e9:.2f} nF | error={c_err:.3f}%")

    l_val, l_err = measure_inductance(10e-3)
    print(f"Inductance  | true=10 mH     | measured={l_val*1e3:.4f} mH | error={l_err:.3f}%")
