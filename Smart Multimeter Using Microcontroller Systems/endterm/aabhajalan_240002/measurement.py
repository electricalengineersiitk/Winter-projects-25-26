"""
measurement.py — R, C, L formulas + noise model
Simulates ADC imprecision via Gaussian noise (σ = 0.5% of true value).
"""

import numpy as np

# Physical constants
V_REF = 5.0        # Reference voltage (V)
R_REF = 10_000     # Reference resistor for voltage divider (10 kΩ)
C_REF = 1e-6       # Reference capacitor for LC resonance (1 µF)

NOISE_SIGMA = 0.005  # 0.5% of true value


def _add_noise(true_val: float) -> float:
    """Return true_val with Gaussian noise added."""
    return np.random.normal(true_val, NOISE_SIGMA * abs(true_val))


# ---------------------------------------------------------------------------
# Resistance  —  voltage divider method
#   R = R_ref × V_adc / (V_ref − V_adc)
# ---------------------------------------------------------------------------
def measure_resistance(true_r: float) -> tuple[float, float]:
    """
    Simulate resistance measurement using a voltage divider.
    Noise is applied to the true value to model ADC imprecision at ±0.5%.

    Args:
        true_r: True resistance in Ohms.

    Returns:
        (measured_value, error_percent)
    """
    # Apply noise directly to the true resistance value (models ADC ± 0.5%)
    measured = _add_noise(true_r)
    measured = max(measured, 1e-3)  # physical floor
    error_pct = abs(measured - true_r) / true_r * 100
    return measured, error_pct


# ---------------------------------------------------------------------------
# Capacitance  —  RC charge-discharge timing
#   C = t / R_ref   where t is the time for V to reach 63.2% of V_supply
# ---------------------------------------------------------------------------
def measure_capacitance(true_c: float) -> tuple[float, float]:
    """
    Simulate capacitance measurement via RC time-constant.

    Args:
        true_c: True capacitance in Farads.

    Returns:
        (measured_value, error_percent)
    """
    # True time constant τ = R_ref × C
    tau_true = R_REF * true_c
    tau_noisy = _add_noise(tau_true)
    tau_noisy = max(tau_noisy, 1e-15)  # physical floor

    # C = τ / R_ref
    measured = tau_noisy / R_REF
    error_pct = abs(measured - true_c) / true_c * 100
    return measured, error_pct


# ---------------------------------------------------------------------------
# Inductance  —  LC resonance frequency method
#   L = 1 / ((2πf)² × C_ref)   where f is the measured resonant frequency
# ---------------------------------------------------------------------------
def measure_inductance(true_l: float) -> tuple[float, float]:
    """
    Simulate inductance measurement via LC resonance.

    Args:
        true_l: True inductance in Henrys.

    Returns:
        (measured_value, error_percent)
    """
    # True resonant frequency f = 1 / (2π√(L·C_ref))
    f_true = 1.0 / (2 * np.pi * np.sqrt(true_l * C_REF))
    f_noisy = _add_noise(f_true)
    f_noisy = max(f_noisy, 1e-3)  # physical floor

    # L = 1 / ((2πf)² × C_ref)
    measured = 1.0 / ((2 * np.pi * f_noisy) ** 2 * C_REF)
    error_pct = abs(measured - true_l) / true_l * 100
    return measured, error_pct


# ---------------------------------------------------------------------------
# Quick sanity check when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== measurement.py self-test ===\n")

    for r in [100, 1_000, 10_000, 100_000, 1_000_000]:
        val, err = measure_resistance(r)
        print(f"R_true={r:>10,.0f} Ω  →  measured={val:>12,.2f} Ω  error={err:.3f}%")

    print()
    for c in [10e-9, 100e-9, 1e-6, 10e-6, 100e-6]:
        val, err = measure_capacitance(c)
        print(f"C_true={c:>12.2e} F  →  measured={val:.4e} F  error={err:.3f}%")

    print()
    for l in [10e-6, 100e-6, 1e-3, 10e-3, 100e-3]:
        val, err = measure_inductance(l)
        print(f"L_true={l:>12.2e} H  →  measured={val:.4e} H  error={err:.3f}%")
