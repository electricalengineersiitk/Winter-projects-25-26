"""
measurement.py
--------------
Physics-based measurement simulation for resistance, capacitance, and inductance.
Each function adds Gaussian noise (σ = 0.5% of true value) to simulate ADC imprecision.
"""

import numpy as np

# ── Constants ──────────────────────────────────────────────────────────────────
V_REF = 5.0       # ADC reference voltage (V)
R_REF = 10_000.0  # Reference resistor in voltage-divider (Ω)
R_1 = 10_000.0    # Charge resistor used in RC timing circuit (Ω)

NOISE_SIGMA_RATIO = 0.005   # 0.5 % of true value


def _add_noise(true_val: float) -> float:
    """Return a noisy measurement using Gaussian noise at σ = 0.5 % of true_val."""
    return float(np.random.normal(true_val, NOISE_SIGMA_RATIO * abs(true_val)))


def measure_resistance(true_val: float) -> tuple[float, float]:
    """
    Simulate a resistor measurement using a voltage-divider ADC circuit.

    Formula: R = R_ref * V_adc / (V_ref - V_adc)
    where V_adc is back-calculated from the noisy resistance reading.

    Args:
        true_val: True resistance in Ohms.

    Returns:
        (measured_value, error_pct) – measured resistance (Ω) and absolute % error.
    """
    # Simulate the noisy raw reading
    noisy_r = _add_noise(true_val)
    # Forward: V_adc = V_ref * R / (R_ref + R)
    v_adc = V_REF * noisy_r / (R_REF + noisy_r)
    # Inverse (as a real meter would compute from V_adc)
    measured = R_REF * v_adc / (V_REF - v_adc)
    error_pct = abs(measured - true_val) / true_val * 100.0
    return measured, error_pct


def measure_capacitance(true_val: float) -> tuple[float, float]:
    """
    Simulate a capacitance measurement via RC charge-time method.

    Formula: C = t / R_1
    where t is the time for the capacitor to reach 63.2 % of V_supply
    (one RC time constant τ = R_1 * C).

    Args:
        true_val: True capacitance in Farads.

    Returns:
        (measured_value, error_pct) – measured capacitance (F) and absolute % error.
    """
    # τ = R_1 * C  →  t = τ  (time to reach 63.2 % of Vsupply)
    tau_true = R_1 * true_val
    tau_noisy = _add_noise(tau_true)
    measured = tau_noisy / R_1
    error_pct = abs(measured - true_val) / true_val * 100.0
    return measured, error_pct


def measure_inductance(true_val: float) -> tuple[float, float]:
    """
    Simulate an inductance measurement via LC resonance method.

    Formula: L = 1 / ((2π f)² * C_ref)
    where f is the measured resonant frequency of an LC tank circuit.

    A reference capacitor C_ref is chosen so that the resonant frequency
    falls in a convenient measurement range (≈ 10 kHz nominal).

    Args:
        true_val: True inductance in Henries.

    Returns:
        (measured_value, error_pct) – measured inductance (H) and absolute % error.
    """
    C_REF = 1e-9  # 1 nF reference capacitor

    # Resonant frequency from true L: f = 1 / (2π √(L * C_ref))
    f_true = 1.0 / (2.0 * np.pi * np.sqrt(true_val * C_REF))
    f_noisy = _add_noise(f_true)
    # Recover L from noisy f
    measured = 1.0 / ((2.0 * np.pi * f_noisy) ** 2 * C_REF)
    error_pct = abs(measured - true_val) / true_val * 100.0
    return measured, error_pct
