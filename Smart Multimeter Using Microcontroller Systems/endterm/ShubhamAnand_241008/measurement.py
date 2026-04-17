"""
measurement.py — R, C, L formulas + noise model
"""

import numpy as np

# Fixed references that is used in measurement formulas
V_REF = 5.0       # Reference voltage (V)
R_REF = 10000.0   # Default reference resistor for RC timing (Ω)
C_REF = 1e-6      # Reference capacitor for LC resonance (F)
NOISE_SIGMA = 0.005  # 0.5% of true value (Gaussian noise std dev)


def _r_ref_for_range(true_r: float) -> float:
    """
    Select R_ref matched to the value being measured for best ADC headroom.
    Optimal voltage divider accuracy is achieved when R ≈ R_ref.
    """
    if true_r < 150:
        return 100.0
    elif true_r < 1500:
        return 1000.0
    elif true_r < 15000:
        return 10000.0
    elif true_r < 150000:
        return 100000.0
    else:
        return 1000000.0


def measure_resistance(true_r: float, rng: np.random.Generator = None) -> tuple[float, float]:
    """
    Measure resistance using voltage divider method.
    Formula: R = R_ref × V_adc / (V_ref − V_adc)
    R_ref is selected per range for best ADC accuracy.

    Args:
        true_r: True resistance value in Ohms
        rng: Optional numpy random generator for reproducibility

    Returns:
        (measured_value, error_percent)
    """
    if rng is None:
        rng = np.random.default_rng()

    r_ref = _r_ref_for_range(true_r)

    # Ideal V_adc from voltage divider: V_adc = V_ref * R / (R + R_ref)
    v_adc_ideal = V_REF * true_r / (true_r + r_ref)

    # Add Gaussian noise to V_adc
    noise = rng.normal(0, NOISE_SIGMA * v_adc_ideal)
    v_adc_noisy = np.clip(v_adc_ideal + noise, 1e-9, V_REF - 1e-9)

    # Derive R from noisy V_adc
    measured_r = r_ref * v_adc_noisy / (V_REF - v_adc_noisy)

    error_pct = abs(measured_r - true_r) / true_r * 100.0
    return measured_r, error_pct


def measure_capacitance(true_c: float, rng: np.random.Generator = None) -> tuple[float, float]:
    """
    Measure capacitance via RC charge-discharge timing.
    Formula: C = t / R_ref   where t = time for V to reach 63.2% of V_supply

    Args:
        true_c: True capacitance in Farads
        rng: Optional numpy random generator

    Returns:
        (measured_value, error_percent)
    """
    if rng is None:
        rng = np.random.default_rng()

    # True time constant τ = R_ref * C
    true_tau = R_REF * true_c

    # Add noise to the measured time constant
    noise = rng.normal(0, NOISE_SIGMA * true_tau)
    measured_tau = max(true_tau + noise, 1e-15)

    # Derive C = τ / R_ref
    measured_c = measured_tau / R_REF

    error_pct = abs(measured_c - true_c) / true_c * 100.0
    return measured_c, error_pct


def measure_inductance(true_l: float, rng: np.random.Generator = None) -> tuple[float, float]:
    """
    Measure inductance via LC resonance frequency.
    Formula: L = 1 / ((2πf)² × C_ref)   where f = resonant frequency

    Args:
        true_l: True inductance in H(unit)
        rng: Optional numpy random generator

    Returns:
        (measured_value, error_percent)
    """
    if rng is None:
        rng = np.random.default_rng()

    # True resonant frequency: f = 1 / (2π√(L·C_ref))
    true_f = 1.0 / (2.0 * np.pi * np.sqrt(true_l * C_REF))

    # Add noise to frequency measurement
    noise = rng.normal(0, NOISE_SIGMA * true_f)
    measured_f = max(true_f + noise, 1e-6)

    # Derive L from measured frequency
    measured_l = 1.0 / ((2.0 * np.pi * measured_f) ** 2 * C_REF)

    error_pct = abs(measured_l - true_l) / true_l * 100.0
    return measured_l, error_pct


# sample-test
if __name__ == "__main__":
    rng = np.random.default_rng(42)

    test_cases = [
        ("Resistance", measure_resistance, [100, 1e3, 10e3, 100e3, 1e6], "Ω"),
        ("Capacitance", measure_capacitance, [10e-9, 100e-9, 1e-6, 10e-6, 100e-6], "F"),
        ("Inductance", measure_inductance, [10e-6, 100e-6, 1e-3, 10e-3, 100e-3], "H"),
    ]

    for name, fn, values, unit in test_cases:
        print(f"\n{'─'*55}")
        print(f"  {name} self-test")
        print(f"{'─'*55}")
        print(f"  {'True Value':>14}  {'Measured':>14}  {'Error %':>8}")
        for v in values:
            meas, err = fn(v, rng)
            print(f"  {v:>14.4g}  {meas:>14.4g}  {err:>7.3f}%")
