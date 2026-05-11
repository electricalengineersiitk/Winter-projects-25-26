from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def _rng_or_default(rng: np.random.Generator | None) -> np.random.Generator:
    return rng if rng is not None else np.random.default_rng()


def _error_percent(true_value: float, measured_value: float) -> float:
    return abs((measured_value - true_value) / true_value) * 100.0


def measure_resistance(
    true_value: float,
    *,
    r_ref: float = 10_000.0,
    v_ref: float = 3.3,
    rng: np.random.Generator | None = None,
) -> Tuple[float, float]:
    if true_value <= 0 or r_ref <= 0 or v_ref <= 0:
        raise ValueError("true_value, r_ref, and v_ref must be positive")

    generator = _rng_or_default(rng)
    v_adc_true = v_ref * true_value / (r_ref + true_value)
    sigma_v = 0.005 * v_adc_true
    v_adc_measured = float(generator.normal(v_adc_true, sigma_v))
    v_adc_measured = min(max(v_adc_measured, 1e-12), v_ref - 1e-9)

    measured_value = r_ref * v_adc_measured / (v_ref - v_adc_measured)
    return measured_value, _error_percent(true_value, measured_value)


def measure_capacitance(
    true_value: float,
    *,
    r_ref: float = 100_000.0,
    rng: np.random.Generator | None = None,
) -> Tuple[float, float]:
    if true_value <= 0 or r_ref <= 0:
        raise ValueError("true_value and r_ref must be positive")

    generator = _rng_or_default(rng)
    t_true = true_value * r_ref
    sigma_t = 0.005 * t_true
    t_measured = max(float(generator.normal(t_true, sigma_t)), 1e-15)

    measured_value = t_measured / r_ref
    return measured_value, _error_percent(true_value, measured_value)


def measure_inductance(
    true_value: float,
    *,
    c_ref: float = 100e-9,
    rng: np.random.Generator | None = None,
) -> Tuple[float, float]:
    if true_value <= 0 or c_ref <= 0:
        raise ValueError("true_value and c_ref must be positive")

    generator = _rng_or_default(rng)
    f_true = 1.0 / (2.0 * math.pi * math.sqrt(true_value * c_ref))
    sigma_f = 0.005 * f_true
    f_measured = max(float(generator.normal(f_true, sigma_f)), 1e-12)

    measured_value = 1.0 / (((2.0 * math.pi * f_measured) ** 2) * c_ref)
    return measured_value, _error_percent(true_value, measured_value)
