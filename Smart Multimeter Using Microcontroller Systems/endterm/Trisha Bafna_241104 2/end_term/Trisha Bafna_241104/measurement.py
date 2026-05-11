import numpy as np

def add_noise(true_val):
    # Simulates ADC noise by adding Gaussian noise (sigma = 0.5% of true_val)
    return np.random.normal(true_val, 0.005 * true_val)

def measure_resistance(true_r, r_ref=1000.0, v_ref=3.3):
    # Resistance: R = R_ref * V_adc / (V_ref - V_adc)
    v_adc_ideal = (true_r * v_ref) / (true_r + r_ref)
    v_adc_noisy = add_noise(v_adc_ideal)
    measured_r = r_ref * v_adc_noisy / (v_ref - v_adc_noisy)
    error = abs(measured_r - true_r) / true_r * 100.0
    return measured_r, error

def measure_capacitance(true_c, r_ref=10000.0):
    # Capacitance: C = t / R_ref
    t_ideal = true_c * r_ref
    t_noisy = add_noise(t_ideal)
    measured_c = t_noisy / r_ref
    error = abs(measured_c - true_c) / true_c * 100.0
    return measured_c, error

def measure_inductance(true_l, c_ref=1e-6):
    # Inductance: L = 1 / ((2 * pi * f)^2 * C_ref)
    f_ideal = 1.0 / (2 * np.pi * np.sqrt(true_l * c_ref))
    f_noisy = add_noise(f_ideal)
    measured_l = 1.0 / ((2 * np.pi * f_noisy)**2 * c_ref)
    error = abs(measured_l - true_l) / true_l * 100.0
    return measured_l, error