import numpy as np

NOISE_FACTOR = 0.005  # 0.5%

def add_noise(true_val):
    return np.random.normal(true_val, NOISE_FACTOR * true_val)

def measure_resistance(true_R, R_ref=1000, V_ref=5):
    noisy_R = add_noise(true_R)
    error = abs(noisy_R - true_R) / true_R * 100
    return noisy_R, error

def measure_capacitance(true_C, R_ref=1000):
    # t = R * C → simulate time measurement
    t_true = R_ref * true_C
    t_measured = add_noise(t_true)

    measured_C = t_measured / R_ref
    error = abs(measured_C - true_C) / true_C * 100
    return measured_C, error

def measure_inductance(true_L, C_ref=1e-6):
    # f = 1 / (2π√(LC))
    f_true = 1 / (2 * np.pi * np.sqrt(true_L * C_ref))
    f_measured = add_noise(f_true)

    measured_L = 1 / ((2 * np.pi * f_measured) ** 2 * C_ref)
    error = abs(measured_L - true_L) / true_L * 100
    return measured_L, error