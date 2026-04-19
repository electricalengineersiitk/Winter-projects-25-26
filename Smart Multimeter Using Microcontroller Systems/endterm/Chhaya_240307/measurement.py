import numpy as np
import math

def measure_resistance(true_r, r_ref=10000.0, v_ref=3.3):
    # Apply noise to true value
    noisy_r = np.random.normal(true_r, 0.005 * true_r)
    # Physical simulation
    v_adc = v_ref * (noisy_r / (noisy_r + r_ref))
    measured_r = r_ref * (v_adc / (v_ref - v_adc))
    
    error_percent = abs(measured_r - true_r) / true_r * 100
    return measured_r, error_percent

def measure_capacitance(true_c, r_ref=10000.0):
    noisy_c = np.random.normal(true_c, 0.005 * true_c)
    t = r_ref * noisy_c
    measured_c = t / r_ref
    
    error_percent = abs(measured_c - true_c) / true_c * 100
    return measured_c, error_percent

def measure_inductance(true_l, c_ref=1e-6):
    noisy_l = np.random.normal(true_l, 0.005 * true_l)
    f = 1 / (2 * math.pi * math.sqrt(noisy_l * c_ref))
    measured_l = 1 / ((2 * math.pi * f)**2 * c_ref)
    
    error_percent = abs(measured_l - true_l) / true_l * 100
    return measured_l, error_percent