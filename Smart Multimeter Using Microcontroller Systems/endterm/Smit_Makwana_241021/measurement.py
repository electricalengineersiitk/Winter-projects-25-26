import numpy as np

def add_noise(true_value):
    """Simulate ADC imprecision by adding Gaussian noise (sigma = 0.5% of true value)."""
    return np.random.normal(true_value, 0.005 * true_value)

def measure_resistance(true_r, active_range=1, v_supply=3.3):
    """R = R_ref * V_adc / (V_ref - V_adc)"""
    range_to_ref = {1: 100, 2: 1000, 3: 10000, 4: 100000, 5: 1000000}
    r_ref = range_to_ref.get(active_range, 1000)
    
    v_adc_ideal = (true_r * v_supply) / (r_ref + true_r)
    v_adc_noisy = add_noise(v_adc_ideal)
    
    v_adc_noisy = np.clip(v_adc_noisy, 0, v_supply - 1e-6)
    
    measured_r = r_ref * v_adc_noisy / (v_supply - v_adc_noisy)
    error_pct = abs(measured_r - true_r) / true_r * 100
    
    return measured_r, error_pct

def measure_capacitance(true_c, r_ref=10000):
    """C = t / R_ref (t = time for V to reach 63.2% of V_supply)"""
    t_ideal = true_c * r_ref
    t_noisy = add_noise(t_ideal)
    
    measured_c = t_noisy / r_ref
    error_pct = abs(measured_c - true_c) / true_c * 100
    return measured_c, error_pct

def measure_inductance(true_l, c_ref=1e-6):
    """L = 1 / ((2 * pi * f)^2 * C_ref)"""
    f_ideal = 1 / (2 * np.pi * np.sqrt(true_l * c_ref))
    f_noisy = add_noise(f_ideal)
    
    measured_l = 1 / ((2 * np.pi * f_noisy)**2 * c_ref)
    error_pct = abs(measured_l - true_l) / true_l * 100
    return measured_l, error_pct