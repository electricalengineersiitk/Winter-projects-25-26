import numpy as np

# Simulate reading Resistance (Ohms)
def measure_resistance(true_value):
    # Hardware constants
    v_ref = 5.0
    r_ref = 10000.0 
    
    # basic voltage divider formula
    v_adc = (true_value * v_ref) / (true_value + r_ref)
    
    # get resistance value from the voltage
    calculated = r_ref * v_adc / (v_ref - v_adc)
    
    # add some noise for realism (0.5%)
    noise = 0.005 * true_value
    measured_with_noise = np.random.normal(calculated, noise)
    
    # calculate the error percentage
    error = abs(true_value - measured_with_noise) / true_value * 100
    return measured_with_noise, error

# Simulate reading Capacitance (Farads)
def measure_capacitance(true_value):
    # adding 0.5% noise to the true value
    noise = 0.005 * true_value
    measured_with_noise = np.random.normal(true_value, noise)
    
    # simple error calculation
    error = abs(true_value - measured_with_noise) / true_value * 100
    return measured_with_noise, error

# Simulate reading Inductance (Henries)
def measure_inductance(true_value):
    # adding 0.5% noise to the true value
    noise = 0.005 * true_value
    measured_with_noise = np.random.normal(true_value, noise)
    
    # simple error calculation
    error = abs(true_value - measured_with_noise) / true_value * 100
    return measured_with_noise, error


