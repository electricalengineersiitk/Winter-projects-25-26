import numpy as np
from autoranging import choose_resistor

def get_resistance(R_true):
    V_supply = 5

    # select suitable reference resistor
    R_ref = choose_resistor(R_true)

    # voltage divider output
    V_out = V_supply * (R_true / (R_true + R_ref))

    # adding small noise
    noise = np.random.normal(0, 0.005 * V_out)
    V_noisy = V_out + noise

    # reverse calculation
    R_calc = (V_noisy * R_ref) / (V_supply - V_noisy)

    # percentage error
    error = abs(R_calc - R_true) / R_true * 100

    return R_calc, error