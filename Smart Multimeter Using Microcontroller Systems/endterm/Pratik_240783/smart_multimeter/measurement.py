import numpy as np

def measure_r(r_true, scale=1, v_ref=5.0):
    #set ref
    refs = {1: 100.0, 2: 1e3, 3: 1e4, 4: 1e5, 5: 1e6}
    ref = refs.get(scale, 10000.0)
    
    #calc ideal
    v_ideal = v_ref * (r_true / (r_true + ref))
    
    #adding noise
    v_noise = np.random.normal(v_ideal, 0.005 * v_ideal)
    
    #clamp vals
    v_noise = min(v_noise, v_ref - 1e-6)
    v_noise = max(v_noise, 1e-6)
    
    r_meas = ref * v_noise / (v_ref - v_noise)
    
    #get error
    err = abs(r_meas - r_true) / r_true * 100.0
    
    return r_meas, err

def measure_c(c_true, scale=1):
    #set ref
    refs = {1: 100000.0, 2: 10000.0, 3: 1000.0, 4: 100.0, 5: 10.0}
    ref = refs.get(scale, 10000.0)
    
    #calc ideal
    t_ideal = c_true * ref
    
    #add noise
    t_noise = np.random.normal(t_ideal, 0.005 * t_ideal)
    t_noise = max(t_noise, 1e-12) 
    
    c_meas = t_noise / ref
    
    #error
    err = abs(c_meas - c_true) / c_true * 100.0
    
    return c_meas, err

def measure_l(l_true, scale=1):
    #set ref
    refs = {1: 1e-6, 2: 1e-7, 3: 1e-8, 4: 1e-9, 5: 1e-10}
    ref = refs.get(scale, 1e-9)
    
    #calc ideal
    f_ideal = 1.0 / (2 * np.pi * np.sqrt(l_true * ref))
    
    #noise add
    f_noise = np.random.normal(f_ideal, 0.005 * f_ideal)
    f_noise = max(f_noise, 1.0) 
    
    l_meas = 1.0 / ((2 * np.pi * f_noise)**2 * ref)
    
    #error
    err = abs(l_meas - l_true) / l_true * 100.0
    
    return l_meas, err