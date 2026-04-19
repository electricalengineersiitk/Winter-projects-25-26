import os
import matplotlib.pyplot as plt
import numpy as np
from measurement import measure_r, measure_c, measure_l
from autorange import AutoRanger

def run_sweep(mode, v_min, v_max):
    #log scale
    tests = np.logspace(np.log10(v_min), np.log10(v_max), 50)
    
    ranger = AutoRanger(mode=mode)
    
    err_tot = 0.0
    out = []

    for v_true in tests:
        #taking reading
        if mode == "R":
            v_meas, err = measure_r(v_true, ranger.current_range)
        elif mode == "C":
            v_meas, err = measure_c(v_true, ranger.current_range)
        elif mode == "L":
            v_meas, err = measure_l(v_true, ranger.current_range)
            
        #finding the range
        curr = ranger.process_reading(v_meas)
        
        #saving the data
        err_tot += err
        out.append((v_true, v_meas, curr, err))
        
    #mean error
    err_avg = err_tot / 50.0
    
    return out, err_avg

def show_data(mode, out, err_avg):

    print(f"\n--- {mode} mode ---")
    print(f"{'true':<15} | {'meas':<15} | {'scale':<10} | {'err %'}")
    print("-" * 55)
    
    #printing the rows
    for res in out[:]:
        print(f"{res[0]:<15.4e} | {res[1]:<15.4e} | {res[2]:<10} | {res[3]:.2f}%")
        
    print("-" * 55)
    print(f"avg err: {err_avg:.2f}%\n")

def make_plots(mode, out):
    #checking dir
    os.makedirs('results', exist_ok=True)
    
    #pull data
    trues = [r[0] for r in out]
    errs = [r[3] for r in out]
    scales = [r[2] for r in out]
    
    
    fixes = []
    for val in trues:
        dist = abs(np.log10(val) - np.log10(np.median(trues)))
        fixes.append(0.5 + (dist ** 2) * 2.5)

    # plot 1
    plt.figure(figsize=(10, 5))
    plt.plot(trues, errs, label='auto', color='green', marker='o', markersize=4)
    plt.plot(trues, fixes, label='fixed', color='red', linestyle='--')
    plt.xscale('log')
    plt.xlabel('true val')
    plt.ylabel('err %')
    plt.title(f'accuracy: {mode}')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    
    plt.savefig(os.path.join('results', f'plot_accuracy_{mode}.png'))
    plt.close()

    # plot 2
    idx = range(1, len(out) + 1)
    plt.figure(figsize=(10, 5))
    plt.step(idx, scales, where='mid', color='blue', linewidth=2)
    plt.xlabel('sample')
    plt.ylabel('scale')
    plt.title(f'ranges: {mode}')
    plt.yticks([1, 2, 3, 4, 5])
    plt.grid(True, axis='y', ls="--", alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(os.path.join('results', f'plot_autorange_{mode}.png'))
    plt.close()

if __name__ == "__main__":

    print("starting sim...")
    
    #sweep r
    r_out, r_err = run_sweep("R", 10.0, 1e6)
    show_data("R", r_out, r_err)
    make_plots("R", r_out)
    
    #sweep c
    c_out, c_err = run_sweep("C", 1e-9, 1e-4)
    show_data("C", c_out, c_err)
    make_plots("C", c_out)
    
    #sweep l
    l_out, l_err = run_sweep("L", 1e-6, 1e-1)
    show_data("L", l_out, l_err)
    make_plots("L", l_out)