"""
generate_plots.py — Python-based result plot generator
EV Throttle Control | End-Term Project | Bhavit Meena (240272)

Generates the 4 required result plots using scipy.signal and matplotlib,
exactly mirroring what the MATLAB scripts produce. Run this script to
reproduce all result images without MATLAB.

Usage:
    python3 generate_plots.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal
import os

# ── Output directory ────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Matplotlib style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.5,
    'figure.dpi': 150,
    'figure.figsize': (9, 5),
})

# ── Plant parameters ──────────────────────────────────────────────────────────
K   = 1.0    # steady-state gain
TAU = 0.5    # time constant (s)

# Transfer function: G(s) = K / (τs + 1)
#   numerator:   [K]
#   denominator: [τ, 1]
G = signal.TransferFunction([K], [TAU, 1])

# ── Tuned PID gains (verified: rise≈0.55s, 0% OS, SSE≈0, settle≈1.04s) ─────
PID_KP, PID_KI, PID_KD = 3.0, 5.0, 0.2

# ─────────────────────────────────────────────────────────────────────────────
def pid_closed_loop(Kp, Ki, Kd, G=G):
    """
    Build closed-loop TF for PID + first-order plant.

    C(s) = Kp + Ki/s + Kd*s  =  (Kd*s^2 + Kp*s + Ki) / s

    Open-loop: L(s) = C(s)*G(s)
        num = K * [Kd, Kp, Ki]
        den = [TAU, 1, 0]  (plant den * s)

    Closed-loop: T(s) = L / (1+L)
    """
    num_L = np.polymul([K], [Kd, Kp, Ki])          # K*(Kd s^2 + Kp s + Ki)
    den_L = np.polymul([TAU, 1.0], [1.0, 0.0])     # (τs+1)*s

    # T(s) = num_L / (den_L + num_L)  — padding shorter poly
    max_len = max(len(num_L), len(den_L))
    num_p = np.concatenate([np.zeros(max_len - len(num_L)), num_L])
    den_p = np.concatenate([np.zeros(max_len - len(den_L)), den_L])
    den_T = den_p + num_p

    return signal.TransferFunction(num_p, den_T)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1: Open-Loop Step Response
# ─────────────────────────────────────────────────────────────────────────────
def plot_open_loop():
    t = np.linspace(0, 5, 1000)
    _, y = signal.step(G, T=t)

    fig, ax = plt.subplots()
    ax.plot(t, y, 'r-', linewidth=2, label=r'Plant $G(s)=\frac{1}{0.5s+1}$  ($K=1$, $\tau=0.5$ s)')
    ax.axhline(K, color='k', linestyle=':', linewidth=1, label='Steady-State = 1.0')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Motor Speed (normalised)')
    ax.set_title('Open-Loop Motor Step Response')
    ax.legend(loc='lower right')
    ax.set_ylim(-0.05, 1.3)

    # Annotate ~63% rise (1 time constant)
    idx_tau = np.argmin(np.abs(t - TAU))
    ax.annotate(f'τ = {TAU} s\n(63% of SS)',
                xy=(TAU, y[idx_tau]),
                xytext=(TAU + 0.3, y[idx_tau] - 0.18),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=9, color='gray')

    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, 'open_loop_response.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'  ✓ Saved {path}')


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2: PID Closed-Loop Step Response
# ─────────────────────────────────────────────────────────────────────────────
def plot_pid():
    Kp, Ki, Kd = PID_KP, PID_KI, PID_KD   # 3.0, 5.0, 0.2
    T_pid = pid_closed_loop(Kp, Ki, Kd)

    t = np.linspace(0, 6, 3000)
    _, y = signal.step(T_pid, T=t)

    overshoot = max((y.max() - 1.0) * 100, 0.0)
    idx_r10 = np.argmax(y >= 0.10)
    idx_r90 = np.argmax(y >= 0.90)
    rise_time = t[idx_r90] - t[idx_r10]
    settled_arr = np.where(np.abs(y - 1.0) < 0.02)[0]
    settle_time = t[settled_arr[0]] if len(settled_arr) else float('nan')
    sse = abs(1.0 - y[-1])

    fig, ax = plt.subplots()
    ax.plot(t, y, 'b-', linewidth=2,
            label=f'PID  ($K_p$={Kp}, $K_i$={Ki}, $K_d$={Kd})')
    ax.axhline(1.0, color='k', linestyle=':', linewidth=1, label='Setpoint = 1.0')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Motor Speed (normalised)')
    ax.set_title('PID Closed-Loop Step Response')
    ax.legend(loc='lower right')
    ax.set_ylim(-0.05, 1.4)

    # Rise time annotation
    ax.annotate(f'Rise time ≈ {rise_time:.2f} s',
                xy=(t[idx_r90], 0.90),
                xytext=(t[idx_r90] + 0.4, 0.65),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=9, color='gray')
    if y.max() > 1.01:
        idx_pk = y.argmax()
        ax.annotate(f'OS ≈ {overshoot:.1f}%',
                    xy=(t[idx_pk], y[idx_pk]),
                    xytext=(t[idx_pk] + 0.2, y[idx_pk] + 0.04),
                    arrowprops=dict(arrowstyle='->', color='orange'),
                    fontsize=9, color='orange')
    # Settling annotation
    if not np.isnan(settle_time):
        ax.axvline(settle_time, color='green', linestyle=':', alpha=0.6,
                   label=f'Settle ≈ {settle_time:.2f} s')
        ax.legend(loc='lower right')

    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, 'pid_response.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'  ✓ Saved {path}')
    print(f'    Metrics → Rise: {rise_time:.3f}s | OS: {overshoot:.2f}% | Settle: {settle_time:.3f}s | SSE: {sse:.5f}')


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3: Gain-Scheduled Response
# ─────────────────────────────────────────────────────────────────────────────
ZONES = [
    {'name': 'Zone 1: Low (0–30%)',    'limits': (0.00, 0.30), 'Kp': 2.0, 'Ki': 3.0, 'Kd': 0.10},
    {'name': 'Zone 2: Mid (30–70%)',   'limits': (0.30, 0.70), 'Kp': 3.0, 'Ki': 5.0, 'Kd': 0.20},
    {'name': 'Zone 3: High (70–100%)', 'limits': (0.70, 1.00), 'Kp': 4.0, 'Ki': 8.0, 'Kd': 0.30},
]
SETPOINTS  = [0.20, 0.50, 0.85]
DURATIONS  = [3.0,  3.0,  3.0]


def get_zone(sp):
    for z in ZONES:
        if z['limits'][0] <= sp <= z['limits'][1]:
            return z
    return ZONES[-1]


def plot_gain_scheduled():
    dt = 0.01
    t_all, y_all, sp_all = [], [], []
    boundaries = []
    t_start = 0.0

    for i, (sp, dur) in enumerate(zip(SETPOINTS, DURATIONS)):
        z = get_zone(sp)
        T_z = pid_closed_loop(z['Kp'], z['Ki'], z['Kd'], G)
        t_seg = np.arange(0, dur, dt)
        _, y_seg = signal.step(T_z, T=t_seg)
        y_seg = y_seg * sp   # scale to actual setpoint fraction

        if i > 0:
            boundaries.append(t_start)

        t_all.append(t_seg + t_start)
        y_all.append(y_seg)
        sp_all.append(np.full_like(t_seg, sp))
        t_start += dur

    t_all  = np.concatenate(t_all)
    y_all  = np.concatenate(y_all)
    sp_all = np.concatenate(sp_all)

    fig, ax = plt.subplots()
    ax.plot(t_all, y_all,  'g-',  linewidth=2, label='Motor Speed (Gain-Scheduled)')
    ax.plot(t_all, sp_all, 'k--', linewidth=1, label='Throttle Setpoint')

    colors = ['#e056ef', '#f0a500', '#0a6cf5']
    labels = ['Zone 1→2 (0.20→0.50)', 'Zone 2→3 (0.50→0.85)']
    for j, xb in enumerate(boundaries):
        ax.axvline(xb, color=colors[j], linestyle='--', linewidth=1.5,
                   label=f'Transition: {labels[j]}')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Motor Speed (normalised)')
    ax.set_title('Gain-Scheduled Throttle Control (3 Zones)')
    ax.legend(loc='upper left', fontsize=8)

    # Zone labels
    zone_centres = [1.5, 4.5, 7.5]
    zone_titles  = ['Zone 1\n(Low)', 'Zone 2\n(Mid)', 'Zone 3\n(High)']
    for xc, zt in zip(zone_centres, zone_titles):
        ax.text(xc, 0.03, zt, ha='center', fontsize=8, color='gray',
                transform=ax.get_xaxis_transform())

    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, 'gainscheduled_response.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'  ✓ Saved {path}')


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4: Comparison Plot
# ─────────────────────────────────────────────────────────────────────────────
def plot_comparison():
    Kp, Ki, Kd = PID_KP, PID_KI, PID_KD   # 3.0, 5.0, 0.2
    T_pid = pid_closed_loop(Kp, Ki, Kd)

    # Gain-scheduled zone 2 (mid, for a single fair comparison)
    z2 = ZONES[1]
    T_gs = pid_closed_loop(z2['Kp'], z2['Ki'], z2['Kd'])

    t = np.linspace(0, 6, 3000)
    _, y_ol  = signal.step(G,     T=t)
    _, y_pid = signal.step(T_pid, T=t)
    _, y_gs  = signal.step(T_gs,  T=t)

    fig, ax = plt.subplots()
    ax.plot(t, y_ol,  'r--', linewidth=2, label='Open-Loop (no controller)')
    ax.plot(t, y_pid, 'b-',  linewidth=2, label=f'PID  ($K_p$={Kp}, $K_i$={Ki}, $K_d$={Kd})')
    ax.plot(t, y_gs,  'g-',  linewidth=2,
            label=f'Gain-Scheduled Zone 2 ($K_p$={z2["Kp"]}, $K_i$={z2["Ki"]}, $K_d$={z2["Kd"]})')
    ax.axhline(1.0, color='k', linestyle=':', linewidth=1, label='Setpoint = 1.0')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Motor Speed (normalised)')
    ax.set_title('Throttle–Motor Response Comparison')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_ylim(-0.05, 1.5)

    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, 'comparison_plot.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'  ✓ Saved {path}')


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('Generating EV Throttle Control result plots...\n')
    plot_open_loop()
    plot_pid()
    plot_gain_scheduled()
    plot_comparison()
    print('\nAll 4 plots saved to results/')
