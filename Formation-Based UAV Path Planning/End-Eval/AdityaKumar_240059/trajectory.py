import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import os


MIN_TIME_SPEED   = 18.0    
MIN_ENERGY_SPEED =  6.0   
DT               =  0.05   

def _build_spline(waypoints):
    wps = np.asarray(waypoints, dtype=float)
    diffs = np.diff(wps, axis=0)
    seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
    cumlen = np.concatenate([[0.0], np.cumsum(seg_lens)])

    cs_x = CubicSpline(cumlen, wps[:, 0])
    cs_y = CubicSpline(cumlen, wps[:, 1])
    return cs_x, cs_y, cumlen[-1]


def _sample_trajectory(cs_x, cs_y, arc_len, speed, dt=DT):
    total_time = arc_len / speed
    t_arr = np.arange(0, total_time + dt, dt)

    s_arr = speed * t_arr
    s_arr = np.clip(s_arr, 0, arc_len)

    x  = cs_x(s_arr)
    y  = cs_y(s_arr)
    vx = cs_x(s_arr, 1) * speed   
    vy = cs_y(s_arr, 1) * speed

    ax = np.gradient(vx, t_arr)
    ay = np.gradient(vy, t_arr)

    speed_arr = np.hypot(vx, vy)
    accel_arr = np.hypot(ax, ay)

    return {
        'x': x, 'y': y, 't': t_arr,
        'vx': vx, 'vy': vy,
        'ax': ax, 'ay': ay,
        'speed': speed_arr,
        'accel': accel_arr,
    }


def _min_energy_profile(cs_x, cs_y, arc_len, dt=DT):
    peak_speed   = MIN_ENERGY_SPEED
    ramp_frac    = 0.25         
    ramp_len     = ramp_frac * arc_len
    cruise_len   = arc_len - 2 * ramp_len

    n_steps = int(arc_len / (peak_speed * dt)) + 200
    s_fine  = np.linspace(0, arc_len, n_steps)

    v_profile = np.empty_like(s_fine)
    for i, s in enumerate(s_fine):
        if s < ramp_len:
            v_profile[i] = peak_speed * (s / ramp_len)
        elif s < ramp_len + cruise_len:
            v_profile[i] = peak_speed
        else:
            remaining = arc_len - s
            v_profile[i] = peak_speed * max(remaining / ramp_len, 0.05)

    ds = np.diff(s_fine, prepend=s_fine[0])
    t_cumul = np.concatenate([[0.0], np.cumsum(ds[1:] / v_profile[1:])])

    x  = cs_x(s_fine)
    y  = cs_y(s_fine)

    t_uniform = np.arange(0, t_cumul[-1] + dt, dt)
    x_u  = np.interp(t_uniform, t_cumul, x)
    y_u  = np.interp(t_uniform, t_cumul, y)
    vx_u = np.gradient(x_u, t_uniform)
    vy_u = np.gradient(y_u, t_uniform)
    ax_u = np.gradient(vx_u, t_uniform)
    ay_u = np.gradient(vy_u, t_uniform)

    return {
        'x': x_u, 'y': y_u, 't': t_uniform,
        'vx': vx_u, 'vy': vy_u,
        'ax': ax_u, 'ay': ay_u,
        'speed': np.hypot(vx_u, vy_u),
        'accel': np.hypot(ax_u, ay_u),
    }

def generate_trajectories(waypoints):
    cs_x, cs_y, arc_len = _build_spline(waypoints)
    print(f"[trajectory] Arc length : {arc_len:.2f} units")

    traj_time   = _sample_trajectory(cs_x, cs_y, arc_len,
                                     speed=MIN_TIME_SPEED)
    traj_energy = _min_energy_profile(cs_x, cs_y, arc_len)

    _print_summary(traj_time,   "Min-Time  ")
    _print_summary(traj_energy, "Min-Energy")

    return traj_time, traj_energy


def _print_summary(traj, label):
    dist = np.sum(np.hypot(np.diff(traj['x']), np.diff(traj['y'])))
    energy = np.trapezoid(traj['accel'] ** 2, traj['t'])   
    print(f"  [{label}] total_time={traj['t'][-1]:.2f}s  "
          f"dist={dist:.2f}u  energy_proxy={energy:.2f}")


def compute_metrics(traj_time, traj_energy):
    def _energy(t):
        return np.trapezoid(t['accel'] ** 2, t['t'])
    return {
        'time_duration'  : traj_time['t'][-1],
        'energy_duration': traj_energy['t'][-1],
        'time_energy'    : _energy(traj_time),
        'energy_energy'  : _energy(traj_energy),
        'time_max_speed' : traj_time['speed'].max(),
        'energy_max_speed': traj_energy['speed'].max(),
    }

def plot_trajectory_comparison(traj_time, traj_energy, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Trajectory Comparison: Min-Time vs Min-Energy',
                 fontsize=13, fontweight='bold')

    ax = axes[0]
    ax.plot(traj_time['t'],   traj_time['speed'],
            color='#e74c3c', linewidth=2, label='Min-Time')
    ax.plot(traj_energy['t'], traj_energy['speed'],
            color='#2980b9', linewidth=2, label='Min-Energy')
    ax.set_title('Speed vs Time')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Speed (units/s)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(traj_time['t'],   traj_time['accel'],
            color='#e74c3c', linewidth=2, label='Min-Time')
    ax.plot(traj_energy['t'], traj_energy['accel'],
            color='#2980b9', linewidth=2, label='Min-Energy')
    ax.set_title('Acceleration vs Time')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Acceleration (units/s²)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"[trajectory] Comparison plot saved → {save_path}")
    plt.close(fig)
    return fig

if __name__ == '__main__':
    from path_planner import plan_path
    wp = plan_path()
    t_traj, e_traj = generate_trajectories(wp)
    plot_trajectory_comparison(
        t_traj, e_traj,
        save_path=os.path.join('results', 'trajectory_comparison.png')
    )
    print("Trajectory generation complete.")