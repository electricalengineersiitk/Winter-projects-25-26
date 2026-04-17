import numpy as np
from scipy.interpolate import CubicSpline

FAST_SPEED  = 8.0   
SLOW_SPEED  = 3.0   

DT = 0.1  

def make_trajectory(path, speed, name):
    pts = np.array(path, dtype=float)
    xs  = pts[:, 0]
    ys  = pts[:, 1]

    diffs = np.diff(pts, axis=0)
    seg_lengths = np.hypot(diffs[:, 0], diffs[:, 1])
    arc_length  = np.concatenate([[0], np.cumsum(seg_lengths)])
    total_dist  = arc_length[-1]

    cs_x = CubicSpline(arc_length, xs)
    cs_y = CubicSpline(arc_length, ys)

    total_time = total_dist / speed
    t = np.arange(0, total_time + DT, DT)

    s = speed * t
    s = np.clip(s, 0, total_dist)

    x  = cs_x(s)
    y  = cs_y(s)
    vx = cs_x(s, 1) * speed  
    vy = cs_y(s, 1) * speed

    ax = np.gradient(vx, t)
    ay = np.gradient(vy, t)

    accel = np.hypot(ax, ay)
    speed_arr = np.hypot(vx, vy)

    print(f"[{name}] total time: {t[-1]:.1f}s, dist: {total_dist:.1f} units")
    return {
        "name":   name,
        "t":      t,
        "x":      x,
        "y":      y,
        "vx":     vx,
        "vy":     vy,
        "ax":     ax,
        "ay":     ay,
        "speed":  speed_arr,
        "accel":  accel,
        "total_time": t[-1],
        "total_dist": total_dist,
        "energy": float(np.trapezoid(accel**2, t) if hasattr(np, "trapezoid") else np.trapz(accel**2, t)),
    }

def generate_trajectories(path):
    fast = make_trajectory(path, FAST_SPEED, "Min-Time")
    slow = make_trajectory(path, SLOW_SPEED, "Min-Energy")
    return fast, slow

if __name__ == "__main__":
    import os, matplotlib.pyplot as plt
    from path_planner import plan_path
    os.makedirs("results", exist_ok=True)

    path = plan_path()
    fast, slow = generate_trajectories(path)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(fast["t"], fast["speed"],  label="Min-Time",   color="orange")
    axes[0].plot(slow["t"], slow["speed"],  label="Min-Energy", color="green")
    axes[0].set_title("Speed vs Time")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Speed (units/s)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(fast["t"], fast["accel"],  label="Min-Time",   color="orange")
    axes[1].plot(slow["t"], slow["accel"],  label="Min-Energy", color="green")
    axes[1].set_title("Acceleration vs Time")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Accel (units/s^2)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Trajectory Comparison: Min-Time vs Min-Energy")
    plt.tight_layout()
    plt.savefig("results/trajectory_comparison.png", dpi=100)
    plt.show()
    print("Saved: results/trajectory_comparison.png")
