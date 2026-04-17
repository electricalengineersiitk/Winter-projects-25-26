import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import os
import map_setup as ms
import path_planner as pp
from trajectory import generate_flight_paths

def compute_energy(cs, duration):
    t = np.linspace(0, duration, 200)
    accel = np.linalg.norm(cs.derivative(2)(t), axis=1)
    # Changed np.trapz to np.trapezoid for compatibility with newer NumPy versions
    energy = np.trapezoid(accel**2, t) 
    return energy

def plot_analysis_graphs(cs_t, dur_t, cs_e, dur_e):
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), facecolor='#101010')
    for cs, dur, label, col in zip([cs_t, cs_e], [dur_t, dur_e], ["Min-Time", "Min-Energy"], ['#00ffcc', '#ff00ff']):
        t = np.linspace(0, dur, 200)
        speed = np.linalg.norm(cs.derivative(1)(t), axis=1)
        accel = np.linalg.norm(cs.derivative(2)(t), axis=1)
        ax1.plot(t, speed, color=col, label=label, linewidth=2.5)
        ax2.plot(t, accel, color=col, label=label, linewidth=2.5)
    
    ax1.set_title("Velocity Profile (m/s)", color='white', pad=15)
    ax2.set_title("Acceleration Profile (m/s²)", color='white', pad=15)
    ax1.legend(); ax2.legend(); ax1.grid(True, alpha=0.1); ax2.grid(True, alpha=0.1)
    plt.savefig('results/trajectory_comparison.png')

def plot_full_rrt_tree(nodes, path_wps):
    plt.style.use('dark_background')
    plt.figure(figsize=(10, 10), facecolor='#101010')
    ax = plt.gca()
    ax.set_facecolor('#101010')
    
    for n in nodes:
        if n.parent:
            plt.plot([n.x, n.parent.x], [n.y, n.parent.y], color='#333333', lw=0.6, alpha=0.7)
    
    pts = np.array(path_wps)
    plt.plot(pts[:,0], pts[:,1], color='#00aaff', lw=3, label="Optimal Path", zorder=4)
    
    # Square Obstacle
    core_sq = patches.Rectangle((ms.CORE_L, ms.CORE_B), ms.HAZARD_WIDTH, ms.HAZARD_WIDTH, facecolor='#252525', edgecolor='#444444', zorder=5)
    ax.add_patch(core_sq)
    
    # START AND END MARKERS
    plt.scatter(ms.MISSION_START[0], ms.MISSION_START[1], color='#00ff00', s=150, marker='D', label='Launch', zorder=6)
    plt.scatter(ms.MISSION_GOAL[0], ms.MISSION_GOAL[1], color='#ff0000', s=200, marker='X', label='Target', zorder=6)
    
    plt.title("RRT Exploration Map", color='white', size=14)
    plt.legend(loc='upper left')
    plt.grid(True, color='#222222', alpha=0.5)
    plt.savefig('results/path_plot.png')

def main():
    os.makedirs('results', exist_ok=True)
    tree, goal = pp.run_mission_planner()
    
    if goal is None:
        raise RuntimeError("Path planning failed. Try increasing iterations.")

    path = pp.extract_waypoints(goal)
    plot_full_rrt_tree(tree, path)
    
    cs_t, dur_t, cs_e, dur_e = generate_flight_paths(path)
    plot_analysis_graphs(cs_t, dur_t, cs_e, dur_e)

    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8), facecolor='#080808')
    
    def update(frame):
        for ax, cs, dur, label, col_main in zip([ax1, ax2], [cs_t, cs_e], [dur_t, dur_e], ["MIN-TIME MODE", "MIN-ENERGY MODE"], ['#00ff88', '#00bcff']):
            ax.clear()
            ax.set_xlim(-10, ms.MAP_LIMIT+10); ax.set_ylim(-10, ms.MAP_LIMIT+10)
            ax.set_facecolor('#0d0d0d')

            # Obstacle
            core_sq = patches.Rectangle((ms.CORE_L, ms.CORE_B), ms.HAZARD_WIDTH, ms.HAZARD_WIDTH, facecolor='#1a1a1a', edgecolor='#333333')
            ax.add_patch(core_sq)
            
            # Bright Launch and Target Points
            ax.scatter(ms.MISSION_START[0], ms.MISSION_START[1], color='#00ff00', s=100, marker='o', alpha=0.6)
            ax.scatter(ms.MISSION_GOAL[0], ms.MISSION_GOAL[1], color='#ff3333', s=150, marker='*', alpha=0.8)
            
            t_curr = (frame / 100) * dur
            vel_vec = cs.derivative(1)(t_curr)
            speed_val = np.linalg.norm(vel_vec)
            
            center_pos = cs(t_curr)
            uav_positions = center_pos + ms.uav_group.drone_offsets
            
            # Plot drones with a slight glow effect
            ax.scatter(uav_positions[:,0], uav_positions[:,1], color=col_main, s=40, edgecolors='white', linewidth=0.8, zorder=10)
            
            ax.set_title(f"{label}\nTime: {t_curr:.2f}s | Speed: {speed_val:.1f}m/s", color='white', fontsize=11)
            ax.grid(True, color='#1a1a1a', linestyle='--')

    ani = animation.FuncAnimation(fig, update, frames=101, interval=80)
    ani.save('results/formation_animation.gif', writer='pillow', fps=15)
    
    # Mission Summary [cite: 76]
    energy_t = compute_energy(cs_t, dur_t)
    energy_e = compute_energy(cs_e, dur_e)
    
    print("\n--- Mission Summary ---")
    print(f"Min-Time Trajectory: Duration = {dur_t:.2f}s | Energy = {energy_t:.2f}")
    print(f"Min-Energy Trajectory: Duration = {dur_e:.2f}s | Energy = {energy_e:.2f}")
    
    plt.show()

if __name__ == "__main__": 
    main()