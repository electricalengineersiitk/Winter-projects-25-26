import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.interpolate import CubicSpline
import heapq, math

os.makedirs("/content/end_term/results", exist_ok=True)

# ── MAP SETTINGS ──────────────────────────────
MAP_SIZE        = 100
START           = (5, 50)
GOAL            = (95, 50)
OBSTACLE_CENTER = (50, 50)
OBSTACLE_RADIUS = 10
SAFETY_MARGIN   = 5

def is_in_obstacle(x, y, margin=0):
    cx, cy = OBSTACLE_CENTER
    return (x-cx)**2 + (y-cy)**2 <= (OBSTACLE_RADIUS+margin)**2

def build_grid():
    grid = np.ones((MAP_SIZE, MAP_SIZE), dtype=bool)
    for row in range(MAP_SIZE):
        for col in range(MAP_SIZE):
            if is_in_obstacle(col, row, SAFETY_MARGIN):
                grid[row, col] = False
    return grid

# ── A* ────────────────────────────────────────
def _heuristic(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def _neighbours(node, grid):
    x, y = node
    rows, cols = grid.shape
    for dx in (-1,0,1):
        for dy in (-1,0,1):
            if dx==0 and dy==0: continue
            nx, ny = x+dx, y+dy
            if 0<=nx<cols and 0<=ny<rows and grid[ny,nx]:
                yield (nx, ny)

def astar(grid, start, goal):
    open_heap = []
    heapq.heappush(open_heap, (0.0, start))
    came_from = {}
    g_score   = {start: 0.0}
    f_score   = {start: _heuristic(start, goal)}
    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path
        for nb in _neighbours(current, grid):
            tg = g_score[current] + math.hypot(nb[0]-current[0], nb[1]-current[1])
            if tg < g_score.get(nb, float('inf')):
                came_from[nb] = current
                g_score[nb]   = tg
                f_score[nb]   = tg + _heuristic(nb, goal)
                heapq.heappush(open_heap, (f_score[nb], nb))
    return []

def plan_path():
    grid     = build_grid()
    raw      = astar(grid, START, GOAL)
    thinned  = raw[::5]
    if thinned[-1] != raw[-1]:
        thinned.append(raw[-1])
    print(f"Raw: {len(raw)}  Thinned: {len(thinned)}")
    return thinned

# ── TRAJECTORY ────────────────────────────────
def make_trajectory(waypoints, v_max, n=500):
    pts  = np.array(waypoints, dtype=float)
    diff = np.diff(pts, axis=0)
    seg  = np.hypot(diff[:,0], diff[:,1])
    s    = np.concatenate([[0.0], np.cumsum(seg)])
    cs_x = CubicSpline(s, pts[:,0])
    cs_y = CubicSpline(s, pts[:,1])
    sv   = np.linspace(0, s[-1], n)
    x    = cs_x(sv); y = cs_y(sv)
    dt   = np.diff(sv) / v_max
    t    = np.concatenate([[0.0], np.cumsum(dt)])
    spd  = np.hypot(cs_x(sv,1), cs_y(sv,1)) * v_max
    acc  = np.hypot(cs_x(sv,2), cs_y(sv,2)) * v_max**2
    try:    eng = float(np.trapezoid(spd**2, t))
    except: eng = float(np.trapz(spd**2, t))
    return {'x':x,'y':y,'t':t,'speed':spd,'accel':acc,
            'total_time':t[-1],'total_dist':s[-1],'energy':eng}

# ── FORMATION ─────────────────────────────────
V_OFFSETS    = np.array([[-4,4],[-2,2],[0,0],[2,2],[4,4]], dtype=float)
N_DRONES     = 5
DRONE_COLORS = ['#e63946','#f4a261','#2a9d8f','#457b9d','#a8dadc']

def get_positions(cx, cy):
    c = np.column_stack([np.atleast_1d(cx), np.atleast_1d(cy)])
    return c[:, np.newaxis, :] + V_OFFSETS[np.newaxis, :, :]

# ══════════════════════════════════════════════
print("[1/4] Planning path...")
waypoints = plan_path()

print("[2/4] Making trajectories...")
tt = make_trajectory(waypoints, 12.0)
te = make_trajectory(waypoints,  4.0)

print("[3/4] Saving plots...")

# PATH PLOT
fig, ax = plt.subplots(figsize=(8,8))
ax.add_patch(plt.Circle(OBSTACLE_CENTER, OBSTACLE_RADIUS+SAFETY_MARGIN,
                        color='pink', alpha=0.35, label='Safety margin'))
ax.add_patch(plt.Circle(OBSTACLE_CENTER, OBSTACLE_RADIUS,
                        color='salmon', alpha=0.8, label='Obstacle'))
ax.plot(*START,'go',markersize=10,zorder=6,label='Start')
ax.plot(*GOAL,marker='*',color='orange',markersize=14,zorder=6,label='Goal')
xs,ys = zip(*waypoints)
ax.plot(xs, ys,'b--o',lw=1.5,ms=4,label='A* waypoints',zorder=4)
ax.plot(tt['x'],tt['y'],'b-',lw=2.5,label='Smooth path',zorder=3,alpha=0.7)
ax.set_xlim(0,100); ax.set_ylim(0,100); ax.set_aspect('equal')
ax.set_xlabel('X'); ax.set_ylabel('Y')
ax.set_title('Collision-Free A* Path (V-Formation UAVs)')
ax.legend(); ax.grid(True,linestyle='--',alpha=0.4)
plt.savefig("/content/end_term/results/path_plot.png", dpi=150, bbox_inches='tight')
plt.show()
print("  ✅ path_plot.png")

# TRAJECTORY COMPARISON
fig, axes = plt.subplots(1,2,figsize=(14,5))
axes[0].plot(tt['t'],tt['speed'],'b-', lw=2,label='Min-time (v=12)')
axes[0].plot(te['t'],te['speed'],'g--',lw=2,label='Min-energy (v=4)')
axes[0].set_xlabel('Time [s]'); axes[0].set_ylabel('Speed [u/s]')
axes[0].set_title('Speed vs Time'); axes[0].legend()
axes[0].grid(True,linestyle='--',alpha=0.5)
axes[1].plot(tt['t'],tt['accel'],'b-', lw=2,label='Min-time')
axes[1].plot(te['t'],te['accel'],'g--',lw=2,label='Min-energy')
axes[1].set_xlabel('Time [s]'); axes[1].set_ylabel('Acceleration [u/s²]')
axes[1].set_title('Acceleration vs Time'); axes[1].legend()
axes[1].grid(True,linestyle='--',alpha=0.5)
fig.suptitle('Trajectory Comparison: Min-Time vs Min-Energy',fontsize=13)
plt.tight_layout()
plt.savefig("/content/end_term/results/trajectory_comparison.png",dpi=150,bbox_inches='tight')
plt.show()
print("  ✅ trajectory_comparison.png")

print("[4/4] Building animation (1-2 min)...")
pos_t = get_positions(tt['x'], tt['y'])
pos_e = get_positions(te['x'], te['y'])
NF    = 120
it    = np.linspace(0, len(tt['x'])-1, NF).astype(int)
ie    = np.linspace(0, len(te['x'])-1, NF).astype(int)
ptf   = pos_t[it]; pef = pos_e[ie]

fig2,(aL,aR) = plt.subplots(1,2,figsize=(16,7))
fig2.patch.set_facecolor('#0d1117')

def setup(ax, title):
    ax.set_facecolor('#0d1117')
    ax.add_patch(plt.Circle(OBSTACLE_CENTER,OBSTACLE_RADIUS+SAFETY_MARGIN,color='#ff6b6b',alpha=0.15))
    ax.add_patch(plt.Circle(OBSTACLE_CENTER,OBSTACLE_RADIUS,color='#e63946',alpha=0.55))
    ax.plot(*START,'o',color='#06d6a0',ms=11,zorder=7)
    ax.plot(*GOAL,'*',color='#ffd166',ms=16,zorder=7)
    ax.plot(tt['x'],tt['y'],color='white',lw=0.8,alpha=0.15,zorder=2)
    ax.set_xlim(-2,102); ax.set_ylim(-2,102); ax.set_aspect('equal')
    ax.set_title(title,color='white',fontsize=12)
    ax.tick_params(colors='white')
    for sp in ax.spines.values(): sp.set_edgecolor('#444')
    ax.set_xlabel('X',color='white'); ax.set_ylabel('Y',color='white')
    ax.grid(True,linestyle='--',alpha=0.15,color='white')

setup(aL,"Min-Time Trajectory")
setup(aR,"Min-Energy Trajectory")

TRAIL=30
scL=[aL.plot([],[],'o',color=DRONE_COLORS[i],ms=9,zorder=8)[0] for i in range(N_DRONES)]
scR=[aR.plot([],[],'o',color=DRONE_COLORS[i],ms=9,zorder=8)[0] for i in range(N_DRONES)]
trL=[aL.plot([],[],'-',color=DRONE_COLORS[i],lw=1.2,alpha=0.5,zorder=5)[0] for i in range(N_DRONES)]
trR=[aR.plot([],[],'-',color=DRONE_COLORS[i],lw=1.2,alpha=0.5,zorder=5)[0] for i in range(N_DRONES)]
lnL,=aL.plot([],[],'--',color='white',lw=0.9,alpha=0.5,zorder=6)
lnR,=aR.plot([],[],'--',color='white',lw=0.9,alpha=0.5,zorder=6)
txL=aL.text(2,96,'',color='white',fontsize=9,va='top')
txR=aR.text(2,96,'',color='white',fontsize=9,va='top')
pch=[mpatches.Patch(color=DRONE_COLORS[i],label=f'Drone {i}') for i in range(N_DRONES)]
pch+=[mpatches.Patch(color='#06d6a0',label='Start'),
      mpatches.Patch(color='#ffd166',label='Goal'),
      mpatches.Patch(color='#e63946',label='Obstacle')]
fig2.legend(handles=pch,loc='lower center',ncol=8,fontsize=8,
            facecolor='#1a1a2e',labelcolor='white',framealpha=0.8,
            bbox_to_anchor=(0.5,-0.01))
fig2.tight_layout(pad=2)

def update(frame):
    pL=ptf[frame]; pR=pef[frame]
    s=max(0,frame-TRAIL)
    for i in range(N_DRONES):
        scL[i].set_data([pL[i,0]],[pL[i,1]])
        scR[i].set_data([pR[i,0]],[pR[i,1]])
        trL[i].set_data(ptf[s:frame+1,i,0],ptf[s:frame+1,i,1])
        trR[i].set_data(pef[s:frame+1,i,0],pef[s:frame+1,i,1])
    lnL.set_data(pL[:,0],pL[:,1])
    lnR.set_data(pR[:,0],pR[:,1])
    txL.set_text(f"t={tt['t'][it[frame]]:.1f}s")
    txR.set_text(f"t={te['t'][ie[frame]]:.1f}s")
    return scL+scR+trL+trR+[lnL,lnR,txL,txR]

anim=FuncAnimation(fig2,update,frames=NF,interval=60,blit=True)
anim.save("/content/end_term/results/formation_animation.gif",
          writer=PillowWriter(fps=18),dpi=90)
plt.close(fig2)
print("  ✅ formation_animation.gif")

print("\n✅ DONE! Files check:")
print(os.listdir("/content/end_term/results"))
print(f"\nMin-time   : {tt['total_time']:.1f}s")
print(f"Min-energy : {te['total_time']:.1f}s")
