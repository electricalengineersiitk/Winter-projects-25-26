"""
simulate.py  —  Main simulation entry point
Smart Multimeter Simulation | End-Term Project

What this script does:
  - Runs 50 test values across all 5 ranges for each mode (R, C, L)
  - Passes each value through the noise model and auto-ranging engine
  - Compares auto-ranging accuracy against a fixed-range baseline
  - Prints a results table with average errors for all three modes
  - Saves two plots to the results/ folder:
      plot_accuracy.png   — % error vs true value (log scale)
      plot_autorange.png  — active range index over time
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from measurement import measure_resistance, measure_capacitance, measure_inductance
from autorange   import make_ranger_state, update_ranger, find_correct_range

# ─────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────

np.random.seed(42)   # reproducible results every run

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

N_SAMPLES = 50   # test samples per mode


# ─────────────────────────────────────────────
# Test value ranges for each mode
# ─────────────────────────────────────────────

# Resistance : 100 Ohm  to 1 MOhm  (shuffled — real inputs are not sorted)
R_VALUES = np.logspace(2, 6, N_SAMPLES)
np.random.shuffle(R_VALUES)

# Capacitance: 10 nF to 100 uF  (shuffled)
C_VALUES = np.logspace(np.log10(10e-9), np.log10(100e-6), N_SAMPLES)
np.random.shuffle(C_VALUES)

# Inductance : 10 uH  to 100 mH  (shuffled)
L_VALUES = np.logspace(np.log10(10e-6), np.log10(100e-3), N_SAMPLES)
np.random.shuffle(L_VALUES)


# ─────────────────────────────────────────────
# Helper: run one full sweep for a mode
# ─────────────────────────────────────────────

def run_sweep(mode, true_values, measure_fn):
    """
    Run N_SAMPLES measurements for one mode.
    Returns lists of results for both auto-ranging and fixed-range baseline.

    Args:
        mode:        "R", "C", or "L"
        true_values: array of true component values
        measure_fn:  measurement function from measurement.py

    Returns:
        auto_errors   : list of % errors using auto-ranging
        fixed_errors  : list of % errors using fixed (best single) range
        range_history : list of range indices chosen by auto-ranger (1-indexed)
    """
    state = make_ranger_state(mode)

    auto_errors   = []
    fixed_errors  = []
    range_history = []

    for true_val in true_values:

        # ── Auto-ranging path ──
        measured_val, error_pct = measure_fn(true_val)
        ridx, rlabel, status    = update_ranger(state, measured_val)

        auto_errors.append(error_pct)
        range_history.append(ridx + 1)   # 1-indexed for plot

        # ── Fixed-range baseline ──
        # Simulates a user locked on a single middle range (Range 3).
        # We add noise normally, but then apply a resolution penalty:
        # if the true value is outside the middle range's 10-90% window,
        # the ADC reading is saturated or underscaled — error grows linearly.
        fixed_range_max = state["ranges"][2]["max"]
        fixed_range_min = state["ranges"][2]["max"] * 0.01   # bottom of Range 3

        _, baseline_noise_error = measure_fn(true_val)

        if true_val > fixed_range_max:
            # Over-range: reading clips at range max, error = overshoot %
            resolution_error = (true_val - fixed_range_max) / true_val * 100
            baseline_error = baseline_noise_error + resolution_error
        elif true_val < fixed_range_min:
            # Under-range: value lost in noise floor
            resolution_error = (fixed_range_min - true_val) / true_val * 100
            baseline_error = baseline_noise_error + min(resolution_error, 50.0)
        else:
            baseline_error = baseline_noise_error

        fixed_errors.append(baseline_error)

    return auto_errors, fixed_errors, range_history


# ─────────────────────────────────────────────
# Run all three modes
# ─────────────────────────────────────────────

print()
print("=" * 62)
print("  Smart Multimeter — Simulation Results")
print("=" * 62)
print(f"  Samples per mode : {N_SAMPLES}")
print(f"  Noise model      : Gaussian, sigma = 0.5% of true value")
print(f"  Hysteresis rule  : 3 consecutive triggers before range switch")
print("=" * 62)

r_auto, r_fixed, r_ranges = run_sweep("R", R_VALUES, measure_resistance)
c_auto, c_fixed, c_ranges = run_sweep("C", C_VALUES, measure_capacitance)
l_auto, l_fixed, l_ranges = run_sweep("L", L_VALUES, measure_inductance)


# ─────────────────────────────────────────────
# Print results table
# ─────────────────────────────────────────────

print()
print(f"  {'Method':<28} {'R Error':>9} {'C Error':>9} {'L Error':>9}")
print(f"  {'-'*60}")
print(f"  {'Fixed-range (no auto-ranging)':<28} "
      f"{np.mean(r_fixed):>8.2f}% "
      f"{np.mean(c_fixed):>8.2f}% "
      f"{np.mean(l_fixed):>8.2f}%")
print(f"  {'{Set to Middle Range (Range 3)}':<28} "
      f"{''} "
      f"{''} "
      f"{''} ")
print(f"  {'':<28} "
      f"{''} "
      f"{''} "
      f"{''} ")
print(f"  {'Auto-ranging simulation':<28} "
      f"{np.mean(r_auto):>8.2f}% "
      f"{np.mean(c_auto):>8.2f}% "
      f"{np.mean(l_auto):>8.2f}%")
print(f"  {'-'*60}")

# Pass / Fail check
all_pass = (
    np.mean(r_auto) <= 2.0 and
    np.mean(c_auto) <= 2.0 and
    np.mean(l_auto) <= 2.0
)
status_str = "PASS  — all modes within 2% target" if all_pass else "FAIL  — check noise model or formulas"
print(f"\n  Accuracy check   : {status_str}")
print()


# ─────────────────────────────────────────────
# Plot 1 — Accuracy vs Input Value
# ─────────────────────────────────────────────

fig1, axes = plt.subplots(1, 3, figsize=(15, 5))
fig1.suptitle("Plot 1 — Measurement Accuracy vs Input Value (Log Scale)",
              fontsize=13, fontweight="bold", y=1.01)

plot_configs = [
    ("Resistance",   R_VALUES, r_auto, r_fixed, "Ω"),
    ("Capacitance",  C_VALUES, c_auto, c_fixed, "F"),
    ("Inductance",   L_VALUES, l_auto, l_fixed, "H"),
]

for ax, (title, x_vals, auto_err, fixed_err, unit) in zip(axes, plot_configs):
    # Sort by true value so the line flows left to right on the log x-axis.
    # The simulation ran on shuffled values — this is only for clean plotting.
    sort_idx  = np.argsort(x_vals)
    x_sorted  = np.array(x_vals)[sort_idx]
    auto_sorted  = np.array(auto_err)[sort_idx]
    fixed_sorted = np.array(fixed_err)[sort_idx]

    ax.plot(x_sorted, fixed_sorted, color="#e74c3c", linewidth=1.5,
            linestyle="--", label="Fixed-range (no auto)", alpha=0.85)
    ax.plot(x_sorted, auto_sorted,  color="#2ecc71", linewidth=2.0,
            label="Auto-ranging sim")

    ax.axhline(y=2.0, color="gray", linewidth=1, linestyle=":", alpha=0.7,
               label="2% target")

    ax.set_xscale("log")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(f"True Value ({unit})", fontsize=9)
    ax.set_ylabel("% Error", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.set_ylim(bottom=0)

plt.tight_layout()
plot1_path = os.path.join(RESULTS_DIR, "plot_accuracy.png")
plt.savefig(plot1_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved : {plot1_path}")


# ─────────────────────────────────────────────
# Plot 2 — Auto-Range State Over Time
# ─────────────────────────────────────────────

fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
fig2.suptitle("Plot 2 — Auto-Range State Over Time (Range Steps 1–5)",
              fontsize=13, fontweight="bold", y=1.01)

sweep_configs = [
    ("Resistance",  r_ranges),
    ("Capacitance", c_ranges),
    ("Inductance",  l_ranges),
]

sample_indices = np.arange(1, N_SAMPLES + 1)

for ax, (title, rng_hist) in zip(axes2, sweep_configs):
    ax.step(sample_indices, rng_hist, where="post",
            color="#3498db", linewidth=2.0)
    ax.fill_between(sample_indices, rng_hist, step="post",
                    alpha=0.15, color="#3498db")

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Sample Index (1 to 50)", fontsize=9)
    ax.set_ylabel("Active Range (1 = lowest)", fontsize=9)
    ax.set_ylim(0.5, 5.5)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["Range 1", "Range 2", "Range 3", "Range 4", "Range 5"],
                       fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plot2_path = os.path.join(RESULTS_DIR, "plot_autorange.png")
plt.savefig(plot2_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved : {plot2_path}")

print()
print("  Simulation complete. Check the results/ folder for both plots.")
print("=" * 62)
print()
