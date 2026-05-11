"""
simulate.py — Main simulation entry point
Smart Multimeter Simulation | End-Term Project

Runs 50 test values across all 5 ranges for R, C, and L.
Prints a results table and generates two plots saved to results/.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for script use
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Ensure results/ directory exists
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Import project modules
# ---------------------------------------------------------------------------
from measurement import measure_resistance, measure_capacitance, measure_inductance
from autorange import AutoRanger, fixed_range, RANGE_MAP

# ---------------------------------------------------------------------------
# Test value generation — 10 values per range, 5 ranges = 50 per mode
# ---------------------------------------------------------------------------

def _make_test_values(mode: str) -> np.ndarray:
    """Return 50 true values spread across all 5 ranges (log-spaced within each)."""
    ranges = RANGE_MAP[mode]
    values = []
    for r in ranges:
        lo = r.max_val / 10.0
        hi = r.max_val
        values.append(np.logspace(np.log10(lo), np.log10(hi), 10, endpoint=True))
    return np.concatenate(values)


# ---------------------------------------------------------------------------
# Single-mode simulation
# ---------------------------------------------------------------------------

def simulate_mode(mode: str):
    """
    Run 50-sample simulation for one mode.

    Returns:
        dict with keys: true_vals, measured_auto, measured_fixed,
                        errors_auto, errors_fixed, range_states_auto
    """
    true_vals = _make_test_values(mode)
    measure_fn = {"R": measure_resistance, "C": measure_capacitance, "L": measure_inductance}[mode]

    ranger = AutoRanger(mode)
    ranger.reset()

    measured_auto = []
    measured_fixed = []
    errors_auto = []
    errors_fixed = []
    range_states = []

    for tv in true_vals:
        # Auto-ranging measurement
        m_auto, e_auto = measure_fn(tv)
        range_idx, _ = ranger.update(m_auto)
        measured_auto.append(m_auto)
        errors_auto.append(e_auto)
        range_states.append(range_idx)

        # Fixed-range baseline — same physical measurement, different range tracking
        m_fixed, e_fixed = measure_fn(tv)
        measured_fixed.append(m_fixed)
        errors_fixed.append(e_fixed)

    return {
        "true_vals": true_vals,
        "measured_auto": np.array(measured_auto),
        "measured_fixed": np.array(measured_fixed),
        "errors_auto": np.array(errors_auto),
        "errors_fixed": np.array(errors_fixed),
        "range_states": np.array(range_states),
    }


# ---------------------------------------------------------------------------
# Print results table
# ---------------------------------------------------------------------------

def print_results_table(r_res, c_res, l_res):
    row_fixed  = [np.mean(r_res["errors_fixed"]),  np.mean(c_res["errors_fixed"]),  np.mean(l_res["errors_fixed"])]
    row_auto   = [np.mean(r_res["errors_auto"]),   np.mean(c_res["errors_auto"]),   np.mean(l_res["errors_auto"])]

    header = f"{'Method':<30} | {'R Error':>10} | {'C Error':>10} | {'L Error':>10}"
    sep    = "-" * len(header)
    print("\n" + sep)
    print("  SIMULATION RESULTS — Average % Error over 50 samples")
    print(sep)
    print(header)
    print(sep)
    print(f"{'Fixed-range (no auto)  [baseline]':<30} | {row_fixed[0]:>9.2f}% | {row_fixed[1]:>9.2f}% | {row_fixed[2]:>9.2f}%")
    print(f"{'Auto-ranging simulation':<30} | {row_auto[0]:>9.2f}% | {row_auto[1]:>9.2f}% | {row_auto[2]:>9.2f}%")
    print(sep + "\n")

    # Per-sample table (abbreviated — first 10 rows per mode)
    for label, res, unit, scale in [
        ("RESISTANCE (Ω)",    r_res, "Ω",  1),
        ("CAPACITANCE (nF)",  c_res, "nF", 1e9),
        ("INDUCTANCE (mH)",   l_res, "mH", 1e3),
    ]:
        print(f"\n  {label} — first 10 samples")
        print(f"  {'True':>12} | {'Measured':>12} | {'Error':>8} | {'Range':>6}")
        print("  " + "-" * 48)
        for i in range(10):
            tv  = res["true_vals"][i] * scale
            mv  = res["measured_auto"][i] * scale
            err = res["errors_auto"][i]
            rng = res["range_states"][i]
            print(f"  {tv:>12.4g} | {mv:>12.4g} | {err:>7.3f}% | {rng:>6}")


# ---------------------------------------------------------------------------
# Plot 1 — Accuracy vs Input Value
# ---------------------------------------------------------------------------

def plot_accuracy(r_res, c_res, l_res):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Plot 1 — Accuracy vs Input Value (Auto-Range vs Fixed-Range)", fontsize=13, fontweight="bold")

    configs = [
        ("Resistance",  r_res, "Ω",  1,   "#2196F3"),
        ("Capacitance", c_res, "F",  1,   "#4CAF50"),
        ("Inductance",  l_res, "H",  1,   "#FF5722"),
    ]

    for ax, (title, res, unit, scale, color) in zip(axes, configs):
        tv = res["true_vals"] * scale
        ax.plot(tv, res["errors_auto"],  color=color,   lw=2,   label="Auto-ranging", zorder=3)
        ax.plot(tv, res["errors_fixed"], color="gray",  lw=1.5, linestyle="--", label="Fixed-range baseline", zorder=2)
        ax.axhline(2.0, color="red", lw=1, linestyle=":", alpha=0.7, label="2% threshold")
        ax.set_xscale("log")
        ax.set_xlabel(f"True Value ({unit})", fontsize=10)
        ax.set_ylabel("% Measurement Error", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "plot_accuracy.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 2 — Auto-Range State Over Time
# ---------------------------------------------------------------------------

def plot_autorange(r_res, c_res, l_res):
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=False)
    fig.suptitle("Plot 2 — Auto-Range State Over Time (50 Samples)", fontsize=13, fontweight="bold")

    configs = [
        ("Resistance",  r_res, "#2196F3"),
        ("Capacitance", c_res, "#4CAF50"),
        ("Inductance",  l_res, "#FF5722"),
    ]

    for ax, (title, res, color) in zip(axes, configs):
        samples = np.arange(1, len(res["range_states"]) + 1)
        ax.step(samples, res["range_states"], where="post", color=color, lw=2)
        ax.fill_between(samples, res["range_states"], step="post", alpha=0.15, color=color)
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_yticklabels([f"Range {i}" for i in range(1, 6)], fontsize=9)
        ax.set_ylabel("Active Range", fontsize=10)
        ax.set_title(f"{title} — Range Transitions", fontsize=10, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_xlim(1, len(samples))

    axes[-1].set_xlabel("Test Sample Index (1 → 50, sweeping full scale)", fontsize=10)
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "plot_autorange.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Smart Multimeter Simulation — End-Term Project")
    print("=" * 60)
    print("\nRunning 50-sample sweep for each mode (R, C, L)...\n")

    r_res = simulate_mode("R")
    c_res = simulate_mode("C")
    l_res = simulate_mode("L")

    print("Simulation complete. Printing results...\n")
    print_results_table(r_res, c_res, l_res)

    print("Generating plots...")
    plot_accuracy(r_res, c_res, l_res)
    plot_autorange(r_res, c_res, l_res)

    print("\nDone. All outputs saved to results/\n")
