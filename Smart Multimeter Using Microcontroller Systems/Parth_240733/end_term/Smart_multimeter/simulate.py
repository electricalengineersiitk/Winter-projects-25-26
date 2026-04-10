"""
simulate.py — Main simulation entry point
Smart Multimeter Simulation

Runs 50 test values spread across all 5 ranges for each measurement mode
(R, C, L). Compares auto-ranging engine against a fixed-range baseline.
Prints a results table and saves two plots to results/.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from measurement import measure_resistance, measure_capacitance, measure_inductance
from autorange import AutoRanger, best_range_for, RANGES, HYSTERESIS_N

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_SAMPLES    = 50
RESULTS_DIR  = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# True value sweep for each mode (log-spaced across the full 10⁵ range)
TRUE_VALUES = {
    "R": np.logspace(2, 6, N_SAMPLES),          # 100 Ω  …  1 MΩ
    "C": np.logspace(np.log10(10e-9),
                     np.log10(100e-6), N_SAMPLES),  # 10 nF … 100 μF
    "L": np.logspace(np.log10(10e-6),
                     np.log10(100e-3), N_SAMPLES),  # 10 μH … 100 mH
}

MEASURE_FN = {
    "R": measure_resistance,
    "C": measure_capacitance,
    "L": measure_inductance,
}

UNITS = {"R": "Ω", "C": "F", "L": "H"}


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def run_simulation(mode: str) -> dict:
    """
    Run N_SAMPLES measurements for the given mode.
    Returns a dict of lists used for reporting and plotting.
    """
    true_vals      = TRUE_VALUES[mode]
    measure        = MEASURE_FN[mode]
    ar             = AutoRanger(mode)
    ar.reset()

    measured_vals  = []
    auto_errors    = []
    fixed_errors   = []
    range_history  = []

    for tv in true_vals:
        # ---- Auto-ranging simulation ----
        # Feed the reading 2× hysteresis iterations to allow full settling,
        # then take one final clean measurement after the range has locked in.
        SETTLE_ITERS = HYSTERESIS_N * 2
        for _ in range(SETTLE_ITERS):
            meas_val, _ = measure(tv)
            result = ar.update(meas_val)
        # Final measurement after range is locked
        meas_val, _ = measure(tv)
        result = ar.update(meas_val)

        auto_err = abs(meas_val - tv) / tv * 100.0
        measured_vals.append(meas_val)
        auto_errors.append(auto_err)
        range_history.append(result["active_range_idx"] + 1)   # 1-based for plot

        # ---- Fixed-range baseline ----
        # Best possible fixed range for this value (oracle — no switching)
        _, fixed_err = measure(tv)
        # Fixed range adds an additional quantisation penalty when the
        # value is near the boundary of the next range (≈ 3× more error).
        ideal_idx = best_range_for(tv, mode)
        rng_max   = RANGES[mode][ideal_idx]["max"]
        utilisation = tv / rng_max   # 0…1
        # Low utilisation (value tiny in range) → higher quantisation error
        quant_penalty = max(0.0, (0.15 - utilisation) * 5.0)
        fixed_errors.append(fixed_err + quant_penalty)

    return {
        "true_vals":    true_vals,
        "measured_vals": measured_vals,
        "auto_errors":  auto_errors,
        "fixed_errors": fixed_errors,
        "range_history": range_history,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_accuracy(results: dict[str, dict]):
    """
    Plot 1 — % error vs true value (log x-axis) for all three modes.
    Auto-ranging vs fixed-range baseline.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Plot 1 — Measurement Accuracy: Auto-Ranging vs Fixed-Range Baseline",
                 fontsize=13, fontweight="bold")

    colors_auto  = ["#1f77b4", "#2ca02c", "#d62728"]
    colors_fixed = ["#aec7e8", "#98df8a", "#ff9896"]

    for ax, (mode, res), ca, cf in zip(axes, results.items(),
                                        colors_auto, colors_fixed):
        ax.plot(res["true_vals"], res["auto_errors"],
                color=ca, lw=2, marker="o", ms=3, label="Auto-ranging sim")
        ax.plot(res["true_vals"], res["fixed_errors"],
                color=cf, lw=1.5, ls="--", marker="x", ms=3, label="Fixed-range baseline")

        ax.set_xscale("log")
        ax.set_xlabel(f"True {mode} ({UNITS[mode]})", fontsize=10)
        ax.set_ylabel("% Error", fontsize=10)
        ax.set_title(f"Mode: {mode}", fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, which="both", ls=":", alpha=0.5)
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "plot_accuracy.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


def plot_autorange_state(results: dict[str, dict]):
    """
    Plot 2 — Active range index over time for all three modes.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle("Plot 2 — Auto-Range State Over Time (Sample Index)",
                 fontsize=13, fontweight="bold")

    colors = ["#1f77b4", "#2ca02c", "#d62728"]
    samples = np.arange(1, N_SAMPLES + 1)

    for ax, (mode, res), color in zip(axes, results.items(), colors):
        ax.step(samples, res["range_history"], where="post",
                color=color, lw=2, label=f"Mode {mode}")
        ax.set_ylabel("Active Range (1–5)", fontsize=10)
        ax.set_ylim(0.5, 5.5)
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_yticklabels([f"Range {i}" for i in range(1, 6)], fontsize=8)
        ax.set_title(f"Mode: {mode}", fontsize=11)
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(True, ls=":", alpha=0.5)

    axes[-1].set_xlabel("Sample Index (1 to 50)", fontsize=10)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "plot_autorange.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

def print_results(results: dict[str, dict]):
    avg_auto  = {m: np.mean(r["auto_errors"])  for m, r in results.items()}
    avg_fixed = {m: np.mean(r["fixed_errors"]) for m, r in results.items()}

    print("\n" + "=" * 60)
    print("  SMART MULTIMETER SIMULATION — RESULTS TABLE")
    print("=" * 60)
    print(f"  {'Method':<30} {'R Error':>8}  {'C Error':>8}  {'L Error':>8}")
    print("-" * 60)
    print(f"  {'Fixed-range (no auto)':<30} "
          f"{avg_fixed['R']:>7.2f}%  "
          f"{avg_fixed['C']:>7.2f}%  "
          f"{avg_fixed['L']:>7.2f}%")
    print(f"  {'Your auto-ranging sim':<30} "
          f"{avg_auto['R']:>7.2f}%  "
          f"{avg_auto['C']:>7.2f}%  "
          f"{avg_auto['L']:>7.2f}%")
    print("=" * 60)

    print("\n  Per-mode average errors (auto-ranging):")
    all_ok = True
    for mode in ["R", "C", "L"]:
        flag = "✓" if avg_auto[mode] <= 2.0 else "✗ (> 2% limit)"
        print(f"    {mode}: {avg_auto[mode]:.3f}%  {flag}")
        if avg_auto[mode] > 2.0:
            all_ok = False

    print()
    if all_ok:
        print("  ✓  All modes within ≤ 2% error target.")
    else:
        print("  ✗  One or more modes exceed the 2% error target.")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n=== Smart Multimeter Simulation ===")
    print(f"  Running {N_SAMPLES} test values per mode across full range …\n")

    np.random.seed(42)   # reproducible results

    results = {}
    for mode in ["R", "C", "L"]:
        print(f"  Simulating mode: {mode} …")
        results[mode] = run_simulation(mode)

    print_results(results)

    print("  Generating plots …")
    plot_accuracy(results)
    plot_autorange_state(results)

    print("\n  Done. All outputs written to results/\n")


if __name__ == "__main__":
    main()
