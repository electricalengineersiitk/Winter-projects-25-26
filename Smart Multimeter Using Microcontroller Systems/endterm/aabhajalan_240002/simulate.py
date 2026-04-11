"""
simulate.py — Main simulation entry point for the Smart Multimeter.

Runs 50 test values spread across all 5 ranges for each measurement mode
(R, C, L), feeds them through the auto-ranging engine, prints a results
table, and saves two plots to results/.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from measurement import measure_resistance, measure_capacitance, measure_inductance
from autorange import AutoRanger

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Test value ranges for each mode  (50 log-spaced values across 10⁵ range)
# ---------------------------------------------------------------------------
TEST_VALUES = {
    "R": np.logspace(2, 6, 50),        # 100 Ω  → 1 MΩ
    "C": np.logspace(-8, -4, 50),      # 10 nF  → 100 µF
    "L": np.logspace(-5, -1, 50),      # 10 µH  → 100 mH
}

MEASURE_FN = {
    "R": measure_resistance,
    "C": measure_capacitance,
    "L": measure_inductance,
}

UNITS = {"R": "Ω", "C": "F", "L": "H"}

# ---------------------------------------------------------------------------
# Fixed-range baseline: always uses mid-range, so error explodes at extremes
# ---------------------------------------------------------------------------
FIXED_RANGE_MAX = {
    "R": 10_000,     # Range 3 midpoint
    "C": 1e-6,
    "L": 1e-3,
}


def run_fixed_baseline(mode: str, true_values: np.ndarray) -> np.ndarray:
    """Return per-sample % error for a fixed-range (no auto) simulation."""
    errors = []
    measure = MEASURE_FN[mode]
    for tv in true_values:
        measured, _ = measure(tv)
        # Clamp measured to fixed range max to simulate saturation
        measured_clamped = min(measured, FIXED_RANGE_MAX[mode])
        err = abs(measured_clamped - tv) / tv * 100
        errors.append(err)
    return np.array(errors)


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------
def run_simulation(mode: str) -> dict:
    """
    Run 50-sample simulation for one mode.

    Returns dict with:
        true_vals, measured_vals, errors, ranges, avg_error
    """
    true_vals = TEST_VALUES[mode]
    measure = MEASURE_FN[mode]
    ar = AutoRanger(mode)

    measured_vals, errors, ranges, statuses = [], [], [], []

    for tv in true_vals:
        measured, error_pct = measure(tv)
        result = ar.process(measured)
        measured_vals.append(measured)
        errors.append(error_pct)
        ranges.append(result["range"])
        statuses.append(result["status"])

    return {
        "true_vals":    np.array(true_vals),
        "measured_vals":np.array(measured_vals),
        "errors":       np.array(errors),
        "ranges":       np.array(ranges),
        "statuses":     statuses,
        "avg_error":    float(np.mean(errors)),
    }


# ---------------------------------------------------------------------------
# Plot 1 — Accuracy vs Input Value
# ---------------------------------------------------------------------------
def plot_accuracy(results: dict[str, dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Plot 1 — Accuracy vs Input Value (Auto-Ranging vs Fixed Range)", fontsize=13)

    for ax, (mode, data) in zip(axes, results.items()):
        true_vals = data["true_vals"]
        auto_errors = data["errors"]
        fixed_errors = run_fixed_baseline(mode, true_vals)

        ax.plot(true_vals, auto_errors,  "b-o", markersize=3, label="Auto-ranging sim")
        ax.plot(true_vals, fixed_errors, "r--s", markersize=3, label="Fixed-range baseline")
        ax.axhline(2.0, color="green", linestyle=":", linewidth=1.5, label="2 % target")

        ax.set_xscale("log")
        ax.set_title(f"{mode} — Measurement Error")
        ax.set_xlabel(f"True {mode} ({UNITS[mode]}) — log scale")
        ax.set_ylabel("% Error")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "plot_accuracy.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  ✓ Saved {out_path}")


# ---------------------------------------------------------------------------
# Plot 2 — Auto-Range State Over Time
# ---------------------------------------------------------------------------
def plot_autorange(results: dict[str, dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Plot 2 — Auto-Range State Over Time (Range 1→5 as Input Sweeps)", fontsize=13)

    for ax, (mode, data) in zip(axes, results.items()):
        sample_idx = np.arange(1, len(data["ranges"]) + 1)
        ax.step(sample_idx, data["ranges"], where="post", color="steelblue", linewidth=2)
        ax.scatter(sample_idx, data["ranges"], s=15, color="steelblue", zorder=3)

        ax.set_title(f"{mode} — Active Range vs Sample Index")
        ax.set_xlabel("Sample index (1–50)")
        ax.set_ylabel("Active range (1–5)")
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_ylim(0.5, 5.5)
        ax.set_xlim(0, 51)
        ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "plot_autorange.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  ✓ Saved {out_path}")


# ---------------------------------------------------------------------------
# Results table printer
# ---------------------------------------------------------------------------
def print_results_table(results: dict[str, dict]) -> None:
    divider = "─" * 72
    print(f"\n{divider}")
    print(f"{'SMART MULTIMETER — SIMULATION RESULTS':^72}")
    print(divider)

    header = f"{'Mode':<6} {'True Value':>14} {'Measured':>14} {'Range':>6} {'Error %':>8} {'Status':<14}"
    print(header)
    print("─" * 72)

    for mode, data in results.items():
        for i in range(len(data["true_vals"])):
            tv  = data["true_vals"][i]
            mv  = data["measured_vals"][i]
            rng = data["ranges"][i]
            err = data["errors"][i]
            sts = data["statuses"][i]
            print(f"{mode:<6} {tv:>14.4e} {mv:>14.4e} {rng:>6}   {err:>7.3f}%  {sts:<14}")
        print("─" * 72)

    print("\n=== AVERAGE ERRORS ===")
    print(f"{'Method':<30} {'R Error':>10} {'C Error':>10} {'L Error':>10}")
    print("─" * 64)

    # Fixed-range baseline averages
    fixed_avgs = {}
    for mode in ["R", "C", "L"]:
        fe = run_fixed_baseline(mode, TEST_VALUES[mode])
        fixed_avgs[mode] = np.mean(fe)

    print(
        f"{'Fixed-range (no auto)':<30} "
        f"{fixed_avgs['R']:>9.2f}%  "
        f"{fixed_avgs['C']:>9.2f}%  "
        f"{fixed_avgs['L']:>9.2f}%"
    )
    print(
        f"{'Your auto-ranging sim':<30} "
        f"{results['R']['avg_error']:>9.3f}%  "
        f"{results['C']['avg_error']:>9.3f}%  "
        f"{results['L']['avg_error']:>9.3f}%"
    )
    print("─" * 64)

    all_pass = all(results[m]["avg_error"] <= 2.0 for m in ["R", "C", "L"])
    status_str = "✓ PASS — all modes ≤ 2 % error" if all_pass else "✗ FAIL — check noise model"
    print(f"\nOverall: {status_str}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)   # Reproducible results

    print("Running Smart Multimeter simulation…")
    print("  Modes: Resistance (R), Capacitance (C), Inductance (L)")
    print("  Samples per mode: 50 log-spaced values across 10⁵ range")
    print("  Noise model: Gaussian σ = 0.5 % of true value\n")

    results = {}
    for mode in ["R", "C", "L"]:
        print(f"  Simulating {mode}…")
        results[mode] = run_simulation(mode)

    print_results_table(results)

    print("Generating plots…")
    plot_accuracy(results)
    plot_autorange(results)

    print("\nDone. Output files are in the results/ folder.")
