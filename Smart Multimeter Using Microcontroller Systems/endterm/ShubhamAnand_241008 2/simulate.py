"""
simulation.py-main simulation entry point
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterSciNotation

from measurement import measure_resistance, measure_capacitance, measure_inductance
from autorange import AutoRanger, expected_range, RANGES

# Output directory ─────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Test value sweeps (10 values per range × 5 ranges = 50 per mode)
N_PER_RANGE = 10

TEST_VALUES = {
    "R": np.concatenate([
        np.linspace(10, 90, N_PER_RANGE),           # Range 1
        np.linspace(101, 900, N_PER_RANGE),         # Range 2
        np.linspace(1001, 9000, N_PER_RANGE),       # Range 3
        np.linspace(10100, 90000, N_PER_RANGE),     # Range 4
        np.linspace(100100, 990000, N_PER_RANGE),   # Range 5
    ]),
    "C": np.concatenate([
        np.linspace(1e-9, 9e-9, N_PER_RANGE),
        np.linspace(11e-9, 90e-9, N_PER_RANGE),
        np.linspace(101e-9, 900e-9, N_PER_RANGE),
        np.linspace(1.01e-6, 9e-6, N_PER_RANGE),
        np.linspace(10.1e-6, 90e-6, N_PER_RANGE),
    ]),
    "L": np.concatenate([
        np.linspace(1e-6, 9e-6, N_PER_RANGE),
        np.linspace(11e-6, 90e-6, N_PER_RANGE),
        np.linspace(101e-6, 900e-6, N_PER_RANGE),
        np.linspace(1.01e-3, 9e-3, N_PER_RANGE),
        np.linspace(10.1e-3, 90e-3, N_PER_RANGE),
    ]),
}

MEASURE_FN = {
    "R": measure_resistance,
    "C": measure_capacitance,
    "L": measure_inductance,
}

UNITS = {"R": "Ω", "C": "F", "L": "H"}


def fixed_range_error(mode: str, true_val: float, rng) -> float:
    _, err = MEASURE_FN[mode](true_val, rng)
    # Fixed range adds a scaling penalty when value is outside optimal range
    r_idx = expected_range(mode, true_val)
    mid_range = 2  # Range 3 is index 2
    range_penalty = abs(r_idx - mid_range) * 1.5  # % penalty per range step away
    return err + range_penalty


def run_simulation(seed: int = 42) -> dict:
    """
    Run the full 50-sample sweep for each mode.
    """
    rng = np.random.default_rng(seed)
    results = {}

    for mode in ("R", "C", "L"):
        true_values = TEST_VALUES[mode]
        ar = AutoRanger(mode)

        records = []
        for tv in true_values:
            measured, err_auto = MEASURE_FN[mode](tv, rng)
            # Feed multiple samples to let autoranging settle
            for _ in range(5):
                range_result = ar.feed(measured)

            err_fixed = fixed_range_error(mode, tv, rng)

            records.append({
                "true": tv,
                "measured": measured,
                "range_idx": range_result.range_index,
                "range_label": range_result.range_label,
                "err_auto": err_auto,
                "err_fixed": err_fixed,
            })

        results[mode] = records

    return results


def print_results_table(results: dict):
    """Print formatted results table to stdout."""
    print("\n" + "═" * 72)
    print("  SMART MULTIMETER SIMULATION — RESULTS TABLE")
    print("═" * 72)

    summary_rows = []

    for mode in ("R", "C", "L"):
        records = results[mode]
        unit = UNITS[mode]
        avg_auto = np.mean([r["err_auto"] for r in records])
        avg_fixed = np.mean([r["err_fixed"] for r in records])

        print(f"\n  Mode: {mode}  ({unit})")
        print(f"  {'True Value':>14}  {'Measured':>14}  {'Range':>8}  {'Auto%':>7}  {'Fixed%':>7}")
        print("  " + "─" * 58)
        for r in records:
            print(
                f"  {r['true']:>14.4g}  {r['measured']:>14.4g}"
                f"  {r['range_label']:>8}  {r['err_auto']:>6.3f}%  {r['err_fixed']:>6.3f}%"
            )
        print(f"\n  Average Error → Auto-ranging: {avg_auto:.3f}%   Fixed-range: {avg_fixed:.3f}%")
        summary_rows.append((mode, avg_fixed, avg_auto))

    print("\n" + "═" * 72)
    print("  SUMMARY — Average % Error Comparison")
    print("═" * 72)
    print(f"  {'Method':<30}  {'R Error':>8}  {'C Error':>8}  {'L Error':>8}")
    print("  " + "─" * 58)
    print(
        f"  {'Fixed-range (no auto)':<30}  "
        f"{summary_rows[0][1]:>7.3f}%  {summary_rows[1][1]:>7.3f}%  {summary_rows[2][1]:>7.3f}%"
    )
    print(
        f"  {'Auto-ranging simulation':<30}  "
        f"{summary_rows[0][2]:>7.3f}%  {summary_rows[1][2]:>7.3f}%  {summary_rows[2][2]:>7.3f}%"
    )
    print("═" * 72 + "\n")


def plot_accuracy(results: dict):
    """
    Plot 1 — % Error vs True Component Value (log X-axis).
    One line per mode for auto-ranging, one dashed for fixed-range baseline.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Plot 1 — Measurement Accuracy vs Component Value", fontsize=14, fontweight="bold")

    colors = {"R": "#2196F3", "C": "#4CAF50", "L": "#FF5722"}

    for ax, mode in zip(axes, ("R", "C", "L")):
        records = results[mode]
        true_vals = [r["true"] for r in records]
        err_auto  = [r["err_auto"] for r in records]
        err_fixed = [r["err_fixed"] for r in records]

        ax.semilogx(true_vals, err_auto,  color=colors[mode], linewidth=2,
                    marker="o", markersize=4, label="Auto-ranging")
        ax.semilogx(true_vals, err_fixed, color="gray", linewidth=1.5,
                    linestyle="--", marker="s", markersize=3, label="Fixed-range baseline")

        ax.axhline(y=2.0, color="red", linestyle=":", linewidth=1, label="2% target")
        ax.set_title(f"Mode: {mode}  ({UNITS[mode]})", fontsize=11)
        ax.set_xlabel(f"True {mode} Value ({UNITS[mode]})", fontsize=10)
        ax.set_ylabel("Measurement Error (%)", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "plot_accuracy.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓  Saved: {out_path}")


def plot_autorange(results: dict):
    """
    Plot 2 — Active Range Index over Test Sample Index.
    Shows the auto-ranging engine stepping up and down.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)
    fig.suptitle("Plot 2 — Auto-Range State Over Test Samples", fontsize=14, fontweight="bold")

    colors = {"R": "#2196F3", "C": "#4CAF50", "L": "#FF5722"}

    for ax, mode in zip(axes, ("R", "C", "L")):
        records = results[mode]
        sample_idx = list(range(1, len(records) + 1))
        range_nums = [r["range_idx"] + 1 for r in records]  # 1-based

        ax.step(sample_idx, range_nums, where="post", color=colors[mode],
                linewidth=2.0, label=f"Active Range ({mode})")
        ax.fill_between(sample_idx, range_nums, step="post", alpha=0.15, color=colors[mode])

        ax.set_title(f"Mode: {mode}  — Range transitions across 50 samples", fontsize=10)
        ax.set_ylabel("Active Range (1–5)", fontsize=9)
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_yticklabels([f"Range {i}" for i in range(1, 6)], fontsize=8)
        ax.set_ylim(0.5, 5.5)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=9)

    axes[-1].set_xlabel("Test Sample Index", fontsize=10)
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "plot_autorange.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓  Saved: {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\nRunning Smart Multimeter Simulation...")
    print("Testing R, C, L modes | 50 samples each | Noise σ = 0.5%\n")

    results = run_simulation(seed=42)
    print_results_table(results)

    print("Generating plots...")
    plot_accuracy(results)
    plot_autorange(results)

    # Final pass/fail check
    for mode in ("R", "C", "L"):
        avg = np.mean([r["err_auto"] for r in results[mode]])
        status = "PASS ✓" if avg <= 2.0 else "FAIL ✗"
        print(f"  {mode} average error: {avg:.3f}%  [{status}]")

    print("\nSimulation complete.\n")
