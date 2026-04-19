"""
simulate.py
-----------
Main simulation entry point for the Smart Digital Multimeter.

Generates 50 test values across all 5 measurement ranges for each of the
three modes (R, C, L), applies Gaussian noise via measurement.py, routes
each reading through the auto-ranging engine in autorange.py, and reports:

  • Average % error per mode (target ≤ 2 %)
  • Final results table printed to console
  • Two plots saved to results/
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (no display required)
import matplotlib.pyplot as plt

# ── Local imports ──────────────────────────────────────────────────────────────
from measurement import measure_resistance, measure_capacitance, measure_inductance
from autorange import AutoRanger, RANGES
from protocol import format_packet

# ── Constants ──────────────────────────────────────────────────────────────────
N_SAMPLES = 50
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

MODES: dict[str, dict] = {
    "R": {
        "label": "Resistance",
        "unit": "Ω",
        "ranges": RANGES["R"],
        "measure_fn": measure_resistance,
        "fmt": lambda v: f"{v:>12.3f} Ω",
    },
    "C": {
        "label": "Capacitance",
        "unit": "F",
        "ranges": RANGES["C"],
        "measure_fn": measure_capacitance,
        "fmt": lambda v: f"{v:>12.3e} F",
    },
    "L": {
        "label": "Inductance",
        "unit": "H",
        "ranges": RANGES["L"],
        "measure_fn": measure_inductance,
        "fmt": lambda v: f"{v:>12.3e} H",
    },
}


# ── Helper ─────────────────────────────────────────────────────────────────────

def generate_test_values(ranges: list[float], n: int = N_SAMPLES) -> np.ndarray:
    """
    Generate n test values distributed across all 5 ranges.

    Each range gets n//5 log-uniform samples between its lower and upper bounds.
    """
    values = []
    lower = 0.0
    per_range = n // len(ranges)
    for upper in ranges:
        lo = max(lower, upper * 0.01)   # at least 1 % of range max as lower bound
        vals = np.logspace(np.log10(lo), np.log10(upper * 0.85), per_range)
        values.extend(vals.tolist())
        lower = upper
    # Trim / pad to exactly n samples
    values = values[:n]
    while len(values) < n:
        values.append(values[-1])
    return np.array(values)


def run_simulation(mode: str) -> dict:
    """
    Run the full simulation for one mode.

    Returns a dict with keys:
        true_values, measured_values, active_ranges, errors,
        settled_indices, packets
    """
    cfg = MODES[mode]
    true_values = generate_test_values(cfg["ranges"], N_SAMPLES)

    ranger = AutoRanger(mode)
    ranger.reset()

    measured_values, active_ranges, errors, settled_indices, packets = (
        [], [], [], [], []
    )

    timestamp_counter = 0
    for i, tv in enumerate(true_values):
        # Apply physics noise
        meas, err = cfg["measure_fn"](tv)

        # Feed into auto-ranging engine (may take multiple passes to settle)
        result = ranger.process(meas)
        # If not settled immediately, keep re-feeding until settled or OL
        max_iter = 20
        itr = 0
        while not result.settled and not result.overload and itr < max_iter:
            result = ranger.process(meas)
            itr += 1

        measured_values.append(meas)
        active_ranges.append(result.active_range)
        errors.append(err)

        if result.settled:
            settled_indices.append(i)
            pkt = format_packet(mode, result.active_range, meas, err, timestamp_counter)
            packets.append(pkt)
            timestamp_counter += 100

    return {
        "true_values": true_values,
        "measured_values": np.array(measured_values),
        "active_ranges": np.array(active_ranges),
        "errors": np.array(errors),
        "settled_indices": settled_indices,
        "packets": packets,
    }


def fixed_range_errors(mode: str, true_values: np.ndarray) -> np.ndarray:
    """
    Baseline: simulate a fixed-range meter stuck on Range 3 (middle range).
    Out-of-range readings are clipped and incur large errors.
    """
    cfg = MODES[mode]
    fixed_max = cfg["ranges"][2]   # Range 3

    errors = []
    for tv in true_values:
        meas, err = cfg["measure_fn"](tv)
        if meas > fixed_max:
            # Saturated — report max, error is very large
            err = abs(fixed_max - tv) / tv * 100.0
        errors.append(err)
    return np.array(errors)


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_accuracy(results: dict[str, dict]) -> str:
    """Plot 1: Accuracy vs Input Value (log X-axis)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Plot 1 – Accuracy vs Input Value (Auto-Range vs Fixed-Range Baseline)",
                 fontsize=13, fontweight="bold")

    mode_colors = {"R": ("#2196F3", "#FF5722"), "C": ("#4CAF50", "#FF9800"),
                   "L": ("#9C27B0", "#F44336")}

    for ax, (mode, res) in zip(axes, results.items()):
        cfg = MODES[mode]
        tvs = res["true_values"]
        auto_err = res["errors"]
        fixed_err = fixed_range_errors(mode, tvs)

        c_auto, c_fixed = mode_colors[mode]
        ax.semilogx(tvs, auto_err, color=c_auto, linewidth=1.8,
                    label="Auto-Range", marker="o", markersize=3)
        ax.semilogx(tvs, fixed_err, color=c_fixed, linewidth=1.4,
                    linestyle="--", label="Fixed-Range (R3)", marker="s", markersize=3)
        ax.axhline(y=2.0, color="grey", linestyle=":", linewidth=1, label="2 % target")

        ax.set_title(f"{cfg['label']} ({cfg['unit']})", fontsize=11)
        ax.set_xlabel(f"True {cfg['label']} ({cfg['unit']}) — log scale", fontsize=9)
        ax.set_ylabel("% Measurement Error", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "plot_accuracy.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_autorange(results: dict[str, dict]) -> str:
    """Plot 2: Auto-Range State Over Time (sample index 1-50)."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("Plot 2 – Auto-Range State Over Time (Sample Index 1 – 50)",
                 fontsize=13, fontweight="bold")

    colors = {"R": "#2196F3", "C": "#4CAF50", "L": "#9C27B0"}

    for ax, (mode, res) in zip(axes, results.items()):
        cfg = MODES[mode]
        x = np.arange(1, N_SAMPLES + 1)
        y = res["active_ranges"]

        ax.step(x, y, where="post", color=colors[mode], linewidth=2)
        ax.fill_between(x, y, step="post", alpha=0.15, color=colors[mode])

        # Mark settled samples
        si = res["settled_indices"]
        if si:
            ax.scatter(np.array(si) + 1, y[si], color=colors[mode],
                       edgecolors="white", zorder=5, s=40, label="Settled")

        ax.set_ylabel(f"{cfg['label']}\nActive Range", fontsize=9)
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_ylim(0.5, 5.5)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")

    axes[-1].set_xlabel("Test Sample Index", fontsize=10)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "plot_autorange.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ── Console Report ─────────────────────────────────────────────────────────────

def print_results_table(results: dict[str, dict]) -> None:
    """Print a formatted results table to stdout."""
    sep = "─" * 90

    print("\n" + sep)
    print(f"{'SMART MULTIMETER — SIMULATION RESULTS':^90}")
    print(sep)
    print(f"{'Mode':<14} {'Samples':>8} {'Avg Error (Auto)':>18} "
          f"{'Avg Error (Fixed)':>19} {'Max Error (Auto)':>18} {'Settled':>9}")
    print(sep)

    for mode, res in results.items():
        cfg = MODES[mode]
        n = N_SAMPLES
        avg_auto = res["errors"].mean()
        avg_fixed = fixed_range_errors(mode, res["true_values"]).mean()
        max_auto = res["errors"].max()
        n_settled = len(res["settled_indices"])
        print(
            f"{cfg['label']:<14} {n:>8} "
            f"{avg_auto:>17.4f}% "
            f"{avg_fixed:>18.4f}% "
            f"{max_auto:>17.4f}% "
            f"{n_settled:>8}/{n}"
        )
        status = "✓ PASS" if avg_auto <= 2.0 else "✗ FAIL"
        print(f"  └─ Target ≤ 2.00% avg error:  {status}")

    print(sep + "\n")


def print_sample_packets(results: dict[str, dict], n: int = 5) -> None:
    """Print the first n OTG packets for each mode."""
    print("── Sample OTG Packets ──────────────────────────────────────────────────────────")
    for mode, res in results.items():
        print(f"\n  {MODES[mode]['label']} Packets:")
        for pkt in res["packets"][:n]:
            print(f"    {pkt}")
            raw = pkt.encode()
            print(f"    hex: {raw.hex().upper()}")
    print()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    np.random.seed(42)   # reproducible results

    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print("║          Smart Digital Multimeter — Python Simulation            ║")
    print("╚══════════════════════════════════════════════════════════════════╝\n")
    print(f"  Running {N_SAMPLES} samples × 3 modes …\n")

    results: dict[str, dict] = {}
    for mode in ("R", "C", "L"):
        print(f"  [{MODES[mode]['label']}] simulating …", end=" ", flush=True)
        results[mode] = run_simulation(mode)
        avg = results[mode]["errors"].mean()
        print(f"avg error = {avg:.4f}%")

    print_results_table(results)
    print_sample_packets(results, n=3)

    # ── Plots ──────────────────────────────────────────────────────────────────
    print("  Generating plots …")
    p1 = plot_accuracy(results)
    p2 = plot_autorange(results)
    print(f"  ✓ Saved: {p1}")
    print(f"  ✓ Saved: {p2}\n")

    print("  Simulation complete.\n")


if __name__ == "__main__":
    main()
