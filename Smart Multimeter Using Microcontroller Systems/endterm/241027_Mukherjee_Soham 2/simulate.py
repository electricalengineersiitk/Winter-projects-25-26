from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from autorange import AutoRangeEngine, Mode
from measurement import measure_capacitance, measure_inductance, measure_resistance


@dataclass(frozen=True)
class ModeConfig:
    mode: Mode
    symbol: str
    unit: str
    min_value: float
    max_value: float
    measure_fn: Callable[[float], Tuple[float, float]]


def _generate_sweep(
    min_value: float, max_value: float, total_points: int = 50
) -> np.ndarray:
    half = total_points // 2
    up = np.logspace(np.log10(min_value), np.log10(max_value), num=half)
    down = np.logspace(
        np.log10(max_value), np.log10(min_value), num=total_points - half
    )
    return np.concatenate([up, down])


def _fixed_range_baseline(
    true_value: float,
    fixed_range_max: float,
    rng: np.random.Generator,
) -> float:
    utilization = true_value / fixed_range_max

    if utilization >= 1.0:
        measured = fixed_range_max
    else:
        scale_penalty = 1.0 if utilization >= 0.1 else (0.1 / max(utilization, 1e-9))
        sigma = 0.005 * scale_penalty * true_value
        measured = float(rng.normal(true_value, sigma))
        measured = max(measured, 1e-15)

    return abs((measured - true_value) / true_value) * 100.0


def _run_mode(config: ModeConfig, rng: np.random.Generator) -> Dict[str, List[float]]:
    values = _generate_sweep(config.min_value, config.max_value, total_points=50)
    engine = AutoRangeEngine(config.mode, hysteresis_count=3)

    measured_values: List[float] = []
    auto_errors: List[float] = []
    baseline_errors: List[float] = []
    range_indices: List[float] = []

    fixed_range_max = 10.0 * config.min_value

    for index, true_value in enumerate(values, start=1):
        measured_value, error_pct = config.measure_fn(float(true_value))
        result = engine.process(measured_value)

        if result.switched:
            print(
                f"[{config.symbol}] sample {index:02d}: switched to range {result.range_index} "
                f"({result.range_label})"
            )

        measured_values.append(measured_value)
        auto_errors.append(error_pct)
        baseline_errors.append(
            _fixed_range_baseline(float(true_value), fixed_range_max, rng)
        )
        range_indices.append(float(result.range_index))

    return {
        "true_values": values.tolist(),
        "measured_values": measured_values,
        "auto_errors": auto_errors,
        "baseline_errors": baseline_errors,
        "range_indices": range_indices,
    }


def _plot_accuracy(
    results: Dict[str, Dict[str, List[float]]], output_path: str
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 13), sharex=False)

    mode_order = ["R", "C", "L"]
    for axis, mode_key in zip(axes, mode_order):
        data = results[mode_key]
        x_values = np.array(data["true_values"])
        auto_err = np.array(data["auto_errors"])
        base_err = np.array(data["baseline_errors"])

        axis.plot(x_values, auto_err, label="Auto-ranging", linewidth=2)
        axis.plot(x_values, base_err, label="Fixed-range baseline", linewidth=2)
        axis.set_xscale("log")
        axis.set_ylabel("Error (%)")
        axis.set_title(f"{mode_key} Accuracy vs Input Value")
        axis.legend()
        axis.grid(True, which="both", linestyle="--", alpha=0.4)

    axes[-1].set_xlabel("True component value (log scale)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_autorange_state(
    results: Dict[str, Dict[str, List[float]]], output_path: str
) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    samples = np.arange(1, 51)

    for mode_key in ["R", "C", "L"]:
        ranges = np.array(results[mode_key]["range_indices"])
        ax.plot(
            samples,
            ranges,
            marker="o",
            linewidth=1.8,
            markersize=3,
            label=f"{mode_key} mode",
        )

    ax.set_title("Auto-Range State Over Time")
    ax.set_xlabel("Test sample index")
    ax.set_ylabel("Active range (1 to 5)")
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _print_summary(results: Dict[str, Dict[str, List[float]]]) -> None:
    print("\nAverage error across 50 tests per mode")
    print("----------------------------------------")
    print(f"{'Method':<28} {'R Error':>10} {'C Error':>10} {'L Error':>10}")

    auto_r = float(np.mean(results["R"]["auto_errors"]))
    auto_c = float(np.mean(results["C"]["auto_errors"]))
    auto_l = float(np.mean(results["L"]["auto_errors"]))

    base_r = float(np.mean(results["R"]["baseline_errors"]))
    base_c = float(np.mean(results["C"]["baseline_errors"]))
    base_l = float(np.mean(results["L"]["baseline_errors"]))

    print(
        f"{'Fixed-range (no auto)':<28} {base_r:>9.2f}% {base_c:>9.2f}% {base_l:>9.2f}%"
    )
    print(
        f"{'Auto-ranging simulation':<28} {auto_r:>9.2f}% {auto_c:>9.2f}% {auto_l:>9.2f}%"
    )


def main() -> None:
    rng = np.random.default_rng(2026)

    mode_configs = [
        ModeConfig(
            Mode.RESISTANCE,
            "R",
            "Ohm",
            100.0,
            1_000_000.0,
            lambda value: measure_resistance(value, r_ref=value),
        ),
        ModeConfig(Mode.CAPACITANCE, "C", "F", 10e-9, 100e-6, measure_capacitance),
        ModeConfig(Mode.INDUCTANCE, "L", "H", 10e-6, 100e-3, measure_inductance),
    ]

    results: Dict[str, Dict[str, List[float]]] = {}
    for config in mode_configs:
        results[config.symbol] = _run_mode(config, rng)

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    _plot_accuracy(results, os.path.join(results_dir, "plot_accuracy.png"))
    _plot_autorange_state(results, os.path.join(results_dir, "plot_autorange.png"))
    _print_summary(results)


if __name__ == "__main__":
    main()
