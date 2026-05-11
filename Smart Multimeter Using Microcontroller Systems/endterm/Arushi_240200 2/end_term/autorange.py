"""
autorange.py — Auto-ranging engine with hysteresis (3-sample rule)
Smart Multimeter Simulation | End-Term Project
"""

from dataclasses import dataclass, field
from typing import Literal

# ---------------------------------------------------------------------------
# Range definitions — shared structure for R, C, and L (scaled differently)
# ---------------------------------------------------------------------------

@dataclass
class RangeConfig:
    label: str
    max_val: float        # absolute max in SI units for this mode
    step_up_frac: float   # fraction of max_val that triggers up-range
    step_down_frac: float # fraction of max_val that triggers down-range (0 = never)


# Resistance ranges (Ω)
RESISTANCE_RANGES = [
    RangeConfig("R1", 100,       0.90, 0.00),
    RangeConfig("R2", 1_000,     0.90, 0.10),
    RangeConfig("R3", 10_000,    0.90, 0.10),
    RangeConfig("R4", 100_000,   0.90, 0.10),
    RangeConfig("R5", 1_000_000, 1.00, 0.10),   # top range: no step-up
]

# Capacitance ranges (F)
CAPACITANCE_RANGES = [
    RangeConfig("C1", 10e-9,   0.90, 0.00),
    RangeConfig("C2", 100e-9,  0.90, 0.10),
    RangeConfig("C3", 1e-6,    0.90, 0.10),
    RangeConfig("C4", 10e-6,   0.90, 0.10),
    RangeConfig("C5", 100e-6,  1.00, 0.10),
]

# Inductance ranges (H)
INDUCTANCE_RANGES = [
    RangeConfig("L1", 10e-6,   0.90, 0.00),
    RangeConfig("L2", 100e-6,  0.90, 0.10),
    RangeConfig("L3", 1e-3,    0.90, 0.10),
    RangeConfig("L4", 10e-3,   0.90, 0.10),
    RangeConfig("L5", 100e-3,  1.00, 0.10),
]

RANGE_MAP = {
    "R": RESISTANCE_RANGES,
    "C": CAPACITANCE_RANGES,
    "L": INDUCTANCE_RANGES,
}

HYSTERESIS_COUNT = 3   # consecutive triggers required before switching


# ---------------------------------------------------------------------------
# AutoRanger class
# ---------------------------------------------------------------------------

@dataclass
class AutoRanger:
    """
    Stateful auto-ranging engine for a single measurement mode.

    Usage:
        ranger = AutoRanger("R")
        range_idx, status = ranger.update(reading)
    """
    mode: Literal["R", "C", "L"]
    _current_idx: int = field(default=0, init=False)
    _up_count: int = field(default=0, init=False)
    _down_count: int = field(default=0, init=False)

    def reset(self):
        self._current_idx = 0
        self._up_count = 0
        self._down_count = 0

    @property
    def ranges(self) -> list[RangeConfig]:
        return RANGE_MAP[self.mode]

    @property
    def current_range(self) -> RangeConfig:
        return self.ranges[self._current_idx]

    def update(self, reading: float) -> tuple[int, str]:
        """
        Feed one measured value into the ranger.

        Returns:
            (range_index_1based, status)
            status ∈ {"SETTLED", "STEP_UP", "STEP_DOWN", "OL"}
        """
        ranges = self.ranges
        n = len(ranges)
        r = ranges[self._current_idx]

        # Overload check — above top range
        if reading > ranges[-1].max_val:
            self._up_count = 0
            self._down_count = 0
            return self._current_idx + 1, "OL"

        # --- Step-up logic ---
        if self._current_idx < n - 1 and reading > r.step_up_frac * r.max_val:
            self._up_count += 1
            self._down_count = 0
            if self._up_count >= HYSTERESIS_COUNT:
                self._current_idx += 1
                self._up_count = 0
                return self._current_idx + 1, "STEP_UP"
            return self._current_idx + 1, "SETTLED"   # pending

        # --- Step-down logic ---
        if self._current_idx > 0 and reading < r.step_down_frac * r.max_val:
            self._down_count += 1
            self._up_count = 0
            if self._down_count >= HYSTERESIS_COUNT:
                self._current_idx -= 1
                self._down_count = 0
                return self._current_idx + 1, "STEP_DOWN"
            return self._current_idx + 1, "SETTLED"   # pending

        # --- Settled ---
        self._up_count = 0
        self._down_count = 0
        return self._current_idx + 1, "SETTLED"


def fixed_range(reading: float, mode: Literal["R", "C", "L"]) -> int:
    """
    Baseline: fixed range is always Range 3 (mid-scale).
    Returns the range index (1-based) regardless of reading.
    """
    return 3


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== autorange.py self-test (Resistance sweep) ===\n")
    import numpy as np

    ranger = AutoRanger("R")
    values = np.logspace(1, 6, 30)   # 10 Ω → 1 MΩ

    print(f"{'True R (Ω)':>15} | {'Range':>6} | {'Status':>10}")
    print("-" * 40)
    for v in values:
        idx, status = ranger.update(v)
        print(f"{v:>15.1f} | {idx:>6} | {status:>10}")
