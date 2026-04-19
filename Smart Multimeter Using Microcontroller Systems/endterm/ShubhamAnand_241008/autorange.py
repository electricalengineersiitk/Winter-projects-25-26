"""
autorange.py — Auto-ranging engine for Smart Multimeter Simulation
"""

from dataclasses import dataclass, field
from typing import Literal

MeasurementMode = Literal["R", "C", "L"]

# Range tables
# Each entry: (label, max_value, step_up_threshold, step_down_threshold)
# step_up / step_down thresholds is absolute values in native units.

RANGES: dict[MeasurementMode, list[dict]] = {
    "R": [
        {"label": "Range 1", "max": 100,    "up": 90,    "down": None},
        {"label": "Range 2", "max": 1e3,    "up": 900,   "down": 100},
        {"label": "Range 3", "max": 10e3,   "up": 9e3,   "down": 1e3},
        {"label": "Range 4", "max": 100e3,  "up": 90e3,  "down": 10e3},
        {"label": "Range 5", "max": 1e6,    "up": None,  "down": 100e3},
    ],
    "C": [
        {"label": "Range 1", "max": 10e-9,  "up": 9e-9,    "down": None},
        {"label": "Range 2", "max": 100e-9, "up": 90e-9,   "down": 10e-9},
        {"label": "Range 3", "max": 1e-6,   "up": 0.9e-6,  "down": 100e-9},
        {"label": "Range 4", "max": 10e-6,  "up": 9e-6,    "down": 1e-6},
        {"label": "Range 5", "max": 100e-6, "up": None,    "down": 10e-6},
    ],
    "L": [
        {"label": "Range 1", "max": 10e-6,  "up": 9e-6,    "down": None},
        {"label": "Range 2", "max": 100e-6, "up": 90e-6,   "down": 10e-6},
        {"label": "Range 3", "max": 1e-3,   "up": 0.9e-3,  "down": 100e-6},
        {"label": "Range 4", "max": 10e-3,  "up": 9e-3,    "down": 1e-3},
        {"label": "Range 5", "max": 100e-3, "up": None,    "down": 10e-3},
    ],
}

HYSTERESIS_COUNT = 3   # consecutive triggers required before switching
OL_SYMBOL = "OL"       # overload 


@dataclass
class AutoRanger:
    """
    Stateful auto-ranging engine.

    Usage:
        ar = AutoRanger("R")
        result = ar.feed(measured_value)
        # result.range_index  → 0-based range index (0–4)
        # result.settled      → True when range is stable
        # result.overload     → True if value exceeds all ranges
    """

    mode: MeasurementMode
    _range_idx: int = field(default=0, init=False)
    _up_count: int = field(default=0, init=False)
    _down_count: int = field(default=0, init=False)
    _transitions: list = field(default_factory=list, init=False)

    # ── Public API ────────────────────────────────────────────────────────────
    def feed(self, value: float) -> "RangeResult":
        """
        Feed one measurement sample into the engine.
        Returns a RangeResult describing the current state.
        """
        ranges = RANGES[self.mode]
        n = len(ranges)

        # Overload: exceeds maximum of highest range
        if value > ranges[-1]["max"]:
            self._up_count = 0
            self._down_count = 0
            return RangeResult(
                range_index=self._range_idx,
                range_label=ranges[self._range_idx]["label"],
                settled=False,
                overload=True,
                switched=False,
            )

        current = ranges[self._range_idx]
        switched = False

        # Step UP check 
        if current["up"] is not None and value > current["up"]:
            self._up_count += 1
            self._down_count = 0
            if self._up_count >= HYSTERESIS_COUNT and self._range_idx < n - 1:
                self._range_idx += 1
                self._up_count = 0
                switched = True
                self._transitions.append((self._range_idx - 1, self._range_idx))
        # Step DOWN check 
        elif current["down"] is not None and value < current["down"]:
            self._down_count += 1
            self._up_count = 0
            if self._down_count >= HYSTERESIS_COUNT and self._range_idx > 0:
                self._range_idx -= 1
                self._down_count = 0
                switched = True
                self._transitions.append((self._range_idx + 1, self._range_idx))
        else:
            # SETTLED — value is in 10–90% band of current range
            self._up_count = 0
            self._down_count = 0

        settled = (self._up_count == 0 and self._down_count == 0)

        return RangeResult(
            range_index=self._range_idx,
            range_label=ranges[self._range_idx]["label"],
            settled=settled,
            overload=False,
            switched=switched,
        )

    def reset(self):
        """Reset engine to Range 1, clear all counters."""
        self._range_idx = 0
        self._up_count = 0
        self._down_count = 0
        self._transitions.clear()

    @property
    def current_range_index(self) -> int:
        return self._range_idx

    @property
    def current_range_label(self) -> str:
        return RANGES[self.mode][self._range_idx]["label"]

    @property
    def transition_count(self) -> int:
        return len(self._transitions)


@dataclass
class RangeResult:
    range_index: int    # 0-based
    range_label: str
    settled: bool
    overload: bool
    switched: bool

    @property
    def range_number(self) -> int:
        """1-based range number (for display)."""
        return self.range_index + 1


def expected_range(mode: MeasurementMode, value: float) -> int:
    """
    return the 0-based range index the value *should* land on.
    Used for validation in tests.
    """
    for i, r in enumerate(RANGES[mode]):
        if value <= r["max"]:
            return i
    return len(RANGES[mode]) - 1  # overload → clamp to last range


# Sample-test
if __name__ == "__main__":
    import numpy as np

    print("AutoRanger self-test — Resistance sweep 10 Ω → 1 MΩ")
    print("─" * 60)
    test_values = np.logspace(1, 6, 30)   # 10 Ω to 1 MΩ, 30 points
    ar = AutoRanger("R")

    prev_range = -1
    for v in test_values:
        result = ar.feed(v)
        marker = " ← SWITCH" if result.switched else ""
        if result.overload:
            marker = " ← OL"
        if result.range_number != prev_range:
            print(f"  {v:>12.2f} Ω  →  {result.range_label}  settled={result.settled}{marker}")
            prev_range = result.range_number

    print(f"\nTotal range transitions: {ar.transition_count}")
    print("\nAutoRanger self-test — Capacitance spot checks")
    print("─" * 60)
    ar2 = AutoRanger("C")
    spot = [5e-9, 50e-9, 500e-9, 5e-6, 50e-6]
    for v in spot:
        # Feed 5 identical samples to let hysteresis settle
        for _ in range(5):
            result = ar2.feed(v)
        exp = expected_range("C", v)
        ok = "✓" if result.range_index == exp else "✗"
        print(f"  {ok}  {v:.2e} F  →  {result.range_label}  (expected Range {exp+1})")
