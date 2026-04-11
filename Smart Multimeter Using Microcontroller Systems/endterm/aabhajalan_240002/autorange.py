"""
autorange.py — Auto-ranging engine for the Smart Multimeter simulation.

Implements a 5-range system for R, C, and L with:
  - Step-up   when reading > 90 % of current range max (3-sample hysteresis)
  - Step-down when reading < 10 % of current range max (3-sample hysteresis)
  - OL (overload) when reading exceeds all ranges
"""

from dataclasses import dataclass, field
from typing import Literal

# ---------------------------------------------------------------------------
# Range tables  (max_value, step_up_threshold, step_down_threshold)
# step_down for range 1 is None (can't go lower)
# step_up  for range 5 is None (can't go higher → OL)
# ---------------------------------------------------------------------------

RANGES_R = [
    (100,          90,    None),    # Range 1: 0–100 Ω
    (1_000,        900,   100),     # Range 2: 0–1 kΩ
    (10_000,       9_000, 1_000),   # Range 3: 0–10 kΩ
    (100_000,      90_000,10_000),  # Range 4: 0–100 kΩ
    (1_000_000,    None,  100_000), # Range 5: 0–1 MΩ
]

RANGES_C = [
    (10e-9,        9e-9,  None),    # Range 1: 0–10 nF
    (100e-9,       90e-9, 10e-9),   # Range 2: 0–100 nF
    (1e-6,         0.9e-6,100e-9),  # Range 3: 0–1 µF
    (10e-6,        9e-6,  1e-6),    # Range 4: 0–10 µF
    (100e-6,       None,  10e-6),   # Range 5: 0–100 µF
]

RANGES_L = [
    (10e-6,        9e-6,  None),    # Range 1: 0–10 µH
    (100e-6,       90e-6, 10e-6),   # Range 2: 0–100 µH
    (1e-3,         0.9e-3,100e-6),  # Range 3: 0–1 mH
    (10e-3,        9e-3,  1e-3),    # Range 4: 0–10 mH
    (100e-3,       None,  10e-3),   # Range 5: 0–100 mH
]

MODE_RANGES = {"R": RANGES_R, "C": RANGES_C, "L": RANGES_L}

HYSTERESIS_COUNT = 3   # consecutive triggers needed before switching


@dataclass
class AutoRanger:
    """
    Stateful auto-ranging engine.  Instantiate one per measurement session.

    Args:
        mode: "R", "C", or "L"
    """
    mode: Literal["R", "C", "L"]
    _range_idx: int = field(default=0, init=False)
    _up_counter: int = field(default=0, init=False)
    _down_counter: int = field(default=0, init=False)

    # -----------------------------------------------------------------------
    def reset(self) -> None:
        """Reset to Range 1 and clear hysteresis counters."""
        self._range_idx = 0
        self._up_counter = 0
        self._down_counter = 0

    # -----------------------------------------------------------------------
    @property
    def current_range(self) -> int:
        """1-based range number (1–5)."""
        return self._range_idx + 1

    # -----------------------------------------------------------------------
    def process(self, reading: float) -> dict:
        """
        Feed one measured value into the auto-ranging engine.

        Returns a dict with keys:
            value        – the reading (unchanged)
            range        – active range number (1–5)
            status       – "SETTLED" | "STEP_UP" | "STEP_DOWN" | "OL"
            range_max    – upper limit of the active range
        """
        ranges = MODE_RANGES[self.mode]
        max_val, up_thresh, down_thresh = ranges[self._range_idx]

        # ---- Overload check -----------------------------------------------
        if reading > ranges[-1][0]:
            return {
                "value": reading,
                "range": self.current_range,
                "status": "OL",
                "range_max": max_val,
            }

        # ---- Step-up check ------------------------------------------------
        if up_thresh is not None and reading > up_thresh:
            self._up_counter += 1
            self._down_counter = 0
            if self._up_counter >= HYSTERESIS_COUNT:
                self._range_idx = min(self._range_idx + 1, len(ranges) - 1)
                self._up_counter = 0
            return {
                "value": reading,
                "range": self.current_range,
                "status": "STEP_UP",
                "range_max": ranges[self._range_idx][0],
            }

        # ---- Step-down check ----------------------------------------------
        if down_thresh is not None and reading < down_thresh:
            self._down_counter += 1
            self._up_counter = 0
            if self._down_counter >= HYSTERESIS_COUNT:
                self._range_idx = max(self._range_idx - 1, 0)
                self._down_counter = 0
            return {
                "value": reading,
                "range": self.current_range,
                "status": "STEP_DOWN",
                "range_max": ranges[self._range_idx][0],
            }

        # ---- Settled -------------------------------------------------------
        self._up_counter = 0
        self._down_counter = 0
        return {
            "value": reading,
            "range": self.current_range,
            "status": "SETTLED",
            "range_max": max_val,
        }


# ---------------------------------------------------------------------------
# Quick sanity check when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np

    print("=== autorange.py self-test (Resistance sweep) ===\n")
    ar = AutoRanger("R")
    test_vals = np.logspace(2, 6, 30)   # 100 Ω → 1 MΩ

    for v in test_vals:
        result = ar.process(v)
        print(
            f"input={v:>12,.1f} Ω  range={result['range']}  "
            f"status={result['status']:<14s}  max={result['range_max']:>10,.0f} Ω"
        )
