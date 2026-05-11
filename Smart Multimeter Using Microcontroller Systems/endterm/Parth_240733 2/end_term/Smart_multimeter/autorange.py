"""
autorange.py — Auto-ranging engine with hysteresis
Smart Multimeter Simulation

Range table (same structure for R, C, L — values differ by mode):
  Range 1: max = 100      (Ω / nF / μH)
  Range 2: max = 1 000    (Ω / nF / μH) × scale
  Range 3: max = 10 000
  Range 4: max = 100 000
  Range 5: max = 1 000 000
"""

# ---------------------------------------------------------------------------
# Range definitions — raw SI units (Ω, F, H)
# ---------------------------------------------------------------------------

RANGES = {
    "R": [
        {"label": "Range 1", "max": 1e2},
        {"label": "Range 2", "max": 1e3},
        {"label": "Range 3", "max": 1e4},
        {"label": "Range 4", "max": 1e5},
        {"label": "Range 5", "max": 1e6},
    ],
    "C": [
        {"label": "Range 1", "max": 10e-9},
        {"label": "Range 2", "max": 100e-9},
        {"label": "Range 3", "max": 1e-6},
        {"label": "Range 4", "max": 10e-6},
        {"label": "Range 5", "max": 100e-6},
    ],
    "L": [
        {"label": "Range 1", "max": 10e-6},
        {"label": "Range 2", "max": 100e-6},
        {"label": "Range 3", "max": 1e-3},
        {"label": "Range 4", "max": 10e-3},
        {"label": "Range 5", "max": 100e-3},
    ],
}

STEP_UP_FRAC   = 0.90   # >90% of range max → step up
STEP_DOWN_FRAC = 0.10   # <10% of range max → step down
HYSTERESIS_N   = 3      # consecutive triggers required before switching


class AutoRanger:
    """
    Stateful auto-ranging engine for a single measurement mode.

    Usage:
        ar = AutoRanger("R")
        result = ar.update(measured_value)
    """

    def __init__(self, mode: str):
        if mode not in RANGES:
            raise ValueError(f"mode must be one of {list(RANGES.keys())}")
        self.mode = mode
        self.ranges = RANGES[mode]
        self.current_range_idx = 0   # start at Range 1
        self._up_count   = 0
        self._down_count = 0

    def reset(self):
        """Reset to initial state (Range 1, no pending switches)."""
        self.current_range_idx = 0
        self._up_count   = 0
        self._down_count = 0

    @property
    def current_range(self) -> dict:
        return self.ranges[self.current_range_idx]

    def update(self, reading: float) -> dict:
        """
        Feed one measurement reading into the engine.

        Returns a dict with:
            active_range_idx  (0-based)
            active_range_label
            status            "SETTLED" | "STEP_UP" | "STEP_DOWN" | "OL"
            value             reading (same as input, after range confirmation)
        """
        n_ranges = len(self.ranges)
        rng_max  = self.current_range["max"]

        # Overload: exceeds top range
        if (self.current_range_idx == n_ranges - 1
                and reading > STEP_UP_FRAC * rng_max):
            self._up_count   = 0
            self._down_count = 0
            return {
                "active_range_idx":   self.current_range_idx,
                "active_range_label": self.current_range["label"],
                "status": "OL",
                "value":  reading,
            }

        # --- Check for step-up condition ---
        if reading > STEP_UP_FRAC * rng_max and self.current_range_idx < n_ranges - 1:
            self._up_count += 1
            self._down_count = 0
            if self._up_count >= HYSTERESIS_N:
                self.current_range_idx += 1
                self._up_count = 0
                return self._settled_result(reading, "STEP_UP")
            return self._settled_result(reading, "SETTLED")   # pending, stay put

        # --- Check for step-down condition ---
        if reading < STEP_DOWN_FRAC * rng_max and self.current_range_idx > 0:
            self._down_count += 1
            self._up_count = 0
            if self._down_count >= HYSTERESIS_N:
                self.current_range_idx -= 1
                self._down_count = 0
                return self._settled_result(reading, "STEP_DOWN")
            return self._settled_result(reading, "SETTLED")   # pending, stay put

        # --- 10–90%: settled ---
        self._up_count   = 0
        self._down_count = 0
        return self._settled_result(reading, "SETTLED")

    def _settled_result(self, reading: float, status: str) -> dict:
        return {
            "active_range_idx":   self.current_range_idx,
            "active_range_label": self.current_range["label"],
            "status": status,
            "value":  reading,
        }


def best_range_for(value: float, mode: str) -> int:
    """
    Return the ideal 0-based range index for a given true value.
    Used to compute the fixed-range baseline.
    """
    ranges = RANGES[mode]
    for i, r in enumerate(ranges):
        if value <= r["max"]:
            return i
    return len(ranges) - 1


if __name__ == "__main__":
    import numpy as np

    print("=== autorange.py sanity check — resistance sweep ===\n")
    ar = AutoRanger("R")
    test_values = np.logspace(1, 6, 30)   # 10 Ω … 1 MΩ

    for v in test_values:
        # Feed same value multiple times to satisfy hysteresis
        for _ in range(HYSTERESIS_N):
            res = ar.update(v)
        print(f"  R={v:>10.1f} Ω  →  {res['active_range_label']}  [{res['status']}]")
