"""
autorange.py
------------
Industry-grade auto-ranging engine for R / C / L measurements.

Features
--------
* 5 measurement ranges per mode (Resistance, Capacitance, Inductance)
* Hysteresis: 3 consecutive out-of-range readings before switching
* Overload (OL) detection when reading exceeds Range-5 maximum
"""

from dataclasses import dataclass, field
from typing import Literal

# ── Range Definitions ──────────────────────────────────────────────────────────
# Keys: max value for each range index (1-indexed, stored at index 0..4)
#
#  Range  Resistance (Ω)   Capacitance (F)      Inductance (H)
#    1        100           10e-9  (10 nF)       10e-6   (10 µH)
#    2      1 000          100e-9 (100 nF)      100e-6  (100 µH)
#    3     10 000            1e-6   (1 µF)        1e-3    (1 mH)
#    4    100 000           10e-6  (10 µF)       10e-3   (10 mH)
#    5  1 000 000          100e-6 (100 µF)      100e-3  (100 mH)

RANGES: dict[str, list[float]] = {
    "R": [100.0, 1_000.0, 10_000.0, 100_000.0, 1_000_000.0],
    "C": [10e-9, 100e-9, 1e-6, 10e-6, 100e-6],
    "L": [10e-6, 100e-6, 1e-3, 10e-3, 100e-3],
}

# Hysteresis threshold: number of consecutive triggers required to switch range
HYSTERESIS_COUNT = 3

# Step thresholds (fraction of range maximum)
UPPER_THRESHOLD = 0.90   # step UP  if reading > 90 % of range max
LOWER_THRESHOLD = 0.10   # step DOWN if reading < 10 % of range max

Mode = Literal["R", "C", "L"]

OVERLOAD = "OL"


@dataclass
class AutoRanger:
    """
    Stateful auto-ranging engine for a single measurement mode.

    Usage
    -----
    ranger = AutoRanger("R")           # one instance per mode
    result = ranger.process(reading)   # returns AutoRangeResult
    """

    mode: Mode
    _range_idx: int = field(default=0, init=False, repr=False)   # 0-based
    _pending_direction: int = field(default=0, init=False, repr=False)  # +1 / -1
    _pending_count: int = field(default=0, init=False, repr=False)

    # ── Public API ─────────────────────────────────────────────────────────────

    @property
    def current_range(self) -> int:
        """Active range (1-based)."""
        return self._range_idx + 1

    @property
    def current_max(self) -> float:
        """Maximum value of the active range."""
        return RANGES[self.mode][self._range_idx]

    def process(self, reading: float) -> "AutoRangeResult":
        """
        Feed one measurement into the auto-ranging engine.

        Returns an AutoRangeResult with the active range, the reading, and
        whether the measurement has SETTLED (is considered valid).

        OL is returned when the reading exceeds the maximum of Range 5.
        """
        # ── Overload check ────────────────────────────────────────────────────
        if reading > RANGES[self.mode][-1]:
            return AutoRangeResult(
                mode=self.mode,
                reading=reading,
                active_range=self.current_range,
                settled=False,
                overload=True,
            )

        range_max = self.current_max
        upper_limit = UPPER_THRESHOLD * range_max
        lower_limit = LOWER_THRESHOLD * range_max

        # ── Determine desired direction ───────────────────────────────────────
        if reading > upper_limit:
            desired_dir = +1   # need to go UP
        elif reading < lower_limit and self._range_idx > 0:
            desired_dir = -1   # need to go DOWN
        else:
            desired_dir = 0    # SETTLED

        # ── Hysteresis counter ────────────────────────────────────────────────
        if desired_dir == 0:
            # Reset hysteresis whenever in the settled zone
            self._pending_direction = 0
            self._pending_count = 0
            return AutoRangeResult(
                mode=self.mode,
                reading=reading,
                active_range=self.current_range,
                settled=True,
                overload=False,
            )

        if desired_dir == self._pending_direction:
            self._pending_count += 1
        else:
            self._pending_direction = desired_dir
            self._pending_count = 1

        # ── Switch range after HYSTERESIS_COUNT consecutive triggers ──────────
        if self._pending_count >= HYSTERESIS_COUNT:
            new_idx = self._range_idx + self._pending_direction
            new_idx = max(0, min(new_idx, len(RANGES[self.mode]) - 1))
            self._range_idx = new_idx
            self._pending_direction = 0
            self._pending_count = 0

        return AutoRangeResult(
            mode=self.mode,
            reading=reading,
            active_range=self.current_range,
            settled=False,
            overload=False,
        )

    def reset(self) -> None:
        """Reset the engine to Range 1 with no pending hysteresis."""
        self._range_idx = 0
        self._pending_direction = 0
        self._pending_count = 0


@dataclass
class AutoRangeResult:
    """Result returned by AutoRanger.process()."""

    mode: str
    reading: float
    active_range: int
    settled: bool
    overload: bool

    def display_reading(self) -> str:
        """Human-readable reading string."""
        if self.overload:
            return OVERLOAD
        units = {"R": "Ω", "C": "F", "L": "H"}
        return f"{self.reading:.6g} {units.get(self.mode, '')}"

    def __str__(self) -> str:
        status = "SETTLED" if self.settled else ("OL" if self.overload else "RANGING")
        return (
            f"[{self.mode}] Range {self.active_range} | "
            f"{self.display_reading()} | {status}"
        )
