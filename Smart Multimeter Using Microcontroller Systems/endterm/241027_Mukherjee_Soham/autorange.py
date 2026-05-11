from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List


class Mode(str, Enum):
	RESISTANCE = "resistance"
	CAPACITANCE = "capacitance"
	INDUCTANCE = "inductance"


@dataclass(frozen=True)
class RangeSpec:
	label: str
	max_value: float
	step_up_threshold: float | None
	step_down_threshold: float | None


MODE_SPECS: Dict[Mode, List[RangeSpec]] = {
	Mode.RESISTANCE: [
		RangeSpec("100 Ohm", 100.0, 90.0, None),
		RangeSpec("1 kOhm", 1_000.0, 900.0, 100.0),
		RangeSpec("10 kOhm", 10_000.0, 9_000.0, 1_000.0),
		RangeSpec("100 kOhm", 100_000.0, 90_000.0, 10_000.0),
		RangeSpec("1 MOhm", 1_000_000.0, None, 100_000.0),
	],
	Mode.CAPACITANCE: [
		RangeSpec("10 nF", 10e-9, 9e-9, None),
		RangeSpec("100 nF", 100e-9, 90e-9, 10e-9),
		RangeSpec("1 uF", 1e-6, 0.9e-6, 0.1e-6),
		RangeSpec("10 uF", 10e-6, 9e-6, 1e-6),
		RangeSpec("100 uF", 100e-6, None, 10e-6),
	],
	Mode.INDUCTANCE: [
		RangeSpec("10 uH", 10e-6, 9e-6, None),
		RangeSpec("100 uH", 100e-6, 90e-6, 10e-6),
		RangeSpec("1 mH", 1e-3, 0.9e-3, 0.1e-3),
		RangeSpec("10 mH", 10e-3, 9e-3, 1e-3),
		RangeSpec("100 mH", 100e-3, None, 10e-3),
	],
}


@dataclass(frozen=True)
class AutoRangeResult:
	range_index: int
	range_label: str
	status: str
	overload: bool
	switched: bool


class AutoRangeEngine:
	def __init__(self, mode: Mode, hysteresis_count: int = 3) -> None:
		if hysteresis_count < 1:
			raise ValueError("hysteresis_count must be >= 1")
		self.mode = mode
		self.ranges = MODE_SPECS[mode]
		self.hysteresis_count = hysteresis_count
		self.current_index = 0
		self._up_counter = 0
		self._down_counter = 0

	def process(self, reading: float) -> AutoRangeResult:
		if reading < 0:
			raise ValueError("reading must be non-negative")

		if reading > self.ranges[-1].max_value:
			self._up_counter = 0
			self._down_counter = 0
			return AutoRangeResult(
				range_index=len(self.ranges),
				range_label=self.ranges[-1].label,
				status="OL",
				overload=True,
				switched=False,
			)

		current = self.ranges[self.current_index]
		switched = False
		status = "SETTLED"

		if current.step_up_threshold is not None and reading > current.step_up_threshold:
			self._up_counter += 1
			self._down_counter = 0
			status = "UP_PENDING"
			if self._up_counter >= self.hysteresis_count and self.current_index < len(self.ranges) - 1:
				self.current_index += 1
				self._up_counter = 0
				switched = True
				status = "UP"
		elif current.step_down_threshold is not None and reading < current.step_down_threshold:
			self._down_counter += 1
			self._up_counter = 0
			status = "DOWN_PENDING"
			if self._down_counter >= self.hysteresis_count and self.current_index > 0:
				self.current_index -= 1
				self._down_counter = 0
				switched = True
				status = "DOWN"
		else:
			self._up_counter = 0
			self._down_counter = 0

		current = self.ranges[self.current_index]
		return AutoRangeResult(
			range_index=self.current_index + 1,
			range_label=current.label,
			status=status,
			overload=False,
			switched=switched,
		)

