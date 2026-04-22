from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class MapConfig:
    width: int = 100
    height: int = 100
    start: Tuple[float, float] = (5.0, 50.0)
    goal: Tuple[float, float] = (95.0, 50.0)
    obstacle_center: Tuple[float, float] = (50.0, 50.0)
    obstacle_radius: float = 10.0
    safety_margin: float = 4.0
    grid_resolution: float = 1.0

    @property
    def inflated_radius(self) -> float:
        return self.obstacle_radius + self.safety_margin


def default_map() -> MapConfig:
    return MapConfig()


def is_blocked(point: Tuple[float, float], config: MapConfig | None = None) -> bool:
    config = config or default_map()
    point_array = np.asarray(point, dtype=float)
    center = np.asarray(config.obstacle_center, dtype=float)
    return float(np.linalg.norm(point_array - center)) <= config.inflated_radius


def results_dir() -> Path:
    return Path(__file__).resolve().parent / "results"


def script_dir() -> Path:
    return Path(__file__).resolve().parent
