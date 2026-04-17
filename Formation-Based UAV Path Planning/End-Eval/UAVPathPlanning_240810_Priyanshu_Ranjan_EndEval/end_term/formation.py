from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass(frozen=True)
class FormationSpec:
    name: str
    offsets: np.ndarray
    labels: List[str]
    connections: List[Tuple[int, int]]


def create_letter_a_formation(scale: float = 1.0) -> FormationSpec:
    offsets = np.array(
        [
            [0.0, 5.0],
            [-2.6, 2.0],
            [2.6, 2.0],
            [-4.0, -1.5],
            [4.0, -1.5],
            [-2.0, -5.0],
            [2.0, -5.0],
        ],
        dtype=float,
    ) * scale

    labels = [f"UAV{i + 1}" for i in range(len(offsets))]
    connections = [(0, 1), (0, 2), (1, 3), (3, 5), (2, 4), (4, 6), (3, 4)]
    return FormationSpec(name="A", offsets=offsets, labels=labels, connections=connections)


def formation_positions(centroid_positions: np.ndarray, headings: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    centroid_positions = np.asarray(centroid_positions, dtype=float)
    headings = np.asarray(headings, dtype=float)
    offsets = np.asarray(offsets, dtype=float)

    cos_heading = np.cos(headings)
    sin_heading = np.sin(headings)
    rotation = np.stack(
        [
            np.stack([cos_heading, -sin_heading], axis=-1),
            np.stack([sin_heading, cos_heading], axis=-1),
        ],
        axis=-2,
    )
    rotated_offsets = np.einsum("tij,nj->tni", rotation, offsets)
    return centroid_positions[:, None, :] + rotated_offsets


def compute_headings(time: np.ndarray, positions: np.ndarray) -> np.ndarray:
    positions = np.asarray(positions, dtype=float)
    time = np.asarray(time, dtype=float)
    if len(time) != len(positions):
        raise ValueError("time and positions must have matching lengths")

    vx = np.gradient(positions[:, 0], time, edge_order=2)
    vy = np.gradient(positions[:, 1], time, edge_order=2)
    headings = np.unwrap(np.arctan2(vy, vx))

    if len(headings) > 1:
        headings[0] = headings[1]
        headings[-1] = headings[-2]
    return headings
