"""A* planner (Stream B owns). See docs/STREAM_B.md, hours 1-4.

plan(cost, blocked, start, goal) -> list[(row, col)] | None
  - 8-connected; step cost = mean(cost[a], cost[b]) * step_len_m (0.25 straight, 0.3536 diagonal)
  - heuristic = Euclidean distance in METRES (admissible because every cell cost >= 1)
  - no diagonal corner-cutting through blocked cells
  - returns None if goal unreachable (rover -> SAFE_STOP)
"""
from __future__ import annotations

import numpy as np

from .contracts import Cell


def plan(cost: np.ndarray, blocked: np.ndarray, start: Cell, goal: Cell) -> list[Cell] | None:
    raise NotImplementedError("Stream B: hours 1-4")
