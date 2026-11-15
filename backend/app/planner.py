"""A* planner (Stream B owns). See docs/STREAM_B.md, hours 1-4.

plan(cost, blocked, start, goal) -> list[(row, col)] | None
  - 8-connected; step cost = mean(cost[a], cost[b]) * step_len_m (0.25 straight, 0.3536 diagonal)
  - heuristic = Euclidean distance in METRES (admissible because every cell cost >= 1)
  - no diagonal corner-cutting through blocked cells
  - returns None if goal unreachable (rover -> SAFE_STOP)
"""
from __future__ import annotations

import heapq
import math

import numpy as np

from .contracts import RES_M, Cell

_DIAG = RES_M * math.sqrt(2.0)
# (d_row, d_col, length_m)
_MOVES = (
    (-1, 0, RES_M), (1, 0, RES_M), (0, -1, RES_M), (0, 1, RES_M),
    (-1, -1, _DIAG), (-1, 1, _DIAG), (1, -1, _DIAG), (1, 1, _DIAG),
)


def _in_bounds(cell: Cell, rows: int, cols: int) -> bool:
    return 0 <= cell[0] < rows and 0 <= cell[1] < cols


def plan(cost: np.ndarray, blocked: np.ndarray, start: Cell, goal: Cell) -> list[Cell] | None:
    """Optimal 8-connected A* from start to goal (both inclusive), or None."""
    rows, cols = cost.shape
    start = (int(start[0]), int(start[1]))
    goal = (int(goal[0]), int(goal[1]))
    if not (_in_bounds(start, rows, cols) and _in_bounds(goal, rows, cols)):
        return None
    if blocked[start] or blocked[goal]:
        return None
    if start == goal:
        return [start]

    # Plain Python lists are much faster than NumPy scalar indexing in the inner loop.
    c = cost.astype(np.float64, copy=False).tolist()
    b = blocked.tolist()
    g = np.full((rows, cols), np.inf).tolist()
    closed = [[False] * cols for _ in range(rows)]
    parent: dict[Cell, Cell] = {}

    gr, gc = goal
    sqrt = math.sqrt

    def h(r: int, col: int) -> float:
        return sqrt((r - gr) ** 2 + (col - gc) ** 2) * RES_M

    sr, sc = start
    g[sr][sc] = 0.0
    heap = [(h(sr, sc), 0.0, sr, sc)]
    push, pop = heapq.heappush, heapq.heappop

    while heap:
        _, gcur, r, col = pop(heap)
        if closed[r][col]:
            continue
        if r == gr and col == gc:
            path = [(r, col)]
            while path[-1] != start:
                path.append(parent[path[-1]])
            path.reverse()
            return path
        closed[r][col] = True
        c_here = c[r][col]
        for dr, dc, length in _MOVES:
            nr, nc = r + dr, col + dc
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                continue
            if b[nr][nc] or closed[nr][nc]:
                continue
            if dr and dc and (b[r][nc] or b[nr][col]):
                continue  # no corner cutting
            ng = gcur + 0.5 * (c_here + c[nr][nc]) * length
            if ng < g[nr][nc]:
                g[nr][nc] = ng
                parent[(nr, nc)] = (r, col)
                push(heap, (ng + h(nr, nc), ng, nr, nc))
    return None


def path_cost(cost: np.ndarray, path: list[Cell]) -> float:
    """Cost of a path in the same units plan() minimises (cost-weighted metres)."""
    total = 0.0
    for (r0, c0), (r1, c1) in zip(path, path[1:]):
        length = _DIAG if (r0 != r1 and c0 != c1) else RES_M
        total += 0.5 * (float(cost[r0, c0]) + float(cost[r1, c1])) * length
    return total


def path_length_m(path: list[Cell]) -> float:
    return sum(_DIAG if (r0 != r1 and c0 != c1) else RES_M
               for (r0, c0), (r1, c1) in zip(path, path[1:]))
