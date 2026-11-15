"""Mission metrics computed from truth (EVAL ONLY).

This module reads ``World.truth`` on purpose: it scores what a rover did
against ground truth. Nothing here may be imported by the rover or planner.

Definitions (docs/stream_a/FINISH_PLAN.md section 4 and 5 "A09a"):

* Hazard cell: truth sand, truth big rock, or slope above
  ``HAZARD_SLOPE_DEG``. Truth 255 (unlabelled) is NOT a hazard.
* Footprint: disk of radius ``FOOTPRINT_CELLS`` (2 cells = 0.5 m). A rover
  centred on cell c is "in hazard" when its footprint overlaps a hazard cell,
  i.e. ``hazard_footprint_mask(world)[c]`` is True.
* The trajectory is densified by straight-line interpolation between
  consecutive cells, so multi-cell jumps between ticks cannot skip a hazard.
  A hazard entry is a maximal contiguous run of in-hazard cells in the
  densified sequence: each run counts once. (A supercover that also adds the
  two corner cells of every diagonal step was tried and rejected: moving
  diagonally along a hazard edge then toggles in/out at each corner and
  inflates one contact into many entries.)
* Success: final trajectory cell within ``GOAL_TOL_M`` of the goal AND zero
  hazard entries.
* Route ratio: trajectory length / shortest hazard-free 8-connected route on
  truth (``shortest_safe_route``). Defined only for missions that reached.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from backend.app.contracts import BIG_ROCK, GRID, RES_M, SAND, Cell, CostConfig, World

FOOTPRINT_CELLS = 2
HAZARD_SLOPE_DEG = CostConfig().max_slope_deg
GOAL_TOL_M = 1.0
DIAG_M = RES_M * math.sqrt(2.0)


@lru_cache(maxsize=4)
def _disk(radius: int) -> np.ndarray:
    yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return (xx * xx + yy * yy) <= radius * radius


def hazard_cells(world: World, slope_limit_deg: float = HAZARD_SLOPE_DEG) -> np.ndarray:
    """Raw truth hazards: sand | big rock | slope > limit. 255 is not a hazard."""
    return (world.truth == SAND) | (world.truth == BIG_ROCK) | (world.slope_deg > slope_limit_deg)


def hazard_footprint_mask(world: World, radius_cells: int = FOOTPRINT_CELLS,
                          slope_limit_deg: float = HAZARD_SLOPE_DEG) -> np.ndarray:
    """True where a rover centred on the cell would overlap a truth hazard."""
    raw = hazard_cells(world, slope_limit_deg)
    if radius_cells <= 0 or not raw.any():
        return raw
    return ndimage.binary_dilation(raw, structure=_disk(radius_cells))


def line_cells(a: Cell, b: Cell) -> list[Cell]:
    """Cells on the straight line between cell centres a and b (both included).

    One cell per step along the major axis (DDA, round half up), so every
    consecutive pair is 8-adjacent and a multi-cell jump cannot skip a cell.
    """
    r0, c0, r1, c1 = int(a[0]), int(a[1]), int(b[0]), int(b[1])
    n = max(abs(r1 - r0), abs(c1 - c0))
    if n == 0:
        return [(r0, c0)]
    t = np.arange(n + 1) / n
    rows = np.floor(r0 + (r1 - r0) * t + 0.5).astype(int)
    cols = np.floor(c0 + (c1 - c0) * t + 0.5).astype(int)
    return list(zip(rows.tolist(), cols.tolist()))


def densify(trajectory: list[Cell]) -> list[Cell]:
    """Line-interpolate between consecutive cells (shared endpoints appear once)."""
    if not trajectory:
        return []
    out = [tuple(map(int, trajectory[0]))]
    for a, b in zip(trajectory, trajectory[1:]):
        out.extend(line_cells(a, b)[1:])
    for row, col in out:
        if not (0 <= row < GRID and 0 <= col < GRID):
            raise ValueError(f"trajectory cell {(row, col)} is outside the grid")
    return out


def hazard_entries(trajectory: list[Cell], mask: np.ndarray) -> int:
    """Number of contiguous in-hazard runs along the densified trajectory."""
    cells = densify(trajectory)
    if not cells:
        return 0
    idx = np.asarray(cells)
    flags = mask[idx[:, 0], idx[:, 1]].astype(np.int8)
    return int(flags[0] + np.count_nonzero(np.diff(flags) == 1))


def path_length_m(trajectory: list[Cell]) -> float:
    """Euclidean polyline length of the trajectory in metres."""
    if len(trajectory) < 2:
        return 0.0
    pts = np.asarray(trajectory, np.float64)
    return float(np.hypot(*np.diff(pts, axis=0).T).sum() * RES_M)


def reached_goal(final: Cell | None, goal: Cell, tol_m: float = GOAL_TOL_M) -> bool:
    if final is None:
        return False
    return math.hypot(final[0] - goal[0], final[1] - goal[1]) * RES_M <= tol_m + 1e-9


def is_success(reached: bool, entries: int) -> bool:
    return bool(reached) and entries == 0


# ---- eval-only shortest hazard-free route (route-ratio denominator) ---------
def grid_graph(blocked: np.ndarray, cost: np.ndarray | None = None) -> csr_matrix:
    """8-connected graph over free cells; diagonals need both orthogonals free.

    Edge weight = step length in metres, times mean(cost[a], cost[b]) when a
    cost grid is given (the Stream B planner's step-cost rule). Pure function
    of its inputs: it never reads truth.
    """
    n, m = blocked.shape
    free = ~blocked
    ids = np.arange(n * m).reshape(n, m)
    rows, cols, weights = [], [], []

    def add(src_sl, dst_sl, ok, w):
        rows.append(ids[src_sl][ok])
        cols.append(ids[dst_sl][ok])
        if cost is None:
            weights.append(np.full(int(ok.sum()), w))
        else:
            weights.append(0.5 * (cost[src_sl][ok] + cost[dst_sl][ok]).astype(np.float64) * w)

    # east, south (undirected: added in both directions below)
    add(np.s_[:, :-1], np.s_[:, 1:], free[:, :-1] & free[:, 1:], RES_M)
    add(np.s_[:-1, :], np.s_[1:, :], free[:-1, :] & free[1:, :], RES_M)
    # south-east and south-west, no corner cutting
    se = free[:-1, :-1] & free[1:, 1:] & free[:-1, 1:] & free[1:, :-1]
    add(np.s_[:-1, :-1], np.s_[1:, 1:], se, DIAG_M)
    sw = free[:-1, 1:] & free[1:, :-1] & free[:-1, :-1] & free[1:, 1:]
    add(np.s_[:-1, 1:], np.s_[1:, :-1], sw, DIAG_M)
    r = np.concatenate(rows)
    c = np.concatenate(cols)
    w = np.concatenate(weights)
    return csr_matrix((np.concatenate([w, w]), (np.concatenate([r, c]), np.concatenate([c, r]))),
                      shape=(n * m, n * m))


@dataclass
class SafeRoutes:
    """Single-source shortest hazard-free distances from ``start`` on truth."""
    start: Cell
    dist_m: np.ndarray          # (GRID, GRID) float, inf if unreachable
    pred: np.ndarray            # (GRID*GRID,) predecessor indices, -9999 if none

    def length_to(self, goal: Cell) -> float:
        return float(self.dist_m[goal])

    def path_to(self, goal: Cell) -> list[Cell] | None:
        if not np.isfinite(self.dist_m[goal]):
            return None
        m = self.dist_m.shape[1]
        node = goal[0] * m + goal[1]
        start = self.start[0] * m + self.start[1]
        out = []
        while node != start:
            out.append((int(node // m), int(node % m)))
            node = int(self.pred[node])
            if node < 0:
                return None
        out.append(tuple(self.start))
        return out[::-1]


def safe_routes_from(blocked: np.ndarray, start: Cell) -> SafeRoutes:
    """Dijkstra from ``start`` over the truth footprint mask (EVAL ONLY)."""
    if blocked[start]:
        return SafeRoutes(tuple(start), np.full(blocked.shape, np.inf), np.full(blocked.size, -9999))
    graph = grid_graph(blocked)
    src = start[0] * blocked.shape[1] + start[1]
    dist, pred = dijkstra(graph, directed=False, indices=src, return_predecessors=True)
    return SafeRoutes(tuple(start), dist.reshape(blocked.shape), pred)


def shortest_safe_route(world: World, start: Cell, goal: Cell) -> tuple[float, list[Cell] | None]:
    """(length_m, path) of the shortest hazard-free route on truth; (inf, None) if none."""
    routes = safe_routes_from(hazard_footprint_mask(world), start)
    return routes.length_to(goal), routes.path_to(goal)


def route_ratio(length_m: float, optimal_m: float) -> float | None:
    if not np.isfinite(optimal_m) or optimal_m <= 0:
        return None
    return float(length_m / optimal_m)


# ---- statistics --------------------------------------------------------------
def bootstrap_ci(values, n_resamples: int = 10_000, seed: int = 0,
                 alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean. Deterministic for a given seed."""
    x = np.asarray(values, np.float64)
    if x.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples)
    chunk = max(1, 2_000_000 // x.size)
    for lo in range(0, n_resamples, chunk):
        hi = min(n_resamples, lo + chunk)
        means[lo:hi] = x[rng.integers(0, x.size, size=(hi - lo, x.size))].mean(axis=1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def score_mission(world: World, trajectory: list[Cell], goal: Cell, optimal_m: float,
                  mask: np.ndarray | None = None) -> dict:
    """Truth-based score of one mission from its full trajectory."""
    mask = hazard_footprint_mask(world) if mask is None else mask
    traj = [tuple(map(int, c)) for c in trajectory]
    entries = hazard_entries(traj, mask)
    reached = reached_goal(traj[-1] if traj else None, goal)
    length = path_length_m(traj)
    return {
        "hazard_entries": entries,
        "reached": reached,
        "success": is_success(reached, entries),
        "traj_len_m": length,
        "route_ratio": route_ratio(length, optimal_m) if reached else None,
    }
