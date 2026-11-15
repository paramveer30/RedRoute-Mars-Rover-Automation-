import time

import numpy as np
import pytest

from backend.app.contracts import GRID, RES_M
from backend.app.planner import path_cost, path_length_m, plan


def grid(n=20, value=1.0):
    return np.full((n, n), value, np.float32), np.zeros((n, n), bool)


def assert_valid(path, blocked, start, goal):
    assert path[0] == start and path[-1] == goal
    for (r0, c0), (r1, c1) in zip(path, path[1:]):
        assert max(abs(r1 - r0), abs(c1 - c0)) == 1
        if r0 != r1 and c0 != c1:
            assert not blocked[r0, c1] and not blocked[r1, c0]
    assert not any(blocked[p] for p in path)


def test_straight_line_on_uniform_cost():
    cost, blocked = grid()
    path = plan(cost, blocked, (10, 2), (10, 17))
    assert path == [(10, c) for c in range(2, 18)]
    assert path_length_m(path) == pytest.approx(15 * RES_M)


def test_diagonal_on_uniform_cost():
    cost, blocked = grid()
    path = plan(cost, blocked, (0, 0), (9, 9))
    assert len(path) == 10
    assert path_cost(cost, path) == pytest.approx(9 * RES_M * np.sqrt(2))


def test_wall_with_gap():
    cost, blocked = grid()
    blocked[:, 10] = True
    blocked[15, 10] = False
    path = plan(cost, blocked, (2, 2), (2, 17))
    assert path is not None
    assert (15, 10) in path
    assert_valid(path, blocked, (2, 2), (2, 17))


def test_no_path_returns_none():
    cost, blocked = grid()
    blocked[:, 10] = True
    assert plan(cost, blocked, (2, 2), (2, 17)) is None


def test_blocked_start_or_goal_returns_none():
    cost, blocked = grid()
    blocked[5, 5] = True
    assert plan(cost, blocked, (5, 5), (0, 0)) is None
    assert plan(cost, blocked, (0, 0), (5, 5)) is None


def test_out_of_bounds_returns_none():
    cost, blocked = grid()
    assert plan(cost, blocked, (0, 0), (25, 0)) is None


def test_start_equals_goal():
    cost, blocked = grid()
    assert plan(cost, blocked, (3, 3), (3, 3)) == [(3, 3)]


def test_prefers_cheap_detour_over_expensive_straight_line():
    cost, blocked = grid()
    cost[3:17, 10] = 50.0  # expensive band across the direct line, open at the ends
    path = plan(cost, blocked, (10, 2), (10, 17))
    assert all(cost[p] == 1.0 for p in path)
    straight = [(10, c) for c in range(2, 18)]
    assert path_cost(cost, path) < path_cost(cost, straight)


def test_crosses_band_when_detour_is_too_long():
    cost, blocked = grid()
    cost[:, 10] = 2.0  # cannot be avoided; must be crossed once
    path = plan(cost, blocked, (10, 2), (10, 17))
    assert sum(cost[p] == 2.0 for p in path) == 1


def test_path_never_touches_blocked_cells():
    rng = np.random.default_rng(0)
    cost = rng.uniform(1, 5, (60, 60)).astype(np.float32)
    blocked = rng.random((60, 60)) < 0.25
    blocked[0, 0] = blocked[59, 59] = False
    path = plan(cost, blocked, (0, 0), (59, 59))
    if path is not None:
        assert_valid(path, blocked, (0, 0), (59, 59))


def test_no_diagonal_corner_cutting():
    cost, blocked = grid(3)
    # Only the diagonal (0,0)->(1,1) would connect; both orthogonals blocked.
    blocked[0, 1] = blocked[1, 0] = True
    blocked[1, 2] = blocked[2, 1] = True
    blocked[0, 2] = blocked[2, 0] = True
    assert plan(cost, blocked, (0, 0), (2, 2)) is None
    # One orthogonal open is not enough either: the diagonal still cuts a corner.
    cost, blocked = grid(3)
    blocked[0, 1] = True
    path = plan(cost, blocked, (0, 0), (1, 1))
    assert path == [(0, 0), (1, 0), (1, 1)]


def test_optimal_matches_dijkstra_on_random_grid():
    rng = np.random.default_rng(1)
    n = 25
    cost = rng.uniform(1, 10, (n, n)).astype(np.float32)
    blocked = np.zeros((n, n), bool)
    path = plan(cost, blocked, (0, 0), (n - 1, n - 1))
    # Reference: Dijkstra over the same graph.
    import heapq
    dist = np.full((n, n), np.inf)
    dist[0, 0] = 0
    heap = [(0.0, 0, 0)]
    while heap:
        d, r, c = heapq.heappop(heap)
        if d > dist[r, c]:
            continue
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < n:
                    length = RES_M * (np.sqrt(2) if dr and dc else 1)
                    nd = d + 0.5 * (cost[r, c] + cost[nr, nc]) * length
                    if nd < dist[nr, nc]:
                        dist[nr, nc] = nd
                        heapq.heappush(heap, (nd, nr, nc))
    assert path_cost(cost, path) == pytest.approx(dist[n - 1, n - 1], rel=1e-5)


def test_full_grid_speed():
    rng = np.random.default_rng(2)
    cost = rng.uniform(1, 3, (GRID, GRID)).astype(np.float32)
    blocked = np.zeros((GRID, GRID), bool)
    blocked[50:150, 100] = True
    t0 = time.perf_counter()
    path = plan(cost, blocked, (190, 10), (10, 190))
    elapsed = time.perf_counter() - t0
    assert path is not None
    assert elapsed < 1.0, f"{elapsed * 1000:.0f} ms"
