"""Eval metric definitions on synthetic worlds and trajectories."""
import math

import numpy as np
import pytest

from backend.app.contracts import BEDROCK, BIG_ROCK, GRID, NO_LABEL, RES_M, SAND, SOIL, World
from eval import metrics as M


def flat_world(fill=SOIL) -> World:
    return World(seed=0, elevation=np.zeros((GRID, GRID), np.float32),
                 slope_deg=np.zeros((GRID, GRID), np.float32),
                 roughness=np.zeros((GRID, GRID), np.float32),
                 truth=np.full((GRID, GRID), fill, np.uint8),
                 geo_obstacle=np.zeros((GRID, GRID), bool),
                 tile_image=[["m3_0"] * 10 for _ in range(10)])


def straight(r0, c0, r1, c1):
    n = max(abs(r1 - r0), abs(c1 - c0))
    return [(round(r0 + (r1 - r0) * t / n), round(c0 + (c1 - c0) * t / n)) for t in range(n + 1)]


def test_contiguous_run_counts_once():
    w = flat_world()
    w.truth[100, 90:110] = SAND               # long band along the path
    mask = M.hazard_footprint_mask(w)
    assert M.hazard_entries(straight(100, 50, 100, 150), mask) == 1


def test_two_separate_runs_count_twice():
    w = flat_world()
    w.truth[100, 80] = SAND
    w.truth[100, 120] = SAND
    assert M.hazard_entries(straight(100, 50, 100, 150), M.hazard_footprint_mask(w)) == 2


def test_footprint_overlap_radius_two_cells():
    w = flat_world()
    w.truth[102, 100] = BIG_ROCK               # 2 cells beside the path: inside footprint
    mask = M.hazard_footprint_mask(w)
    assert M.hazard_entries(straight(100, 50, 100, 150), mask) == 1
    w2 = flat_world()
    w2.truth[103, 100] = BIG_ROCK              # 3 cells away: outside footprint
    assert M.hazard_entries(straight(100, 50, 100, 150), M.hazard_footprint_mask(w2)) == 0
    w3 = flat_world()
    w3.truth[102, 102] = SAND                  # diagonal (2,2): 2.83 cells > radius 2
    assert not M.hazard_footprint_mask(w3)[100, 100]


def test_unlabelled_and_safe_classes_are_not_hazards():
    w = flat_world()
    w.truth[:, 100] = NO_LABEL
    w.truth[:, 120] = BEDROCK
    assert M.hazard_entries(straight(100, 50, 100, 150), M.hazard_footprint_mask(w)) == 0


def test_slope_above_limit_is_hazard():
    w = flat_world()
    w.slope_deg[100, 100] = M.HAZARD_SLOPE_DEG + 0.1
    w.slope_deg[100, 120] = M.HAZARD_SLOPE_DEG     # at the limit: allowed
    assert M.hazard_entries(straight(100, 50, 100, 150), M.hazard_footprint_mask(w)) == 1


def test_jump_between_ticks_cannot_skip_hazard():
    w = flat_world()
    w.truth[100, 100] = SAND
    mask = M.hazard_footprint_mask(w, radius_cells=0)
    # Diagonal 2-cell-per-tick jumps never sample (100,100), but the segment
    # (99,99)->(101,101) passes straight through it.
    coarse = [(95 + 2 * k, 95 + 2 * k) for k in range(6)]
    assert (100, 100) not in coarse
    assert M.hazard_entries(coarse, mask) == 1
    # A multi-cell straight jump over a band is caught too.
    w.truth[50, 60] = SAND
    assert M.hazard_entries([(50, 40), (50, 80)], M.hazard_footprint_mask(w, radius_cells=0)) == 1


def test_diagonal_crossing_with_footprint_between_ticks():
    w = flat_world()
    w.truth[110, 92] = SAND                    # 2 cells off the line (100,80)->(120,100)
    mask = M.hazard_footprint_mask(w)
    assert not mask[100, 80] and not mask[120, 100]
    assert M.hazard_entries([(100, 80), (120, 100)], mask) == 1


def test_diagonal_edge_grazing_counts_once():
    w = flat_world()
    for k in range(20):                        # staircase hazard edge parallel to the path
        w.truth[100 + k, 103 + k] = SAND
    mask = M.hazard_footprint_mask(w)
    assert M.hazard_entries([(100 + k, 100 + k) for k in range(20)], mask) == 0
    w.truth[100 + np.arange(20), 102 + np.arange(20)] = SAND
    assert M.hazard_entries([(100 + k, 100 + k) for k in range(20)], M.hazard_footprint_mask(w)) == 1


def test_line_cells_contiguous_and_complete():
    for a, b in [((0, 0), (7, 3)), ((10, 10), (2, 15)), ((5, 5), (5, 0)), ((3, 3), (3, 3))]:
        cells = M.line_cells(a, b)
        assert cells[0] == a and cells[-1] == b
        assert len(cells) == max(abs(a[0] - b[0]), abs(a[1] - b[1])) + 1
        for p, q in zip(cells, cells[1:]):
            assert max(abs(p[0] - q[0]), abs(p[1] - q[1])) == 1


def test_out_of_grid_trajectory_rejected():
    with pytest.raises(ValueError):
        M.hazard_entries([(0, 0), (-1, 0)], np.zeros((GRID, GRID), bool))


def test_success_logic():
    assert M.reached_goal((100, 104), (100, 100))          # exactly 1 m
    assert not M.reached_goal((100, 105), (100, 100))
    assert not M.reached_goal(None, (100, 100))
    assert M.is_success(True, 0)
    assert not M.is_success(True, 1)
    assert not M.is_success(False, 0)


def test_score_mission_success_and_failure():
    w = flat_world()
    goal = (100, 150)
    traj = straight(100, 50, 100, 150)
    s = M.score_mission(w, traj, goal, optimal_m=25.0)
    assert s["success"] and s["reached"] and s["hazard_entries"] == 0
    assert s["route_ratio"] == pytest.approx(1.0)
    w.truth[100, 70] = SAND
    s = M.score_mission(w, traj, goal, optimal_m=25.0)
    assert s["reached"] and not s["success"] and s["hazard_entries"] == 1
    s = M.score_mission(flat_world(), straight(100, 50, 100, 120), goal, optimal_m=25.0)
    assert not s["reached"] and s["route_ratio"] is None


def test_route_ratio_is_one_for_optimal_path():
    w = flat_world()
    w.truth[60:140, 98:102] = SAND            # wall forcing a detour
    start, goal = (100, 60), (100, 140)
    length, path = M.shortest_safe_route(w, start, goal)
    assert path[0] == start and path[-1] == goal
    assert length > (140 - 60) * RES_M        # detour is longer than the straight line
    s = M.score_mission(w, path, goal, length)
    assert s["hazard_entries"] == 0 and s["success"]
    assert s["route_ratio"] == pytest.approx(1.0)


def test_shortest_route_open_ground_matches_octile_distance():
    length, _ = M.shortest_safe_route(flat_world(), (10, 10), (40, 30))
    assert length == pytest.approx((20 * math.sqrt(2) + 10) * RES_M)


def test_shortest_route_unreachable_and_blocked_start():
    w = flat_world()
    w.truth[:, 100] = BIG_ROCK
    assert M.shortest_safe_route(w, (100, 50), (100, 150)) == (math.inf, None)
    assert M.shortest_safe_route(w, (100, 99), (100, 20))[1] is None


def test_bootstrap_deterministic_and_sane():
    x = np.r_[np.ones(30), np.zeros(70)]
    a = M.bootstrap_ci(x, seed=0)
    assert a == M.bootstrap_ci(x, seed=0)
    y = np.random.default_rng(7).normal(size=50)    # continuous: seed changes the CI
    assert M.bootstrap_ci(y, seed=0) == M.bootstrap_ci(y, seed=0)
    assert M.bootstrap_ci(y, seed=0) != M.bootstrap_ci(y, seed=1)
    assert a[0] < 0.3 < a[1]
    assert 0.18 < a[0] and a[1] < 0.42
    assert M.bootstrap_ci(np.ones(10)) == (1.0, 1.0)
