"""Cost-map correctness, information boundaries and runtime checks."""
from dataclasses import replace
import json
from time import perf_counter

import numpy as np
import pytest

from backend.app.contracts import BEDROCK, BIG_ROCK, GRID, SAND, SOIL, Belief, CostConfig
from backend.app.mapping import cost_grid, explain_cell
from world.stub_world import build_stub_world


@pytest.fixture
def scene():
    world = build_stub_world(0)
    world.slope_deg.fill(0)
    world.roughness.fill(0)
    world.geo_obstacle.fill(False)
    return world, Belief.empty(), CostConfig()


def test_cost_floor_and_dtypes(scene):
    world, belief, cfg = scene
    belief.known.fill(True)
    belief.probs.fill(-1)  # Defensive floor also protects against malformed upstream values.
    belief.unc.fill(-3)
    cost, blocked = cost_grid(world, belief, cfg)
    assert cost.shape == blocked.shape == (GRID, GRID)
    assert cost.dtype == np.float32
    assert blocked.dtype == np.bool_
    assert np.all(cost >= 1)


def test_slope_limit_before_inflation(scene):
    world, belief, cfg = scene
    world.slope_deg[100, 100] = 25
    world.slope_deg[101, 100] = 15
    world.slope_deg[102, 100] = 20
    _, blocked = cost_grid(world, belief, replace(cfg, inflate_m=0))
    assert blocked[100, 100]
    assert not blocked[101, 100]
    assert not blocked[102, 100]


def test_big_rock_blob_and_isolated_noise(scene):
    world, belief, cfg = scene
    belief.known.fill(True)
    belief.probs[50:52, 50:52] = [0.1, 0, 0, 0.9]
    belief.probs[100, 100] = [0.1, 0, 0, 0.9]
    _, blocked = cost_grid(world, belief, replace(cfg, inflate_m=0))
    assert blocked[50:52, 50:52].all()
    assert not blocked[100, 100]
    assert blocked.sum() == cfg.min_blob_cells


def test_unknown_cells_cannot_complete_semantic_blob(scene):
    world, belief, cfg = scene
    belief.probs[50:52, 50:52] = [0.1, 0, 0, 0.9]
    belief.known[50:52, 50:52] = True
    belief.known[51, 51] = False
    _, blocked = cost_grid(world, belief, replace(cfg, inflate_m=0))
    assert not blocked.any()
    belief.known.fill(False)
    _, blocked = cost_grid(world, belief, replace(cfg, min_blob_cells=1))
    assert not blocked.any()


def test_geometric_obstacles_only_block_after_observation(scene):
    world, belief, cfg = scene
    world.geo_obstacle[100, 100] = True
    _, hidden = cost_grid(world, belief, cfg)
    assert not hidden.any()
    belief.geo_seen[100, 100] = True
    _, seen = cost_grid(world, belief, cfg)
    assert seen[100, 100]
    # A single sensed geometric obstacle survives semantic blob filtering.
    assert seen.sum() == 113


def test_default_inflation_is_six_cell_disk(scene):
    world, belief, cfg = scene
    world.slope_deg[100, 100] = 25
    _, blocked = cost_grid(world, belief, cfg)
    yy, xx = np.ogrid[:GRID, :GRID]
    expected = (yy - 100)**2 + (xx - 100)**2 <= 6**2
    np.testing.assert_array_equal(blocked, expected)
    assert blocked.sum() == 113


def test_custom_inflation_radius_and_edge_clipping(scene):
    world, belief, cfg = scene
    belief.geo_seen[0, 0] = True
    _, blocked = cost_grid(world, belief, replace(cfg, inflate_m=0.5))
    assert blocked.sum() == 6
    assert blocked[0, 2] and blocked[2, 0]
    assert not blocked[-1, -1]


def test_unknown_uses_half_uncertainty_and_zero_semantic_risk(scene):
    world, belief, cfg = scene
    belief.probs[:] = [0, 0, 1, 0]
    belief.unc.fill(0.99)
    costs, _ = cost_grid(world, belief, cfg)
    np.testing.assert_array_equal(costs, np.float32(1 + cfg.w_u * 0.5))
    terms = explain_cell(world, belief, cfg, (100, 100))
    assert terms["S"] == 0
    assert terms["U"] == 0.5


def test_configured_unknown_uncertainty_is_respected(scene):
    world, belief, cfg = scene
    costs, _ = cost_grid(world, belief, replace(cfg, unknown_u=0.2))
    np.testing.assert_allclose(costs, 1 + cfg.w_u * 0.2)


def test_geometry_only_ignores_probabilities_and_uncertainty(scene):
    world, belief, cfg = scene
    world.roughness.fill(0.25)
    world.slope_deg[30, 30] = 25
    belief.geo_seen[160, 160] = True
    cfg = replace(cfg, use_semantics=False)
    first = cost_grid(world, belief, cfg)
    belief.known.fill(True)
    belief.probs[:] = [0, 0, 0, 1]
    belief.unc.fill(1)
    second = cost_grid(world, belief, cfg)
    for old, new in zip(first, second):
        np.testing.assert_array_equal(old, new)
    np.testing.assert_array_equal(second[0], 1 + cfg.w_r * world.roughness)
    assert not second[1][100, 100]


def test_zero_uncertainty_weight_ablation(scene):
    world, belief, cfg = scene
    belief.known.fill(True)
    cfg = replace(cfg, w_u=0)
    first = cost_grid(world, belief, cfg)
    belief.unc.fill(1)
    second = cost_grid(world, belief, cfg)
    for old, new in zip(first, second):
        np.testing.assert_array_equal(old, new)


def test_sand_costs_more_than_soil_and_bedrock(scene):
    world, belief, cfg = scene
    belief.known.fill(True)
    belief.unc.fill(0)
    for col, terrain_class in enumerate((BEDROCK, SOIL, SAND)):
        belief.probs[100, col] = 0
        belief.probs[100, col, terrain_class] = 1
    costs, _ = cost_grid(world, belief, cfg)
    assert costs[100, 2] > costs[100, 1] > costs[100, 0]
    np.testing.assert_allclose(costs[100, :3], [1.4, 1.8, 5])


def test_mapping_never_reads_truth_or_unsensed_geometry(scene):
    world, belief, cfg = scene
    class ForbiddenArray:
        def __array__(self, *args, **kwargs):
            raise AssertionError("mapping attempted to read hidden world data")
        def __getitem__(self, item):
            raise AssertionError("mapping attempted to read hidden world data")
    world.truth = ForbiddenArray()
    world.geo_obstacle = ForbiddenArray()
    costs, blocked = cost_grid(world, belief, cfg)
    assert not blocked.any()
    assert costs.min() >= 1
    assert not explain_cell(world, belief, cfg, (100, 100))["blocked"]


@pytest.mark.parametrize("cause", ["slope", "big_rock", "geo_obstacle"])
def test_explain_cell_direct_and_inflated_block_reasons(scene, cause):
    world, belief, cfg = scene
    if cause == "slope":
        world.slope_deg[100, 100] = 25
    elif cause == "big_rock":
        belief.known[100:102, 100:102] = True
        belief.probs[100:102, 100:102] = [0.1, 0, 0, 0.9]
    else:
        belief.geo_seen[100, 100] = True
    direct = explain_cell(world, belief, cfg, (100, 100))
    assert direct["blocked"]
    assert direct["blocked_reason"] == cause
    near = explain_cell(world, belief, cfg, (100, 96))
    assert near["blocked"]
    assert near["blocked_reason"] == "inflation"
    far = explain_cell(world, belief, cfg, (10, 10))
    assert not far["blocked"]
    assert far["blocked_reason"] is None
    json.dumps(direct)


def test_explain_cell_matches_weighted_cost(scene):
    world, belief, cfg = scene
    cell = (100, 100)
    world.roughness[cell] = 0.4
    belief.known[cell] = True
    belief.probs[cell] = [0.5, 0.2, 0.3, 0]
    belief.unc[cell] = 0.2
    terms = explain_cell(world, belief, cfg, cell)
    expected = max(1, 1 + cfg.w_s*terms["S"] + cfg.w_r*terms["R"] + cfg.w_u*terms["U"])
    assert terms["cost"] == pytest.approx(expected)
    assert terms["slope_deg"] == 0
    with pytest.raises(IndexError):
        explain_cell(world, belief, cfg, (-1, 1))


def test_cost_grid_under_fifteen_ms(scene):
    world, belief, cfg = scene
    rng = np.random.default_rng(8)
    belief.known[:] = rng.random((GRID, GRID)) > 0.2
    probs = rng.random((GRID, GRID, 4), dtype=np.float32)
    belief.probs[:] = probs / probs.sum(axis=-1, keepdims=True)
    belief.probs[80:90, 80:90] = [0.1, 0, 0, 0.9]
    world.slope_deg[30:40, 30:40] = 25
    belief.geo_seen[160, 160] = True
    for _ in range(5):
        cost_grid(world, belief, cfg)
    timings = []
    for _ in range(25):
        start = perf_counter()
        cost_grid(world, belief, cfg)
        timings.append((perf_counter() - start) * 1000)
    median, p95 = np.percentile(timings, [50, 95])
    print(f"cost_grid 200x200: median={median:.3f} ms p95={p95:.3f} ms (25 samples)")
    assert median < 15, f"median cost_grid latency {median:.3f} ms >= 15 ms"


def test_likely_sand_blob_is_blocked_and_inflated(scene):
    world, belief, cfg = scene
    belief.known.fill(True)
    belief.probs[100:102, 100:102] = [0.3, 0, 0.6, 0.1]
    _, raw = cost_grid(world, belief, replace(cfg, inflate_m=0))
    assert raw[100:102, 100:102].all()
    assert raw.sum() == 4
    _, blocked = cost_grid(world, belief, cfg)
    assert blocked[100, 98] and blocked[100, 103]   # 2-cell (0.5 m footprint) inflation
    assert not blocked[100, 97] and not blocked[100, 104]
    terms = explain_cell(world, belief, cfg, (100, 100))
    assert terms["blocked_reason"] == "predicted sand"
    assert terms["blocked_label"] == "blocked: predicted sand"
    assert explain_cell(world, belief, cfg, (100, 98))["blocked_reason"] == "inflation"
    assert not explain_cell(world, belief, cfg, (100, 96))["blocked"]


def test_sand_inflation_matches_footprint_and_rock_keeps_full_clearance(scene):
    world, belief, cfg = scene
    belief.known.fill(True)
    belief.probs[100, 100:102] = belief.probs[101, 100:102] = [0, 0, 1, 0]
    belief.probs[50:52, 50:52] = [0, 0, 0, 1]
    _, blocked = cost_grid(world, belief, cfg)
    yy, xx = np.ogrid[:GRID, :GRID]
    sand = np.zeros((GRID, GRID), bool)
    for r, c in [(100, 100), (100, 101), (101, 100), (101, 101)]:
        sand |= (yy - r)**2 + (xx - c)**2 <= 2**2
    np.testing.assert_array_equal(blocked[80:120, 80:120], sand[80:120, 80:120])
    assert blocked[50, 44] and not blocked[50, 43]       # big rock still 6 cells


def test_sand_threshold_is_inclusive_and_isolated_sand_is_noise(scene):
    world, belief, cfg = scene
    belief.known.fill(True)
    belief.probs[50:52, 50:52] = [0.5, 0, 0.5, 0]       # exactly at threshold
    belief.probs[150, 150] = [0, 0, 1, 0]               # single cell: filtered as noise
    _, blocked = cost_grid(world, belief, replace(cfg, inflate_m=0))
    assert blocked[50:52, 50:52].all()
    assert blocked.sum() == 4


def test_sub_threshold_sand_is_soft_cost_only(scene):
    world, belief, cfg = scene
    belief.known.fill(True)
    belief.unc.fill(0)
    belief.probs[90:110, 90:110] = [0.55, 0, 0.45, 0]
    costs, blocked = cost_grid(world, belief, cfg)
    assert not blocked.any()
    np.testing.assert_allclose(costs[100, 100], 1 + cfg.w_s * (0.55 * 0.2 + 0.45 * 1.0), rtol=1e-6)
    assert explain_cell(world, belief, cfg, (100, 100))["blocked_reason"] is None


def test_unknown_or_geometry_only_sand_never_blocks(scene):
    world, belief, cfg = scene
    belief.probs[:] = [0, 0, 1, 0]
    _, blocked = cost_grid(world, belief, cfg)          # nothing known yet
    assert not blocked.any()
    belief.known.fill(True)
    costs, blocked = cost_grid(world, belief, replace(cfg, use_semantics=False))
    assert not blocked.any()
    np.testing.assert_array_equal(costs, 1)
