"""Harness: scenarios, fake rover, real-rover guard and an end-to-end smoke run."""
import json
import math

import pytest

from backend.app.contracts import RES_M, CostConfig
from eval import harness, metrics
from eval.rover_api import FakeRover, MissionResult, as_mission_result
from world.builder import build_world


def test_scenarios_deterministic_valid_and_separated():
    for seed in (0, 1, 2):
        a, b = harness.make_scenario(seed), harness.make_scenario(seed)
        assert a == b
        world = build_world(seed)
        mask = metrics.hazard_footprint_mask(world)
        assert not mask[a.start] and not mask[a.goal]
        assert math.hypot(a.goal[0] - a.start[0], a.goal[1] - a.start[1]) * RES_M >= 30.0
        assert math.isfinite(a.optimal_m) and a.optimal_m >= a.straight_m - 1e-9


def test_fake_rover_returns_full_trajectory():
    seed = 1
    sc = harness.make_scenario(seed)
    res = FakeRover(build_world(seed), "oracle", CostConfig()).run_to_completion(sc.start, sc.goal, 300)
    assert isinstance(res, MissionResult)
    assert res.trajectory[0] == sc.start
    for p, q in zip(res.trajectory, res.trajectory[1:]):
        assert max(abs(p[0] - q[0]), abs(p[1] - q[1])) == 1       # every traversed cell
    assert res.observe_calls == res.steps == len(res.replan_latencies_ms)
    assert res.reached == metrics.reached_goal(res.trajectory[-1], sc.goal)


def test_fake_rover_never_reads_truth(monkeypatch):
    world = build_world(2)
    sc = harness.make_scenario(2, world)

    class Guard:
        def __getattr__(self, name):
            raise AssertionError("FakeRover read World.truth in model mode")

        def __getitem__(self, item):
            raise AssertionError("FakeRover read World.truth in model mode")

    world.truth = Guard()
    FakeRover(world, "model", CostConfig()).run_to_completion(sc.start, sc.goal, 5)


def test_as_mission_result_rejects_missing_fields():
    class Partial:
        reached, final_state, path_len_m, steps, replans, replan_latencies_ms = True, "IDLE", 1.0, 1, 0, []

    with pytest.raises(TypeError, match="trajectory"):
        as_mission_result(Partial())


def test_real_rover_missing_exits_clearly():
    try:
        from backend.app.rover import Rover  # noqa: F401
    except ImportError:
        with pytest.raises(SystemExit, match="does not exist yet"):
            harness.main(["--rover", "real", "--n", "1", "--workers", "1"])
    else:
        pytest.skip("Stream B's Rover exists")


def test_fake_refuses_to_write_real_results():
    with pytest.raises(SystemExit, match="refusing"):
        harness.main(["--rover", "fake", "--n", "1", "--out", str(harness.REAL_OUT)])


def test_end_to_end_fake_smoke(tmp_path):
    summary = harness.main(["--rover", "fake", "--n", "4", "--workers", "2", "--out", str(tmp_path)])
    data = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    md = (tmp_path / "results.md").read_text(encoding="utf-8")
    assert data["fake"] and "NOT RedRoute performance" in md
    assert set(data["configs"]) == set(harness.CONFIGS)
    for cfg in data["configs"].values():
        assert cfg["n"] == 4 and cfg["n_errors"] == 0
        assert 0.0 <= cfg["success_rate"] <= 1.0
        lo, hi = cfg["success_rate_ci95"]
        assert lo <= cfg["success_rate"] <= hi
    # identical scenarios across configs
    by_seed = {}
    for m in data["missions"]:
        by_seed.setdefault(m["seed"], set()).add((tuple(m["start"]), tuple(m["goal"])))
    assert sorted(by_seed) == [0, 1, 2, 3] and all(len(v) == 1 for v in by_seed.values())
    assert summary["throughput"]["missions"] == 16
