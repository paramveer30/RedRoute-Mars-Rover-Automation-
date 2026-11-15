"""One test per state transition of the Rover safety state machine."""
import base64

import numpy as np
import pytest

from backend.app.contracts import GRID, SAND, SOIL, CostConfig
from backend.app.rover import (AUTONOMY_LIMIT_M, T_STALE_S, MissionResult, Rover, State)
from world.stub_world import build_stub_world


@pytest.fixture
def world():
    w = build_stub_world(0)
    w.slope_deg.fill(0)
    w.roughness.fill(0)
    w.geo_obstacle.fill(False)
    w.truth.fill(SOIL)
    return w


def run_until(rover, pred, max_steps=500):
    for _ in range(max_steps):
        rover.step()
        if pred(rover):
            return True
    return False


def test_idle_to_planning_on_target(world):
    r = Rover(world, start=(150, 100))
    assert r.state == State.IDLE
    r.step()
    assert r.state == State.IDLE
    r.set_target((100, 100))
    assert r.state == State.PLANNING


def test_planning_to_driving_bumps_route_version(world):
    r = Rover(world, start=(150, 100))
    r.set_target((100, 100))
    events = r.step()
    assert r.state == State.DRIVING
    assert r.route_version == 1
    assert r.path[0] == (150, 100) and r.path[-1] == (100, 100)
    assert any(e.kind == "route" for e in events)


def test_driving_moves_one_metre_per_tick(world):
    r = Rover(world, start=(150, 100))
    r.set_target((100, 100))
    r.step()
    r.step()
    assert r.cell == (146, 100)
    assert r.odometer_m == pytest.approx(1.0)


def test_driving_to_scanning_every_two_metres(world):
    r = Rover(world, start=(150, 100))
    r.set_target((100, 100))
    r.step()          # plan
    r.step()          # 1 m
    assert r.state == State.DRIVING
    r.step()          # 2 m
    assert r.state == State.SCANNING


def test_scanning_to_driving(world):
    r = Rover(world, start=(150, 100))
    r.set_target((100, 100))
    assert run_until(r, lambda r: r.state == State.SCANNING)
    r.step()
    assert r.state == State.DRIVING


def test_driving_to_idle_on_arrival(world):
    r = Rover(world, start=(150, 100))
    r.set_target((130, 100))
    assert run_until(r, lambda r: r.state == State.IDLE)
    assert abs(r.cell[0] - 130) <= 4
    assert r.path == []


def test_replan_when_hidden_rock_blocks_path(world):
    r = Rover(world, start=(150, 100))
    r.set_target((60, 100))
    r.step()
    v0 = r.route_version
    world.geo_obstacle[120:124, 98:103] = True  # on the straight line, ~7 m ahead
    assert run_until(r, lambda r: r.state == State.REPLANNING, 20)
    r.step()
    assert r.state == State.DRIVING
    assert r.route_version == v0 + 1
    assert r.replans == 1 and len(r.replan_latencies_ms) == 1
    _, blocked = r._grids()
    assert not any(blocked[c] for c in r.path[1:])
    assert run_until(r, lambda r: r.state == State.IDLE)
    assert r.hazard_entries == 0


def test_replan_when_path_cost_rises(world):
    r = Rover(world, start=(150, 100))
    r.set_target((100, 100))
    r.step()
    r._cost, r._blocked = r._grids()
    r._cost = r._cost.copy()
    r._cost[100:140, :] *= 5.0   # everything ahead got much more expensive
    assert r._replan_reason().startswith("path cost +")


def test_planning_to_safe_stop_when_no_route(world):
    world.geo_obstacle[90:110, 90:110] = True
    r = Rover(world, start=(150, 100))
    r.belief.geo_seen[90:110, 90:110] = True
    r.set_target((100, 100))
    r.step()
    assert r.state == State.SAFE_STOP
    assert "no route" in r.safe_stop_reason


def test_watchdog_safe_stop_when_frames_drop(world):
    r = Rover(world, start=(150, 100))
    r.set_target((60, 100))
    r.step()
    r.drop_frames = 100
    assert run_until(r, lambda r: r.state == State.SAFE_STOP, 40)
    assert "observation" in r.safe_stop_reason
    assert r.mission_time - r.last_obs_time > T_STALE_S


def test_corrupt_frame_rejected_then_rescan(world):
    r = Rover(world, start=(150, 100))
    r.set_target((60, 100))
    assert run_until(r, lambda r: r.state == State.SCANNING)
    r.corrupt_frames = 1
    events = r.step()
    assert any(e.kind == "frame_rejected" and e.level == "error" for e in events)
    assert r.state == State.SCANNING   # rescan
    r.step()
    assert r.state == State.DRIVING


def test_autonomy_limit_safe_stop_without_earth_contact(world):
    r = Rover(world, start=(190, 100))
    r.set_target((10, 100))   # 45 m
    r.step()
    r.link_up = False
    assert run_until(r, lambda r: r.state == State.SAFE_STOP)
    assert "Earth contact" in r.safe_stop_reason
    assert r.since_contact_m > AUTONOMY_LIMIT_M


def test_safe_stop_holds_until_operator_resume(world):
    r = Rover(world, start=(150, 100))
    r.set_target((60, 100))
    r.step()
    r.safe_stop("test")
    for _ in range(20):
        r.step()
    assert r.state == State.SAFE_STOP
    assert r.resume()
    assert r.state == State.PLANNING
    r.step()
    assert r.state == State.DRIVING


def test_geometry_only_half_speed(world):
    r = Rover(world, mode="geometry_only", start=(150, 100))
    r.set_target((100, 100))
    r.step()
    r.step()
    assert r.odometer_m == pytest.approx(0.5)


def test_model_offline_switches_mode_and_ignores_semantics(world):
    r = Rover(world, mode="oracle", start=(150, 100))
    r.set_target((100, 100))
    r.step()
    r.set_mode("geometry_only")
    assert r._effective_cfg().use_semantics is False
    events = r.step()
    assert r.mode == "geometry_only"


def test_run_to_completion_result(world):
    r = Rover(world, mode="oracle", cost_cfg=CostConfig())
    res = r.run_to_completion((180, 20), (40, 150), 500)
    assert isinstance(res, MissionResult)
    assert res.reached and res.final_state == "IDLE"
    assert res.hazard_entries == 0
    assert res.trajectory[0] == (180, 20)
    assert res.path_len_m > 0 and res.steps > 0


def test_simulator_counts_hazard_entries_once_per_run(world):
    world.truth[100:110, :] = SAND
    r = Rover(world, mode="geometry_only", cost_cfg=CostConfig(use_semantics=False))
    res = r.run_to_completion((150, 100), (60, 100), 1000)
    assert res.reached
    assert res.hazard_entries == 1


def test_snapshot_matches_tick_schema(world):
    r = Rover(world, start=(150, 100))
    r.set_target((100, 100))
    ev = r.step()
    snap = r.snapshot(ev)
    for key in ("type", "mission_time", "seq", "rover_id", "pose", "state", "battery",
                "route_version", "path", "belief_delta", "perception_mode", "link", "events"):
        assert key in snap
    assert snap["pose"] == {"x": 100.0, "y": 150.0, "heading": 0.0}
    assert snap["path"][0] == [100, 150]
    tiles = snap["belief_delta"]["tiles"]
    assert tiles
    cls = np.frombuffer(base64.b64decode(next(iter(tiles.values()))["cls"]), np.uint8)
    assert cls.size == 400
    assert r.snapshot()["belief_delta"]["tiles"] == {}  # drained


def test_planner_never_reads_truth(world):
    """Scramble truth after the rover starts: plans must be identical."""
    r1 = Rover(world, mode="geometry_only", cost_cfg=CostConfig(use_semantics=False),
               start=(150, 100))
    r1.set_target((60, 120))
    r1.step()
    path1 = list(r1.path)
    world.truth[:] = np.random.default_rng(0).integers(0, 4, (GRID, GRID))
    r2 = Rover(world, mode="geometry_only", cost_cfg=CostConfig(use_semantics=False),
               start=(150, 100))
    r2.set_target((60, 120))
    r2.step()
    assert r2.path == path1


def test_escape_from_own_clearance_buffer_does_not_thrash(world):
    """Rover starts inside the inflated buffer of a seen obstacle: it must plan out once,
    not replan every tick (regression: seed 13 oracle watchdog SAFE_STOP)."""
    world.geo_obstacle[146:149, 103:106] = True
    r = Rover(world, start=(150, 100))
    r.belief.geo_seen[146:149, 103:106] = True
    _, blocked = r._grids()
    assert blocked[150, 100]
    r.set_target((60, 100))
    assert run_until(r, lambda r: r.state in (State.IDLE, State.SAFE_STOP))
    assert r.state == State.IDLE
    assert r.replans <= 2
