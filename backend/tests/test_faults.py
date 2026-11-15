"""Each of the seven faults produces its expected state or event."""
import pytest

from backend.app import api, faults
from backend.app.contracts import SOIL
from backend.app.link import Link
from backend.app.missionlog import MissionLog
from backend.app.rover import Rover, State
from world.stub_world import build_stub_world


@pytest.fixture
def world():
    w = build_stub_world(0)
    w.slope_deg.fill(0)
    w.roughness.fill(0)
    w.geo_obstacle.fill(False)
    w.truth.fill(SOIL)
    return w


@pytest.fixture
def mission(world):
    rover = Rover(world, start=(190, 100))
    link = Link(real_delay_s=60, compression=60)   # 1 s each way
    m = api.Mission(id="t", seed=0, world=world, rover=rover, link=link, run_id=-1)
    rover.set_target((20, 100))
    for _ in range(3):
        api.tick(m)
    assert rover.state in (State.DRIVING, State.SCANNING)
    return m


def run(m, n, until=None):
    events = []
    for _ in range(n):
        delivered, payload = api.tick(m)
        events += payload["events"]
        if until and until(m.rover):
            break
    return events


def kinds(events):
    return [e["kind"] for e in events]


def test_hidden_rock_detected_blocked_replan(mission):
    v0 = mission.rover.route_version
    res = faults.inject(mission.rover, mission.link, "hidden_rock")
    r0, c0 = res["cell"]
    assert mission.world.geo_obstacle[r0, c0]
    events = run(mission, 12)
    assert mission.rover.belief.geo_seen[r0, c0]
    assert "replan_trigger" in kinds(events)
    assert mission.rover.route_version > v0
    assert mission.rover.hazard_entries == 0
    assert (r0, c0) not in mission.rover.path


def test_drop_frame_keeps_map_then_watchdog_safe_stop(mission):
    known_before = mission.rover.belief.known.sum()
    faults.inject(mission.rover, mission.link, "drop_frame")
    events = run(mission, 40, until=lambda r: r.state == State.SAFE_STOP)
    assert mission.rover.state == State.SAFE_STOP
    assert "observation" in mission.rover.safe_stop_reason
    assert "frame_dropped" in kinds(events)
    assert mission.rover.belief.known.sum() >= known_before   # last map kept


def test_corrupt_frame_rejected_and_rescan(mission):
    faults.inject(mission.rover, mission.link, "corrupt_frame")
    events = run(mission, 8)
    rejected = [e for e in events if e["kind"] == "frame_rejected"]
    assert len(rejected) == 1 and rejected[0]["level"] == "error"
    assert mission.rover.state != State.SAFE_STOP


def test_low_confidence_sets_blur_flags(mission):
    faults.inject(mission.rover, mission.link, "low_confidence")
    assert mission.rover.blur_frames > 0
    assert mission.rover.blur_level == pytest.approx(faults.BLUR_LEVEL)
    run(mission, 1, until=lambda r: r.state == State.SCANNING)
    # oracle mode ignores blur; the flag is still passed to observe()
    run(mission, 6, until=lambda r: r.flags.blur > 0)
    assert mission.rover.flags.blur == pytest.approx(faults.BLUR_LEVEL)


def test_model_offline_geometry_only_half_speed(mission):
    faults.inject(mission.rover, mission.link, "model_offline")
    assert mission.rover.mode == "geometry_only"
    run(mission, 3, until=lambda r: r.state == State.DRIVING)
    odo = mission.rover.odometer_m
    mission.rover.step() if mission.rover.state == State.DRIVING else None
    assert mission.rover.odometer_m - odo <= 0.5 + 1e-9
    snap = mission.rover.snapshot()
    assert snap["perception_mode"] == "geometry_only"


def test_blackout_clocks_diverge_buffer_and_ordered_late_replay(mission):
    faults.inject(mission.rover, mission.link, "blackout", duration_s=4.0)
    delivered_during = []
    for _ in range(7):   # 3.5 s of the 4 s blackout
        d, _ = api.tick(mission)
        delivered_during += d
        assert not mission.rover.link_up
    info = mission.link.info(mission.rover.mission_time)
    assert info["buffered"] >= 6
    assert info["rover_time"] - info["ground_time"] > 3
    after = []
    for _ in range(6):
        d, _ = api.tick(mission)
        after += d
    late = [p for p in after if p["late"]]
    assert len(late) >= 6
    times = [p["mission_time"] for p in late]
    assert times == sorted(times)
    assert mission.rover.link_up


def test_long_blackout_hits_autonomy_limit(mission):
    faults.inject(mission.rover, mission.link, "blackout", duration_s=60.0)
    run(mission, 120, until=lambda r: r.state == State.SAFE_STOP)
    assert mission.rover.state == State.SAFE_STOP
    assert "Earth contact" in mission.rover.safe_stop_reason


def test_stale_command_rejected_with_reason(mission):
    mission.rover.route_version = 5
    res = faults.inject(mission.rover, mission.link, "stale_command", route_version=3)
    assert res["sent"]
    events = run(mission, 4)
    rejected = [e for e in events if e["kind"] == "command_rejected"]
    assert rejected and "route v3 superseded by v5" in rejected[0]["msg"]
    assert mission.rover.state != State.IDLE   # the stale 'stop' did not take effect


def test_unknown_fault_raises(mission):
    with pytest.raises(ValueError):
        faults.inject(mission.rover, mission.link, "meteor")


def test_mission_log_records_routes_events_telemetry(world, tmp_path):
    log = MissionLog(tmp_path / "m.sqlite")
    run_id = log.start("t", 0, "oracle")
    r = Rover(world, start=(190, 100))
    r.set_target((150, 100))
    for _ in range(10):
        log.record(run_id, r, r.step())
    assert log.query("SELECT COUNT(*) FROM telemetry WHERE run_id=?", (run_id,))[0][0] == 10
    routes = log.query("SELECT route_version, reason FROM route WHERE run_id=?", (run_id,))
    assert routes[0] == (1, "initial")
    assert log.query("SELECT COUNT(*) FROM event WHERE kind='route'")[0][0] >= 1
    log.close()
