import pytest

from backend.app.link import Link, check_precondition, load_light_delay


def test_real_delay_from_cached_horizons():
    info = load_light_delay()
    assert 180 < info["delay_s"] < 1400    # Earth-Mars one-way light time is 3 to 22 min
    assert info["source"]


def test_compression_divides_real_delay():
    link = Link(real_delay_s=600, compression=60)
    assert link.delay_s == pytest.approx(10.0)
    assert link.info(0)["real_delay_min"] == pytest.approx(10.0)


def test_downlink_arrives_after_delay():
    link = Link(real_delay_s=60, compression=60)   # 1 s
    link.send_down(0.0, {"n": 0})
    assert link.recv_down(0.5) == []
    got = link.recv_down(1.0)
    assert [m.body["n"] for m in got] == [0]
    assert link.ground_time == 0.0


def test_blackout_buffers_then_replays_sorted_and_late():
    link = Link(real_delay_s=60, compression=60)   # 1 s
    link.start_blackout(1.0, 3.0)                  # blackout over [1, 4)
    assert link.status(2.0) == "BLACKOUT"
    for t in (1.5, 2.0, 2.5, 3.0):
        link.send_down(t, {"t": t})
        assert link.recv_down(t) == []
    assert link.info(3.0)["buffered"] == 4
    assert link.recv_down(4.0) == []               # restored: flushed, now in flight
    got = link.recv_down(5.0)
    assert [m.body["t"] for m in got] == [1.5, 2.0, 2.5, 3.0]
    assert all(m.late for m in got)
    assert link.ground_time == 3.0


def test_ground_clock_freezes_during_blackout():
    link = Link(real_delay_s=60, compression=60)
    for i in range(4):
        t = i * 0.5
        link.send_down(t, {})
        link.recv_down(t)
    before = link.ground_time
    link.start_blackout(2.0, 5.0)
    for i in range(4, 12):
        t = i * 0.5
        link.send_down(t, {})
        link.recv_down(t)
    assert link.ground_time <= before + 1.0       # only messages already in flight landed
    assert link.info(5.5)["rover_time"] - link.ground_time > 3


def test_uplink_delayed_and_dropped_in_blackout():
    link = Link(real_delay_s=120, compression=60)  # 2 s
    assert link.send_up(0.0, {"type": "stop"}) is not None
    assert link.recv_up(1.0) == []
    assert [m.body["type"] for m in link.recv_up(2.0)] == ["stop"]
    link.start_blackout(3.0, 2.0)
    assert link.send_up(3.5, {"type": "stop"}) is None
    assert link.dropped_uplink == 1


def test_stale_route_version_rejected():
    res = check_precondition({"precondition": {"route_version": 3}}, 5, "DRIVING")
    assert not res.accepted
    assert res.reason == "route v3 superseded by v5"
    assert check_precondition({"precondition": {"route_version": 5}}, 5, "DRIVING").accepted
    assert check_precondition({}, 5, "DRIVING").accepted


def test_expected_state_precondition():
    res = check_precondition({"precondition": {"expected_state": "SAFE_STOP"}}, 1, "DRIVING")
    assert not res.accepted and "SAFE_STOP" in res.reason
