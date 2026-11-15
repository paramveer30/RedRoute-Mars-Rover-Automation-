import numpy as np

from backend.app.contracts import GRID, Belief, CostConfig, Pose, SensorConfig
from backend.app.mapping import cost_grid
from backend.app.perception import observe, sensor_cone
from world.stub_world import build_stub_world


def test_stub_world_shapes():
    w = build_stub_world(1)
    for a in (w.elevation, w.slope_deg, w.roughness, w.truth, w.geo_obstacle):
        assert a.shape == (GRID, GRID)
    assert len(w.tile_image) == 10 and len(w.tile_image[0]) == 10


def test_sensor_cone_points_north():
    cone = sensor_cone(Pose(100, 100, 0), SensorConfig())
    assert cone[90, 100]       # 2.5 m north: inside
    assert not cone[110, 100]  # 2.5 m south: outside
    assert not cone[60, 100]   # 10 m north: beyond range


def test_oracle_observe_and_cost():
    w = build_stub_world(2)
    b = Belief.empty()
    res = observe(w, b, Pose(100, 100, 0))
    assert res.ok and res.tiles_seen and b.known.any() and b.version == 1
    cost, blocked = cost_grid(w, b, CostConfig())
    assert cost.shape == blocked.shape == (GRID, GRID)
    assert cost.min() >= 1.0


def test_dropped_frame_leaves_belief_untouched():
    from backend.app.contracts import ObserveFlags
    w, b = build_stub_world(3), Belief.empty()
    res = observe(w, b, Pose(100, 100, 0), flags=ObserveFlags(drop_frame=True))
    assert not res.ok and b.version == 0 and not b.known.any()
    assert np.allclose(b.probs, 0.25)
