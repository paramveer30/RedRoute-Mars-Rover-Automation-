"""Cached model perception, semantic information boundaries, and fusion state."""
import gc
from time import perf_counter
import weakref

import numpy as np
import pytest

from backend.app.contracts import GRID, TILE, Belief, ObserveFlags, Pose, SensorConfig
from backend.app import perception
from world.stub_world import build_stub_world


@pytest.fixture
def scene(monkeypatch, tmp_path):
    world = build_stub_world(0)
    world.tile_image = [["m3_0000" for _ in range(10)] for _ in range(10)]
    world.geo_obstacle.fill(False)
    cache = tmp_path / "predictions.npz"
    probs = np.broadcast_to(np.array([.05, .05, .85, .05], np.float32), (1, TILE, TILE, 4)).copy()
    np.savez_compressed(cache, ids=np.array(["m3_0000"]), probs=probs, temperature=1.0)
    monkeypatch.setattr(perception, "PREDS_NPZ", cache)
    return world, Belief.empty(), Pose(100, 100, 0), SensorConfig(), probs


def test_model_cone_and_probability_contract(scene):
    world, belief, pose, sensor, _ = scene
    cone = perception.sensor_cone(pose, sensor)
    result = perception.observe(world, belief, pose, sensor, mode="model")
    assert result.ok and result.tiles_seen
    np.testing.assert_array_equal(belief.known, cone)
    np.testing.assert_allclose(belief.probs.sum(axis=-1), 1, atol=1e-6)
    assert ((belief.unc >= 0) & (belief.unc <= 1)).all()
    np.testing.assert_array_equal(belief.probs[~cone], np.full((np.sum(~cone), 4), .25, np.float32))
    assert belief.version == 1


def test_identical_repeat_preserves_probability_and_accumulates_weight(scene):
    world, belief, pose, sensor, _ = scene
    perception.observe(world, belief, pose, sensor, mode="model")
    first = belief.probs.copy()
    weights = perception._weights_for(belief).copy()
    perception.observe(world, belief, pose, sensor, mode="model")
    np.testing.assert_allclose(belief.probs, first, atol=1e-7)
    np.testing.assert_allclose(perception._weights_for(belief), 2 * weights, atol=1e-6)
    assert belief.version == 2


def test_confident_prediction_dominates_uniform(scene, monkeypatch):
    world, belief, pose, sensor, probs = scene
    confident = probs[0].copy()
    monkeypatch.setattr(perception, "_predictions", lambda: {"m3_0000": confident})
    perception.observe(world, belief, pose, sensor, mode="model")
    monkeypatch.setattr(perception, "_predictions", lambda: {"m3_0000": np.full((20, 20, 4), .25)})
    perception.observe(world, belief, pose, sensor, mode="model")
    p = belief.probs[90, 100]
    assert np.linalg.norm(p - confident[0, 0]) < np.linalg.norm(p - .25)


def test_fusion_sidecar_is_identity_isolated_and_released():
    first, second = Belief.empty(), Belief.empty()
    assert perception._weights_for(first) is not perception._weights_for(second)
    key = id(first)
    reference = weakref.ref(first)
    del first
    gc.collect()
    assert reference() is None and key not in perception._fusion_state


def test_fusion_state_resets_with_new_buffer_or_cleared_known():
    belief = Belief.empty()
    belief.known.fill(True)
    perception._weights_for(belief).fill(10)
    belief.known.fill(False)
    assert not perception._weights_for(belief).any()
    belief.known.fill(True)
    perception._weights_for(belief).fill(10)
    belief.probs = belief.probs.copy()
    assert not perception._weights_for(belief).any()


@pytest.mark.parametrize("mode", ["model", "oracle", "offline", "geometry_only"])
def test_offline_flag_precedes_semantics_and_truth(scene, mode):
    world, belief, pose, sensor, _ = scene
    class ForbiddenTruth:
        def __getitem__(self, item):
            raise AssertionError("offline mode read truth")
    world.truth = ForbiddenTruth()
    world.geo_obstacle[90, 100] = True
    before = belief.probs.copy(), belief.unc.copy(), belief.known.copy()
    result = perception.observe(world, belief, pose, sensor,
                                flags=ObserveFlags(model_offline=True), mode=mode)
    assert result.ok and belief.geo_seen[90, 100] and belief.version == 1
    for old, new in zip(before, (belief.probs, belief.unc, belief.known)):
        np.testing.assert_array_equal(old, new)


@pytest.mark.parametrize("mode", ["offline", "geometry_only"])
def test_geometry_mode_preserves_prior_semantics_across_transitions(scene, mode):
    world, belief, pose, sensor, _ = scene
    perception.observe(world, belief, pose, sensor, mode=mode)
    assert not belief.known.any()
    perception.observe(world, belief, pose, sensor, mode="model")
    before = belief.probs.copy(), belief.unc.copy(), belief.known.copy()
    perception.observe(world, belief, Pose(20, 20, 180), sensor, mode=mode)
    for old, new in zip(before, (belief.probs, belief.unc, belief.known)):
        np.testing.assert_array_equal(old, new)
    perception.observe(world, belief, pose, sensor, mode="model")
    np.testing.assert_allclose(belief.probs, before[0])


@pytest.mark.parametrize("flag,reason", [(ObserveFlags(drop_frame=True), "dropped"),
                                         (ObserveFlags(corrupt_frame=True), "corrupt_frame")])
def test_rejected_frames_do_not_update_belief(scene, flag, reason):
    world, belief, pose, sensor, _ = scene
    world.geo_obstacle.fill(True)
    result = perception.observe(world, belief, pose, sensor, flags=flag, mode="model")
    assert not result.ok and result.reason == reason and result.latency_ms >= 0
    assert belief.version == 0 and not belief.known.any() and not belief.geo_seen.any()
    np.testing.assert_array_equal(belief.probs, .25)


def test_missing_prediction_fails_without_partial_semantic_updates(scene):
    world, belief, pose, sensor, _ = scene
    world.tile_image[5][5] = "m3_missing"
    world.geo_obstacle[90, 100] = True
    result = perception.observe(world, belief, pose, sensor, mode="model")
    assert not result.ok and result.reason.startswith("model_error:")
    assert not belief.known.any()
    np.testing.assert_array_equal(belief.probs, .25)
    assert belief.geo_seen[90, 100] and belief.version == 1


def test_model_never_reads_truth(scene):
    world, belief, pose, sensor, _ = scene
    class ForbiddenTruth:
        def __getitem__(self, item):
            raise AssertionError("model mode read truth")
    world.truth = ForbiddenTruth()
    assert perception.observe(world, belief, pose, sensor, mode="model").ok


def test_live_branch_called_once_per_unique_source_and_errors_caught(scene, monkeypatch):
    world, belief, pose, sensor, probs = scene
    calls = []
    def live(source, blur):
        calls.append((source, blur))
        return probs[0]
    monkeypatch.setattr(perception, "_predict_live", live)
    result = perception.observe(world, belief, pose, sensor, ObserveFlags(blur=.8), mode="model")
    assert result.ok and calls == [("m3_0000", .8)]
    def broken(*args):
        raise RuntimeError("inference failed")
    monkeypatch.setattr(perception, "_predict_live", broken)
    fresh = Belief.empty()
    result = perception.observe(world, fresh, pose, sensor, ObserveFlags(blur=.8), mode="model")
    assert not result.ok and "inference failed" in result.reason
    assert not fresh.known.any()


def test_oracle_still_works_and_resets_previous_fusion(scene):
    world, belief, pose, sensor, _ = scene
    perception.observe(world, belief, pose, sensor, mode="model")
    world.truth.fill(1)
    assert perception.observe(world, belief, pose, sensor, mode="oracle").ok
    np.testing.assert_array_equal(belief.probs[90, 100], [0, 1, 0, 0])
    assert not perception._weights_for(belief).any()
    assert perception.observe(world, belief, pose, sensor, mode="model").ok
    np.testing.assert_allclose(belief.probs[90, 100], [.05, .05, .85, .05])


def test_frame_id_is_straight_ahead_and_clamped(scene):
    world, belief, _, sensor, _ = scene
    world.tile_image[4][5] = "m3_ahead"
    result = perception.observe(world, belief, Pose(110, 110, 0), sensor, mode="offline")
    assert result.frame_image_id == "m3_ahead"
    world.tile_image[0][0] = "m3_edge"
    result = perception.observe(world, belief, Pose(1, 1, 315), sensor, mode="offline")
    assert result.frame_image_id == "m3_edge"


def test_cached_observation_under_twenty_ms(scene):
    world, belief, pose, sensor, _ = scene
    for _ in range(3):
        perception.observe(world, belief, pose, sensor, mode="model")
    timings = []
    for _ in range(20):
        started = perf_counter()
        result = perception.observe(world, belief, pose, sensor, mode="model")
        assert result.ok
        timings.append((perf_counter() - started) * 1000)
    median, p95 = np.percentile(timings, [50, 95])
    print(f"cached observe median={median:.3f} ms p95={p95:.3f} ms")
    assert median < 20
