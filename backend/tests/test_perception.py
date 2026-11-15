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


def test_live_predict_uses_actual_degraded_pixels_and_normalizes(monkeypatch, tmp_path):
    import torch
    from ml import infer
    from world.warp import homography_for
    checker = ((np.indices((1024, 1024)).sum(axis=0) // 8 % 2) * 255).astype(np.uint8)
    inputs = []
    class FakeModel(torch.nn.Module):
        def forward(self, tensor):
            inputs.append(tensor.clone())
            x = tensor[:, :1]
            return torch.cat((x, -x, torch.zeros_like(x), torch.ones_like(x)), dim=1)
    monkeypatch.setattr(infer, "_dataset", lambda: [{"image": checker}])
    monkeypatch.setattr(infer, "get_model", lambda: FakeModel())
    monkeypatch.setattr(infer, "CALIBRATION", tmp_path / "absent.json")
    body = np.zeros((1024, 1024), np.uint8)
    body[700:] = 1
    monkeypatch.setattr(infer, "_homographies", lambda: {"m3_0000": homography_for(body)})
    for blur, noise in [(0, 0), (.8, 0), (0, .5)]:
        result = infer.predict_tile("m3_0000", blur=blur, noise=noise)
        assert result.shape == (20, 20, 4) and result.dtype == np.float32
        np.testing.assert_allclose(result.sum(-1), 1, atol=1e-6)
    assert inputs[1].std() < inputs[0].std()
    assert not torch.equal(inputs[2], inputs[0])
    with pytest.raises(ValueError):
        infer.predict_tile("m3_0000", blur=float("nan"))
    with pytest.raises(ValueError):
        infer.predict_tile("m3_0001")


def test_temperature_reads_validation_calibration_and_reloads(monkeypatch, tmp_path):
    from ml import infer
    calibration = tmp_path / "calibration.json"
    monkeypatch.setattr(infer, "CALIBRATION", calibration)
    assert infer.temperature() == 1
    calibration.write_text('{"T": 1.5}', encoding="utf-8")
    assert infer.temperature() == 1.5
    calibration.write_text('{"T": 2.25}', encoding="utf-8")
    assert infer.temperature() == 2.25
    calibration.write_text('{"T": -1}', encoding="utf-8")
    with pytest.raises(ValueError):
        infer.temperature()


def test_model_mode_on_real_world_layout(scene, monkeypatch):
    from world.builder import build_world
    _, belief, pose, sensor, probs = scene
    world = build_world(1)
    sources = {source for row in world.tile_image for source in row}
    monkeypatch.setattr(perception, "_predictions", lambda: {source: probs[0] for source in sources})
    result = perception.observe(world, belief, pose, sensor, mode="model")
    assert result.ok
    np.testing.assert_array_equal(belief.known, perception.sensor_cone(pose, sensor))
    np.testing.assert_allclose(belief.probs.sum(axis=-1), 1, atol=1e-6)


@pytest.fixture(scope="module")
def real_demo_predictions():
    from world.paths import WEIGHTS
    if not WEIGHTS.exists():
        pytest.skip("real A01 weights are not available")
    from ml import infer
    from world.builder import build_world
    from unittest.mock import patch
    # Optional real-data checks are cache-only: never download on pytest.
    with patch("datasets.config.HF_DATASETS_OFFLINE", True), patch("huggingface_hub.constants.HF_HUB_OFFLINE", True):
        try:
            infer._dataset()
        except (ConnectionError, FileNotFoundError):
            pytest.skip("AI4Mars test split is not cached locally")
    world, pose, sensor = build_world(1), Pose(190, 10, 0), SensorConfig()
    cone = perception.sensor_cone(pose, sensor)
    rows, cols = np.nonzero(cone)
    visible = np.unique((rows // TILE) * 10 + cols // TILE)
    sources = sorted({world.tile_image[int(index // 10)][int(index % 10)] for index in visible})
    clear = {source: infer.predict_tile(source) for source in sources}
    return world, pose, sensor, cone, clear


def test_real_cpu_live_prediction_latency(real_demo_predictions):
    from ml import infer
    # Model/data are warmed by the fixed demo fixture; cold start is separate.
    timings = []
    for _ in range(5):
        started = perf_counter()
        probs = infer.predict_tile("m3_0160", blur=.8)
        timings.append((perf_counter() - started) * 1000)
        assert probs.shape == (20, 20, 4) and np.isfinite(probs).all()
        np.testing.assert_allclose(probs.sum(-1), 1, atol=1e-6)
    median, p95 = np.percentile(timings, [50, 95])
    print(f"live CPU tile median={median:.3f} ms p95={p95:.3f} ms")
    assert median < 1000


def test_real_demo_blur_fresh_belief_contract(real_demo_predictions, monkeypatch):
    # A single blurred frame on a fresh belief is NOT guaranteed to raise U.
    # Measured over all 149 usable tiles (ml/blur_study.py, eval/blur_study.json):
    # blur=0.8 raises per-tile cell U on only 62% (T=1.0) / 64% (T=1.52) of
    # tiles, while label accuracy falls 0.97 -> 0.66; the model mostly becomes
    # confidently wrong. This fixed demo start is one where U falls, so only
    # the belief contract is asserted here; the rover-sequence test below
    # carries the blur acceptance criterion.
    world, pose, sensor, cone, clear_predictions = real_demo_predictions
    monkeypatch.setattr(perception, "_predictions", lambda: clear_predictions)
    clear, blurred = Belief.empty(), Belief.empty()
    normal = perception.observe(world, clear, pose, sensor, mode="model")
    degraded = perception.observe(world, blurred, pose, sensor, ObserveFlags(blur=.8), mode="model")
    assert normal.ok and degraded.ok
    np.testing.assert_array_equal(blurred.known, cone)
    np.testing.assert_allclose(blurred.probs.sum(axis=-1), 1, atol=1e-6)
    assert ((blurred.unc >= 0) & (blurred.unc <= 1)).all()
    changed = float((blurred.probs[cone].argmax(-1) != clear.probs[cone].argmax(-1)).mean())
    print(f"fixed seed=1 pose=(190,10,0) mean U clear={clear.unc[cone].mean():.8f} "
          f"blur={blurred.unc[cone].mean():.8f} argmax changed={changed:.3f}")
    assert changed > 0  # blur reaches the model: predictions actually differ


def test_real_rover_sequence_blur_raises_fused_uncertainty(real_demo_predictions):
    # Blur acceptance: the rover drives north on seeds 1-3, 6 clear ticks then
    # 4 ticks with blur=0.8, versus the same 10 poses all clear. Clear and blur
    # frames both come from live CPU inference at the same temperature, so a
    # cache built at another T cannot fake the effect. Measured on seeds 1-20
    # (96 tiles): fused U rises on 81% (T=1.0) / 77% (T=1.52) of blurred tiles;
    # on seeds 1-3 it rises on 13/15 tiles at both temperatures. Require two
    # thirds, which also holds on the 20-seed population at both temperatures.
    from ml.blur_study import rover_study
    result = rover_study(seeds=(1, 2, 3))
    print({key: value for key, value in result.items() if key != "tiles"})
    assert result["n_tiles"] >= 10
    assert result["frac_u_up"] >= 2 / 3
    assert result["mean_u_blur_phase"] > result["mean_u_clear"]


def test_published_cache_matches_live_cpu_predictions(real_demo_predictions):
    from world.paths import PREDS_NPZ
    from ml import infer
    if not PREDS_NPZ.exists():
        pytest.skip("real A06 prediction cache is not available")
    with np.load(PREDS_NPZ, allow_pickle=False) as archive:
        if not np.isclose(float(archive["temperature"]), infer.temperature()):
            pytest.skip("cached and live temperatures differ; rerun A06 after calibration")
    _, _, _, _, clean = real_demo_predictions
    cache = perception._predictions()
    for source, live in clean.items():
        # Cached GPU AMP/fp16 and live CPU fp32 have rounding differences.
        np.testing.assert_allclose(cache[source], live, atol=.01, rtol=.01)
