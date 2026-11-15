"""Cone-limited oracle/model perception with confidence-weighted semantic fusion.

Only the explicitly labelled oracle branch reads World.truth. Accumulated
model weights live in an identity-keyed, weak-reference sidecar because the
shared Belief contract has no weight field and its dataclass is unhashable.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import time
import weakref

import numpy as np

from world.paths import PREDS_NPZ
from .contracts import (GRID, N_CLASSES, N_TILES, NO_LABEL, RES_M, TILE, Belief,
                        ObserveFlags, ObserveResult, PerceptionMode, Pose,
                        SensorConfig, World, cell_to_tile, tile_id)

_rr, _cc = np.mgrid[0:GRID, 0:GRID]
# id -> (weak belief reference, weak probability-buffer reference, weights).
# Never retain a belief strongly: finished missions must release their state.
_fusion_state: dict[int, tuple[weakref.ReferenceType, weakref.ReferenceType, np.ndarray]] = {}


def _weights_for(belief: Belief) -> np.ndarray:
    key = id(belief)
    entry = _fusion_state.get(key)
    if entry is None or entry[0]() is not belief or entry[1]() is not belief.probs:
        def discard(reference, key=key):
            current = _fusion_state.get(key)
            if current is not None and current[0] is reference:
                _fusion_state.pop(key, None)
        reference = weakref.ref(belief, discard)
        weights = np.zeros(belief.known.shape, np.float32)
        _fusion_state[key] = (reference, weakref.ref(belief.probs), weights)
    else:
        weights = entry[2]
    # In-place mission resets commonly clear known while retaining buffers.
    weights[~belief.known] = 0.0
    return weights


def normalized_entropy(probs: np.ndarray) -> np.ndarray:
    """Shannon entropy normalized to [0, 1] for four-class probabilities."""
    p = np.asarray(probs, np.float32)
    return np.clip(-np.sum(p * np.log(np.maximum(p, 1e-12)), axis=-1) / np.log(N_CLASSES),
                   0.0, 1.0).astype(np.float32)


def _probabilities(probs: np.ndarray) -> np.ndarray:
    p = np.asarray(probs, np.float32)
    if p.shape != (TILE, TILE, N_CLASSES):
        raise ValueError(f"tile probabilities must have shape {(TILE, TILE, N_CLASSES)}")
    if not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("tile probabilities must be finite and nonnegative")
    total = p.sum(axis=-1, keepdims=True)
    if (total <= 0).any():
        raise ValueError("tile probabilities must have positive mass")
    return p / total


@lru_cache(maxsize=1)
def _read_predictions(path: str, modified: int, size: int) -> dict[str, np.ndarray]:
    # File identity participates in the key so calibrated/rebuilt caches reload.
    with np.load(path, allow_pickle=False) as data:
        ids = data["ids"]
        probs = data["probs"]
        if ids.ndim != 1 or probs.shape != (len(ids), TILE, TILE, N_CLASSES):
            raise ValueError("prediction cache ids/probs shapes do not match")
        if len(set(ids.tolist())) != len(ids):
            raise ValueError("prediction cache contains duplicate source ids")
        if "temperature" in data:
            temperature = float(data["temperature"])
            if not np.isfinite(temperature) or temperature <= 0:
                raise ValueError("prediction cache has invalid temperature")
        return {str(source): _probabilities(p) for source, p in zip(ids, probs)}


def _predictions() -> dict[str, np.ndarray]:
    path = Path(PREDS_NPZ)
    stat = path.stat()
    return _read_predictions(str(path), stat.st_mtime_ns, stat.st_size)


def _source_ref(reference: str) -> tuple[str, int]:
    # A04 currently uses no rotations. Accept explicit rotations for callers
    # with saved tile layouts, applying the identical spatial rotation to p.
    source, marker, angle = reference.partition("@r")
    if not marker:
        return source, 0
    if angle not in ("0", "90", "180", "270"):
        raise ValueError(f"invalid tile rotation: {reference}")
    return source, int(angle) // 90


def _frame_source(world: World, pose: Pose) -> str:
    angle = np.radians(pose.heading_deg)
    ahead = (int(np.clip(round(pose.row - 4.0 / RES_M * np.cos(angle)), 0, GRID - 1)),
             int(np.clip(round(pose.col + 4.0 / RES_M * np.sin(angle)), 0, GRID - 1)))
    tr, tc = cell_to_tile(ahead)
    reference = world.tile_image[tr][tc]
    if not reference:
        own = (int(np.clip(round(pose.row), 0, GRID - 1)),
               int(np.clip(round(pose.col), 0, GRID - 1)))
        tr, tc = cell_to_tile(own)
        reference = world.tile_image[tr][tc]
    return _source_ref(reference)[0]


def sensor_cone(pose: Pose, sensor: SensorConfig) -> np.ndarray:
    """Bool mask of cells inside the range/FOV cone; north is decreasing row."""
    dy = (_rr - pose.row) * RES_M
    dx = (_cc - pose.col) * RES_M
    dist = np.hypot(dx, dy)
    bearing = np.degrees(np.arctan2(dx, -dy)) % 360
    diff = np.abs((bearing - pose.heading_deg + 180) % 360 - 180)
    return (dist <= sensor.range_m) & ((diff <= sensor.fov_deg / 2) | (dist <= 1.0))


def _predict_live(source: str, blur: float) -> np.ndarray:
    from ml.infer import predict_tile
    return predict_tile(source, blur=blur)


def observe(world: World, belief: Belief, pose: Pose,
            sensor: SensorConfig | None = None,
            flags: ObserveFlags | None = None,
            mode: PerceptionMode = "oracle") -> ObserveResult:
    sensor = sensor or SensorConfig()
    flags = flags or ObserveFlags()
    t0 = time.perf_counter()

    def result(**kwargs):
        return ObserveResult(latency_ms=(time.perf_counter() - t0) * 1000, **kwargs)

    if flags.drop_frame:
        return result(ok=False, reason="dropped")
    if flags.corrupt_frame:
        return result(ok=False, reason="corrupt_frame")

    cone = sensor_cone(pose, sensor)
    revealed = cone & world.geo_obstacle
    geometry_changed = bool((revealed & ~belief.geo_seen).any())
    belief.geo_seen |= revealed
    rows, cols = np.nonzero(cone)
    visible = np.unique((rows // TILE) * N_TILES + cols // TILE)
    tiles = [tile_id(int(index // N_TILES), int(index % N_TILES)) for index in visible]
    frame = _frame_source(world, pose)

    if flags.model_offline or mode in ("geometry_only", "offline"):
        # known is semantic evidence. Geometry sensing must not mark stale or
        # uniform probabilities as trusted observations on a later mode switch.
        pass
    elif mode == "oracle":
        labels = world.truth[cone]
        valid = labels != NO_LABEL
        onehot = np.full((labels.size, N_CLASSES), 1.0 / N_CLASSES, np.float32)
        onehot[valid] = 0.0
        onehot[np.flatnonzero(valid), labels[valid]] = 1.0
        belief.probs[cone] = onehot
        belief.unc[cone] = np.where(valid, 0.0, 1.0)
        belief.known[cone] = True
        _weights_for(belief)[cone] = 0.0
    elif mode == "model":
        try:
            cache = None if flags.blur > 0 else _predictions()
            # Resolve and validate every source before mutating semantic state.
            # A missing/broken tile cannot leave a half-fused observation.
            resolved = {}
            updates = []
            for index in visible:
                tr, tc = int(index // N_TILES), int(index % N_TILES)
                source, rotation = _source_ref(world.tile_image[tr][tc])
                if source not in resolved:
                    resolved[source] = _probabilities(_predict_live(source, flags.blur)
                                                       if cache is None else cache[source])
                p_new = np.rot90(resolved[source], rotation, axes=(0, 1))
                region = np.s_[tr*TILE:(tr+1)*TILE, tc*TILE:(tc+1)*TILE]
                mask = cone[region]
                updates.append((region, mask, p_new[mask]))
            weights = _weights_for(belief)
            for region, mask, p_new in updates:
                new_weight = np.maximum(1.0 - normalized_entropy(p_new), 0.05)
                old_weight = weights[region][mask]
                total = old_weight + new_weight
                fused = ((old_weight[:, None] * belief.probs[region][mask] +
                          new_weight[:, None] * p_new) / total[:, None])
                fused /= fused.sum(axis=-1, keepdims=True)
                belief.probs[region][mask] = fused
                belief.unc[region][mask] = normalized_entropy(fused)
                belief.known[region][mask] = True
                weights[region][mask] = total
        except Exception as exc:
            if geometry_changed:
                belief.version += 1
            return result(tiles_seen=tiles, ok=False, reason=f"model_error: {exc}",
                          frame_image_id=frame)
    else:
        raise ValueError(f"unsupported perception mode: {mode}")

    belief.version += 1
    return result(tiles_seen=tiles, ok=True, frame_image_id=frame)
