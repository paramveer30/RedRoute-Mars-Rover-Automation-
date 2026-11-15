"""Perception (Stream A owns). observe() updates the Belief in place.

Current state: ORACLE stub only. Reveals truth labels inside the sensor cone as one-hot
probabilities with unc = 0. The UI must show an "ORACLE MODE" banner while mode == "oracle".

Stream A TODO (hours 9-12):
  - mode "model": warp cached/live U-Net probabilities for each tile in the cone into cells
  - confidence-weighted fusion across repeat observations (weight = 1 - U)
  - entropy uncertainty  U = -sum p ln p / ln 4
  - honour flags.blur by running the model LIVE on the degraded image (cache can't do this)
  - flags.corrupt_frame -> ok=False, reason="corrupt_frame"; flags.drop_frame -> ok=False
"""
from __future__ import annotations

import time

import numpy as np

from .contracts import (GRID, N_CLASSES, NO_LABEL, RES_M, Belief, ObserveFlags,
                        ObserveResult, PerceptionMode, Pose, SensorConfig, World,
                        cell_to_tile, tile_id)

_rr, _cc = np.mgrid[0:GRID, 0:GRID]


def sensor_cone(pose: Pose, sensor: SensorConfig) -> np.ndarray:
    """Bool mask of cells inside the range/FOV cone."""
    dy = (_rr - pose.row) * RES_M
    dx = (_cc - pose.col) * RES_M
    dist = np.hypot(dx, dy)
    # heading 0 = north (-row), clockwise positive
    bearing = np.degrees(np.arctan2(dx, -dy)) % 360
    diff = np.abs((bearing - pose.heading_deg + 180) % 360 - 180)
    return (dist <= sensor.range_m) & ((diff <= sensor.fov_deg / 2) | (dist <= 1.0))


def observe(world: World, belief: Belief, pose: Pose,
            sensor: SensorConfig | None = None,
            flags: ObserveFlags | None = None,
            mode: PerceptionMode = "oracle") -> ObserveResult:
    sensor = sensor or SensorConfig()
    flags = flags or ObserveFlags()
    t0 = time.perf_counter()

    if flags.drop_frame:
        return ObserveResult(ok=False, reason="dropped")
    if flags.corrupt_frame:
        return ObserveResult(ok=False, reason="corrupt_frame")

    cone = sensor_cone(pose, sensor)
    # geometric channel works in every mode (stands in for stereo/hazcam)
    belief.geo_seen |= cone & world.geo_obstacle

    if mode == "oracle":
        lab = world.truth[cone]
        valid = lab != NO_LABEL
        onehot = np.zeros((lab.size, N_CLASSES), np.float32)
        onehot[np.arange(lab.size)[valid], lab[valid]] = 1.0
        onehot[~valid] = 1.0 / N_CLASSES
        belief.probs[cone] = onehot
        belief.unc[cone] = np.where(valid, 0.0, 1.0)
        belief.known[cone] = True
    elif mode in ("geometry_only", "offline"):
        belief.known[cone] = True  # geometry known, semantics not
    else:
        raise NotImplementedError("model mode: Stream A hours 9-12")

    belief.version += 1
    rows, cols = np.nonzero(cone)
    tiles = sorted({tile_id(*cell_to_tile((r, c))) for r, c in zip(rows[::7], cols[::7])})
    tr, tc = cell_to_tile(pose.cell)
    return ObserveResult(tiles_seen=tiles, ok=True,
                         frame_image_id=world.tile_image[tr][tc],
                         latency_ms=(time.perf_counter() - t0) * 1000)
