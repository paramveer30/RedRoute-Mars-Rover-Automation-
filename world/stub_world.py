"""Synthetic placeholder world so Stream B is never blocked.

Stream A replaces this with world/builder.py::build_world(seed) built from the real
Jezero DTM and AI4Mars test_min3 tiles (handoff at hour 6). Same return type.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from backend.app.contracts import (BIG_ROCK, BEDROCK, GRID, N_TILES, RES_M, SAND,
                                   SOIL, World)


def build_stub_world(seed: int = 0) -> World:
    rng = np.random.default_rng(seed)
    # smooth random hills, a few metres of relief over 50 m
    elev = gaussian_filter(rng.normal(size=(GRID, GRID)), sigma=18) * 60
    elev = elev.astype(np.float32)
    gy, gx = np.gradient(elev, RES_M)
    slope = np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)
    rough = gaussian_filter(np.abs(rng.normal(size=(GRID, GRID))), 2)
    rough = ((rough - rough.min()) / (np.ptp(rough) + 1e-9)).astype(np.float32)

    # per-tile dominant class with some speckle
    truth = np.full((GRID, GRID), SOIL, np.uint8)
    tile = GRID // N_TILES
    for tr in range(N_TILES):
        for tc in range(N_TILES):
            cls = rng.choice([SOIL, BEDROCK, SAND], p=[0.5, 0.3, 0.2])
            truth[tr * tile:(tr + 1) * tile, tc * tile:(tc + 1) * tile] = cls
    for _ in range(25):  # scattered big rocks
        r, c = rng.integers(5, GRID - 5, size=2)
        truth[r - 2:r + 2, c - 2:c + 2] = BIG_ROCK

    return World(
        seed=seed,
        elevation=elev,
        slope_deg=slope,
        roughness=rough,
        truth=truth,
        geo_obstacle=np.zeros((GRID, GRID), bool),
        tile_image=[[f"stub_{tr}_{tc}" for tc in range(N_TILES)] for tr in range(N_TILES)],
    )
