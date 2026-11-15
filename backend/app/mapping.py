"""Cost map (Stream A owns). Reference implementation of PRD section 7.3 so B can plan now.

C = 1 + w_s*S + w_r*R + w_u*U
S = sum_k p_k * risk_k     (soil 0.2, bedrock 0.1, sand 1.0; big rock handled as hard block)
Hard blocks: slope > max, P(big rock) > 0.5 (known cells only), any detected geo obstacle.
Blocked mask is cleaned of tiny blobs, then inflated by inflate_m for body clearance.

Stream A TODO: tune, unit-test, add geometry-only path for model offline.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .contracts import BIG_ROCK, RES_M, Belief, CostConfig, World


def cost_grid(world: World, belief: Belief, cfg: CostConfig | None = None
              ) -> tuple[np.ndarray, np.ndarray]:
    cfg = cfg or CostConfig()
    known = belief.known

    if cfg.use_semantics:
        risk = np.asarray(cfg.risk, np.float32)
        S = (belief.probs[..., :3] * risk).sum(-1)
        S = np.where(known, S, 0.0)
        U = np.where(known, belief.unc, cfg.unknown_u)
    else:
        S = np.zeros_like(world.roughness)
        U = np.zeros_like(world.roughness)

    cost = (1.0 + cfg.w_s * S + cfg.w_r * world.roughness + cfg.w_u * U).astype(np.float32)

    sem_block = np.zeros_like(known)
    if cfg.use_semantics:
        sem_block = known & (belief.probs[..., BIG_ROCK] > cfg.p_big_rock_block)
        if cfg.min_blob_cells > 1:
            lab, n = ndimage.label(sem_block)
            if n:
                sizes = ndimage.sum(sem_block, lab, range(1, n + 1))
                sem_block = np.isin(lab, np.nonzero(sizes >= cfg.min_blob_cells)[0] + 1)

    raw = (world.slope_deg > cfg.max_slope_deg) | sem_block | belief.geo_seen
    r = int(round(cfg.inflate_m / RES_M))
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    disk = (xx * xx + yy * yy) <= r * r
    blocked = ndimage.binary_dilation(raw, structure=disk) if raw.any() else raw
    return cost, blocked
