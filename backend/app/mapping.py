"""Belief-only terrain cost and clearance masks for PRD section 7.3.

C = max(1, 1 + w_s*S + w_r*R + w_u*U). Semantic noise filtering
applies only to big-rock predictions; terrain slopes and sensed geometry
remain hard obstacles even when they occupy a single cell.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy import ndimage

from .contracts import BIG_ROCK, RES_M, Belief, Cell, CostConfig, World


@lru_cache(maxsize=32)
def _inflation_disk(radius: int) -> np.ndarray:
    if radius < 0:
        raise ValueError("inflate_m must be non-negative")
    yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    disk = (xx * xx + yy * yy) <= radius * radius
    disk.flags.writeable = False
    return disk


# Build the common 1.5 m clearance footprint once, rather than every tick.
_inflation_disk(round(CostConfig().inflate_m / RES_M))


def _cost_terms(world: World, belief: Belief, cfg: CostConfig
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if cfg.use_semantics:
        risk = np.asarray(cfg.risk, np.float32)
        semantic = (belief.probs[..., :3] * risk).sum(axis=-1)
        semantic = np.where(belief.known, semantic, 0.0)
        uncertainty = np.where(belief.known, belief.unc, cfg.unknown_u)
    else:
        semantic = np.zeros_like(world.roughness)
        uncertainty = np.zeros_like(world.roughness)
    return semantic, world.roughness, uncertainty


def _block_sources(world: World, belief: Belief, cfg: CostConfig
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    semantic = np.zeros_like(belief.known)
    if cfg.use_semantics:
        semantic = belief.known & (belief.probs[..., BIG_ROCK] > cfg.p_big_rock_block)
        if cfg.min_blob_cells > 1 and semantic.any():
            labels, _ = ndimage.label(semantic)
            sizes = np.bincount(labels.ravel())
            keep = sizes >= cfg.min_blob_cells
            keep[0] = False
            semantic = keep[labels]
    return world.slope_deg > cfg.max_slope_deg, semantic, belief.geo_seen


def cost_grid(world: World, belief: Belief, cfg: CostConfig | None = None
              ) -> tuple[np.ndarray, np.ndarray]:
    """Return float32 costs >= 1 and a bool mask inflated for body clearance.

    Geometry is blocked only when observed in ``belief.geo_seen``. This path
    never accesses truth labels or unrevealed ``world.geo_obstacle``.
    """
    cfg = cfg or CostConfig()
    semantic, roughness, uncertainty = _cost_terms(world, belief, cfg)
    cost = (1.0 + cfg.w_s * semantic + cfg.w_r * roughness + cfg.w_u * uncertainty).astype(np.float32)
    np.maximum(cost, 1.0, out=cost)

    slope_block, semantic_block, geo_block = _block_sources(world, belief, cfg)
    raw = slope_block | semantic_block | geo_block
    disk = _inflation_disk(int(round(cfg.inflate_m / RES_M)))
    blocked = ndimage.binary_dilation(raw, structure=disk) if raw.any() and disk.size > 1 else raw
    return cost, blocked


def explain_cell(world: World, belief: Belief, cfg: CostConfig, cell: Cell) -> dict:
    """Explain the exact cost terms and final blocking decision for one cell.

    S, R and U are unweighted; slope_deg is in degrees. ``blocked_reason``
    lists direct causes, or ``inflation`` for clearance around a neighbour.
    All values are plain Python types so the result is JSON serializable.
    """
    row, col = cell
    if not (0 <= row < belief.known.shape[0] and 0 <= col < belief.known.shape[1]):
        raise IndexError(f"cell {cell} is outside the belief grid")
    semantic, roughness, uncertainty = _cost_terms(world, belief, cfg)
    costs, blocked = cost_grid(world, belief, cfg)
    source_masks = _block_sources(world, belief, cfg)
    reasons = [name for name, mask in zip(("slope", "big_rock", "geo_obstacle"), source_masks) if mask[cell]]
    if blocked[cell] and not reasons:
        reasons.append("inflation")
    return {
        "S": float(semantic[cell]),
        "R": float(roughness[cell]),
        "U": float(uncertainty[cell]),
        "slope_deg": float(world.slope_deg[cell]),
        "cost": float(costs[cell]),
        "known": bool(belief.known[cell]),
        "blocked": bool(blocked[cell]),
        "blocked_reason": ", ".join(reasons) if reasons else None,
    }
