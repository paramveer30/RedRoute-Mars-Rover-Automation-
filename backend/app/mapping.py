"""Belief-only terrain cost and clearance masks for PRD section 7.3.

C = max(1, 1 + w_s*S + w_r*R + w_u*U). With semantics on, known cells
whose belief P(big rock) > p_big_rock_block or P(sand) >= P_SAND_BLOCK are
hard obstacles; sand below the threshold keeps only its soft risk term.
Blocked sand is inflated by SAND_INFLATE_M (the rover body radius) instead
of cfg.inflate_m: with 1.5 m around every sand blob, a perfect-knowledge map
joined start and goal in only 4 of the first 20 eval scenarios (19 of 20 at
0.5 m), because AI4Mars terrain is about a quarter sand.
Semantic noise filtering (min_blob_cells) applies to each semantic class
separately; terrain slopes and sensed geometry remain hard obstacles even
when they occupy a single cell.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy import ndimage

from .contracts import BIG_ROCK, RES_M, SAND, Belief, Cell, CostConfig, World

# Belief P(sand) at or above which a known cell is blocked. Reuses the big-rock
# threshold so CostConfig (shared contract) needs no new field.
P_SAND_BLOCK = CostConfig().p_big_rock_block
# Clearance around blocked sand: the rover footprint radius (0.5 m = 2 cells),
# so an unblocked cell's footprint never overlaps predicted sand. Capped by
# cfg.inflate_m so inflate_m=0 still disables all inflation.
SAND_INFLATE_M = 0.5


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


def _drop_small_blobs(mask: np.ndarray, min_cells: int) -> np.ndarray:
    if min_cells <= 1 or not mask.any():
        return mask
    labels, _ = ndimage.label(mask)
    keep = np.bincount(labels.ravel()) >= min_cells
    keep[0] = False
    return keep[labels]


def _block_sources(world: World, belief: Belief, cfg: CostConfig
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Raw (pre-inflation) blocks: slope, big rock, sand, sensed geometry."""
    rock = np.zeros_like(belief.known)
    sand = np.zeros_like(belief.known)
    if cfg.use_semantics:
        rock = _drop_small_blobs(belief.known & (belief.probs[..., BIG_ROCK] > cfg.p_big_rock_block),
                                 cfg.min_blob_cells)
        sand = _drop_small_blobs(belief.known & (belief.probs[..., SAND] >= P_SAND_BLOCK),
                                 cfg.min_blob_cells)
    return world.slope_deg > cfg.max_slope_deg, rock, sand, belief.geo_seen


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

    slope_block, rock_block, sand_block, geo_block = _block_sources(world, belief, cfg)
    blocked = _inflate(slope_block | rock_block | geo_block, cfg.inflate_m)
    blocked |= _inflate(sand_block, min(cfg.inflate_m, SAND_INFLATE_M))
    return cost, blocked


def _inflate(raw: np.ndarray, radius_m: float) -> np.ndarray:
    disk = _inflation_disk(int(round(radius_m / RES_M)))
    return ndimage.binary_dilation(raw, structure=disk) if raw.any() and disk.size > 1 else raw


_REASONS = ("slope", "big_rock", "predicted sand", "geo_obstacle")


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
    reasons = [name for name, mask in zip(_REASONS, source_masks) if mask[cell]]
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
        "blocked_label": f"blocked: {', '.join(reasons)}" if reasons else None,
    }
