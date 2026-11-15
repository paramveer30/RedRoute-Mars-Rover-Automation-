"""Shared contract between Stream A (perception/world) and Stream B (autonomy/UI).

Change this file ONLY with both people agreeing. Everything else codes against it.

Coordinates: grid index (row, col) == (y, x). Row 0 is north. Every array is (GRID, GRID).
Units: distances in metres unless a name ends in _cells. Angles in degrees.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np

# ---- Grid geometry ----------------------------------------------------------
GRID = 200          # cells per side
RES_M = 0.25        # metres per cell -> world is 50 m x 50 m
TILE = 20           # cells per tile side -> 10 x 10 tiles of 5 m
N_TILES = GRID // TILE

# ---- Terrain classes (AI4Mars encoding) ------------------------------------
SOIL, BEDROCK, SAND, BIG_ROCK = 0, 1, 2, 3
NO_LABEL = 255
N_CLASSES = 4
CLASS_NAMES = ("soil", "bedrock", "sand", "big_rock")

Cell = tuple[int, int]  # (row, col)


def tile_id(tr: int, tc: int) -> str:
    """Tile id string used everywhere (WS messages, logs): t_<row>_<col>."""
    return f"t_{tr}_{tc}"


def cell_to_tile(cell: Cell) -> tuple[int, int]:
    return cell[0] // TILE, cell[1] // TILE


# ---- World: built by Stream A, read by both --------------------------------
@dataclass
class World:
    seed: int
    elevation: np.ndarray       # (200,200) float32, metres
    slope_deg: np.ndarray       # (200,200) float32
    roughness: np.ndarray       # (200,200) float32, 0..1
    truth: np.ndarray           # (200,200) uint8, 0..3 or 255. NEVER read by the rover/planner
    geo_obstacle: np.ndarray    # (200,200) bool, injected rocks (fault panel). Revealed by sensor range
    tile_image: list[list[str]]  # (10,10) AI4Mars image ids backing each tile


# ---- Belief: owned by Stream A, mutated by observe() -----------------------
@dataclass
class Belief:
    probs: np.ndarray           # (200,200,4) float32, class probabilities
    unc: np.ndarray             # (200,200) float32, normalized entropy 0..1
    known: np.ndarray           # (200,200) bool
    geo_seen: np.ndarray        # (200,200) bool, geometric obstacles the sensor has detected
    version: int = 0            # bumps on every update

    @classmethod
    def empty(cls) -> "Belief":
        return cls(
            probs=np.full((GRID, GRID, N_CLASSES), 1.0 / N_CLASSES, np.float32),
            unc=np.full((GRID, GRID), 0.5, np.float32),
            known=np.zeros((GRID, GRID), bool),
            geo_seen=np.zeros((GRID, GRID), bool),
        )


@dataclass
class Pose:
    row: float
    col: float
    heading_deg: float          # 0 = north (row decreasing), clockwise positive

    @property
    def cell(self) -> Cell:
        return int(round(self.row)), int(round(self.col))


PerceptionMode = Literal["oracle", "model", "geometry_only", "offline"]


@dataclass
class CostConfig:
    w_s: float = 4.0            # terrain risk weight
    w_r: float = 2.0            # roughness weight
    w_u: float = 3.0            # uncertainty weight (0 = "RedRoute without U" ablation)
    risk: tuple[float, float, float] = (0.2, 0.1, 1.0)  # soil, bedrock, sand
    unknown_u: float = 0.5
    max_slope_deg: float = 20.0
    p_big_rock_block: float = 0.5
    inflate_m: float = 1.5
    min_blob_cells: int = 4     # drop blocked blobs smaller than this before inflating (noise)
    use_semantics: bool = True  # False = geometry-only baseline


@dataclass
class SensorConfig:
    range_m: float = 8.0
    fov_deg: float = 45.0


@dataclass
class ObserveFlags:
    """Fault hooks Stream B sets and Stream A's observe() honours."""
    drop_frame: bool = False        # no observation this tick
    corrupt_frame: bool = False     # checksum fails -> reject, return []
    blur: float = 0.0               # 0..1, degrade image before inference (raises U)
    model_offline: bool = False     # geometry only


@dataclass
class ObserveResult:
    tiles_seen: list[str] = field(default_factory=list)
    ok: bool = True
    reason: str = ""                # e.g. "corrupt_frame", "dropped"
    frame_image_id: str | None = None  # AI4Mars id of the frame "taken" this tick (camera panel)
    latency_ms: float = 0.0


# ---- Function signatures both streams code against -------------------------
class ObserveFn(Protocol):
    def __call__(self, world: World, belief: Belief, pose: Pose,
                 sensor: SensorConfig, flags: ObserveFlags,
                 mode: PerceptionMode) -> ObserveResult: ...


class CostGridFn(Protocol):
    def __call__(self, world: World, belief: Belief,
                 cfg: CostConfig) -> tuple[np.ndarray, np.ndarray]: ...  # (cost float32, blocked bool)


class PlanFn(Protocol):
    def __call__(self, cost: np.ndarray, blocked: np.ndarray,
                 start: Cell, goal: Cell) -> list[Cell] | None: ...
