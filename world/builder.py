"""Seeded worlds from Jezero terrain and expert-labelled AI4Mars tiles.

Tiles are NEVER rotated: tile_image contains plain m3_#### source ids, and
truth/probability/image grids retain the exact orientation in tiles.npz.
Only the offline ORACLE demo-selection helpers below inspect truth hazards;
this module does not supply a runtime rover/planner policy.
"""
from __future__ import annotations

from collections import deque
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

from backend.app.contracts import (
    BIG_ROCK, GRID, N_TILES, NO_LABEL, RES_M, SAND, TILE, CostConfig, World,
)
from world.paths import DEMO_SEEDS, RUNS_DIR, TILES_NPZ, USABLE_TILES
from world.terrain import load_terrain


@lru_cache(maxsize=1)
def _assets():
    terrain = load_terrain()
    with np.load(TILES_NPZ, allow_pickle=False) as archive:
        ids = archive["ids"].tolist()
        labels = archive["label20"].copy()
    usable = sorted(json.loads(USABLE_TILES.read_text(encoding="utf-8")))
    if not usable or len(set(usable)) != len(usable):
        raise ValueError("usable tile ids must be nonempty and unique")
    lookup = {source_id: i for i, source_id in enumerate(ids)}
    if any(source_id not in lookup for source_id in usable):
        raise ValueError("usable tile list references a missing tile")
    if labels.shape != (len(ids), TILE, TILE) or labels.dtype != np.uint8:
        raise ValueError("tile labels must be (N, TILE, TILE) uint8")
    if not np.isin(labels, [0, 1, 2, 3, NO_LABEL]).all():
        raise ValueError("tile library contains invalid class ids")
    indices = np.array([lookup[source_id] for source_id in usable])
    selected_labels = labels[indices]
    hazards = ((selected_labels == SAND) | (selected_labels == BIG_ROCK)).mean(axis=(1, 2))
    return terrain, usable, selected_labels, hazards


def build_world(seed: int, sand_bias: float = 0.0) -> World:
    """Build a reproducible world with independent, writable array storage.

    Sample without replacement when at least 100 usable tiles are available.
    A positive sand_bias increases the weights of sand/big-rock-rich tiles;
    it never changes source labels, terrain, or the selected tile orientation.
    """
    if not np.isfinite(sand_bias) or sand_bias < 0:
        raise ValueError("sand_bias must be finite and non-negative")
    terrain, ids, labels, hazards = _assets()
    rng = np.random.default_rng(seed)
    probabilities = None
    if sand_bias > 0:
        weights = 1.0 + sand_bias * hazards
        probabilities = weights / weights.sum()
    chosen = rng.choice(len(ids), size=N_TILES * N_TILES,
                        replace=len(ids) < N_TILES * N_TILES, p=probabilities)
    chosen = chosen.reshape(N_TILES, N_TILES)
    truth = labels[chosen].transpose(0, 2, 1, 3).reshape(GRID, GRID).copy()
    return World(
        seed=int(seed),
        elevation=terrain["elevation"].copy(),
        slope_deg=terrain["slope_deg"].copy(),
        roughness=terrain["roughness"].copy(),
        truth=truth,
        geo_obstacle=np.zeros((GRID, GRID), dtype=bool),
        tile_image=[[ids[int(index)] for index in row] for row in chosen],
    )


def render_world(world: World, path) -> None:
    """Render offline ORACLE truth over hillshade with north-up tile references."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    palette = np.array([[200, 161, 101], [138, 143, 152],
                        [242, 209, 107], [229, 72, 77]], dtype=np.float32) / 255
    shaded = LightSource(azdeg=315, altdeg=45).shade(
        world.elevation, cmap=plt.get_cmap("gray"), dx=RES_M, dy=RES_M,
        vert_exag=1, blend_mode="soft")
    rgba = np.zeros((GRID, GRID, 4), np.float32)
    for class_id, color in enumerate(palette):
        rgba[world.truth == class_id] = [*color, 0.62]
    rgba[world.truth == NO_LABEL] = [0.05, 0.05, 0.05, 0.58]
    fig, axis = plt.subplots(figsize=(12, 13))
    fig.subplots_adjust(top=0.87, bottom=0.07, left=0.07, right=0.98)
    extent = (0, GRID * RES_M, GRID * RES_M, 0)
    axis.imshow(shaded, extent=extent, origin="upper")
    axis.imshow(rgba, extent=extent, origin="upper", interpolation="nearest")
    centres = (np.arange(GRID) + 0.5) * RES_M
    axis.contour(centres, centres, world.slope_deg, levels=[20], colors="cyan", linewidths=1)
    start, goal = (190, 10), (10, 120)
    route = _oracle_demo_route(world, start, goal)
    if route is not None:
        points = np.asarray(route)
        axis.plot((points[:, 1] + 0.5) * RES_M, (points[:, 0] + 0.5) * RES_M,
                  color="deepskyblue", linewidth=2.0, zorder=4)
        axis.plot((start[1] + 0.5) * RES_M, (start[0] + 0.5) * RES_M,
                  marker="o", color="lime", markeredgecolor="black", markersize=10, zorder=5)
        axis.plot((goal[1] + 0.5) * RES_M, (goal[0] + 0.5) * RES_M,
                  marker="*", color="magenta", markeredgecolor="black", markersize=15, zorder=5)
    for tr in range(N_TILES):
        for tc in range(N_TILES):
            axis.text((tc + 0.5) * TILE * RES_M, (tr + 0.5) * TILE * RES_M,
                      world.tile_image[tr][tc], fontsize=6.2, ha="center", va="center",
                      color="white", bbox=dict(facecolor="black", alpha=0.5, pad=1, edgecolor="none"))
    ticks = np.arange(N_TILES + 1) * TILE * RES_M
    axis.set_xticks(ticks)
    axis.set_yticks(ticks)
    axis.grid(color="white", alpha=0.75, linewidth=0.7)
    axis.set_xlabel("East (m)")
    axis.set_ylabel("South from north edge (m)")
    fig.suptitle(f"Seed {world.seed} | ORACLE expert truth + Jezero hillshade\n"
                 "North up; unrotated 5 m tiles; cyan contour = 20 degree slope", y=0.985, fontsize=15)
    legend = [Patch(facecolor=color, label=name) for color, name in
              zip(palette, ("soil", "bedrock", "sand", "big rock"))]
    legend.append(Patch(facecolor="#202020", label="unlabelled / masked"))
    if route is not None:
        legend.extend([
            Line2D([], [], color="deepskyblue", label="offline ORACLE route"),
            Line2D([], [], color="lime", marker="o", linestyle="none", label="start (190, 10)"),
            Line2D([], [], color="magenta", marker="*", linestyle="none", label="goal (10, 120)"),
        ])
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.945),
               ncol=4, fontsize=9)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _oracle_demo_route(world: World, start, goal):
    """Offline ORACLE demo selection only: find a conservative cardinal route.

    Sand, big rock and slopes above the configured limit are hazards. Apply
    the default 1.5 m planner clearance to all hazards. Four-connected BFS
    proves reachability without diagonal corner cutting. Unknown truth is
    not declared a hazard; no claim is made about unlabelled terrain safety.
    This deliberately does not call or replace Stream B's runtime planner.
    """
    config = CostConfig()
    raw = ((world.truth == SAND) | (world.truth == BIG_ROCK) |
           (world.slope_deg > config.max_slope_deg))
    radius = round(config.inflate_m / RES_M)
    yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    blocked = ndimage.binary_dilation(raw, structure=xx * xx + yy * yy <= radius * radius)
    if blocked[start] or blocked[goal]:
        return None
    previous = {start: None}
    pending = deque([start])
    while pending:
        current = pending.popleft()
        if current == goal:
            route = []
            while current is not None:
                route.append(current)
                current = previous[current]
            return route[::-1]
        row, col = current
        for other in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            rr, cc = other
            if 0 <= rr < GRID and 0 <= cc < GRID and not blocked[other] and other not in previous:
                previous[other] = current
                pending.append(other)
    return None


def _find_demos(count=3, max_seed=500):
    """Offline ORACLE seed search, restricted to the specified seed interval."""
    start, goal = (190, 10), (10, 120)
    line = np.rint(np.linspace(start, goal, max(abs(np.subtract(start, goal))) + 1)).astype(int)
    demos = []
    for seed in range(max_seed + 1):
        world = build_world(seed)
        sand_cells = line[world.truth[line[:, 0], line[:, 1]] == SAND]
        sand_tiles = {tuple(cell // TILE) for cell in sand_cells}
        if len(sand_tiles) < 2:
            continue
        route = _oracle_demo_route(world, start, goal)
        if route is None:
            continue
        demos.append((world, {"seed": seed, "start": start, "goal": goal,
                              "sand_tiles_on_straight_line": len(sand_tiles),
                              "sand_cells_on_straight_line": len(sand_cells),
                              "oracle_cardinal_route_m": (len(route) - 1) * RES_M}))
        if len(demos) == count:
            return demos
    raise RuntimeError(f"Only {len(demos)} suitable demo seeds found through {max_seed}")


def main():
    """Generate committed demo notes and regenerable world plots."""
    demos = _find_demos()
    lines = ["# Demo seeds: real Jezero terrain and AI4Mars tiles", "",
             "Tiles are sampled without replacement and are not rotated. All seeds use",
             "the same real elevation, slope and roughness grids. Seed changes only the",
             "expert-labelled surface tile layout. Geometric obstacles start empty.", "",
             "## Reproducible candidates", "",
             "Start `(190, 10)` and goal `(10, 120)` are `(row, col)` cells.", "",
             "| Seed | Sand tiles crossed by straight line | Sand cells on line | ORACLE cardinal route (m) |",
             "| --- | --- | --- | --- |"]
    for world, stats in demos:
        path = RUNS_DIR / "worlds" / f"seed_{world.seed}.png"
        render_world(world, path)
        print(json.dumps({**stats, "png": str(path)}), flush=True)
        lines.append(f'| {world.seed} | {stats["sand_tiles_on_straight_line"]} | '
                     f'{stats["sand_cells_on_straight_line"]} | {stats["oracle_cardinal_route_m"]:.2f} |')
    lines.extend(["", "Each candidate offers sand on the direct line and a verified route around",
                  "the labelled hazards; the fixed eastern scarp also appears in every plot.", "",
                  "## What the reachability measurement means", "",
                  "This is an offline **ORACLE demo-selection check**, not rover performance.",
                  "Search covers seeds 0 through 500 and selects the first three qualifying",
                  "seeds. It blocks truth sand, truth big rock, and slope above 20 degrees,",
                  "then inflates all hazards by the default 1.5 m (6-cell) clearance.",
                  "A four-connected breadth-first search verifies free start/goal and a route",
                  "without corner cutting. Reported lengths are cardinal BFS path lengths,",
                  "not the future eight-connected planner's optimal route lengths.",
                  "Truth 255 stays unlabelled; it is not classified as a hazard or proven safe.", "",
                  "## Preview and Stream B handoff", "",
                  "Run `python -m world.builder` with `RR_ARTIFACTS` set to regenerate plots",
                  "under `RUNS_DIR / worlds`: " + ", ".join(f"`seed_{w.seed}.png`" for w, _ in demos) + ".", "",
                  "Drop-in API import replacement, preserving existing call sites:", "",
                  "```python", "from world.builder import build_world as build_stub_world", "```", "",
                  "Alternatively import `build_world` directly and rename the two current",
                  "`build_stub_world(...)` call sites in `backend/app/api.py`.",
                  "Stream B owns that edit. Pick the final demo seed together.", "",
                  "Honesty: unrelated Curiosity NAVCAM images are stitched over Jezero",
                  "terrain; tile edges are discontinuous, and their top-down geometry is",
                  "an uncalibrated flat-ground approximation.", ""])
    DEMO_SEEDS.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
