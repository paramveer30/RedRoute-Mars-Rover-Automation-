# Demo seeds: real Jezero terrain and AI4Mars tiles

Tiles are sampled without replacement and are not rotated. All seeds use
the same real elevation, slope and roughness grids. Seed changes only the
expert-labelled surface tile layout. Geometric obstacles start empty.

## Reproducible candidates

Start `(190, 10)` and goal `(10, 120)` are `(row, col)` cells.

| Seed | Sand tiles crossed by straight line | Sand cells on line | ORACLE cardinal route (m) |
| --- | --- | --- | --- |
| 1 | 2 | 13 | 72.50 |
| 10 | 2 | 21 | 76.00 |
| 13 | 4 | 37 | 79.00 |

Each candidate offers sand on the direct line and a verified route around
the labelled hazards; the fixed eastern scarp also appears in every plot.

## What the reachability measurement means

This is an offline **ORACLE demo-selection check**, not rover performance.
Search covers seeds 0 through 500 and selects the first three qualifying
seeds. It blocks truth sand, truth big rock, and slope above 20 degrees,
then inflates all hazards by the default 1.5 m (6-cell) clearance.
A four-connected breadth-first search verifies free start/goal and a route
without corner cutting. Reported lengths are cardinal BFS path lengths,
not the future eight-connected planner's optimal route lengths.
Truth 255 stays unlabelled; it is not classified as a hazard or proven safe.

## Preview and Stream B handoff

Run `python -m world.builder` with `RR_ARTIFACTS` set to regenerate plots
under `RUNS_DIR / worlds`: `seed_1.png`, `seed_10.png`, `seed_13.png`.

Drop-in API import replacement, preserving existing call sites:

```python
from world.builder import build_world as build_stub_world
```

Alternatively import `build_world` directly and rename the two current
`build_stub_world(...)` call sites in `backend/app/api.py`.
Stream B owns that edit. Pick the final demo seed together.

Honesty: unrelated Curiosity NAVCAM images are stitched over Jezero
terrain; tile edges are discontinuous, and their top-down geometry is
an uncalibrated flat-ground approximation.
