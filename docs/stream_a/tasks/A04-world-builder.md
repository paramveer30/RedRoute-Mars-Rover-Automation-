# A04: Seeded world builder

**Wave 2.** Needs A02 (`world/terrain.npz`) and A03 (`world/tiles.npz`, `world/usable_tiles.json`) on main. CPU, about 1 to 1.5 hours.
**This is the hour-6 handoff to Stream B**, so tell the human the moment it lands.

## Goal
`build_world(seed) -> World` built from real Jezero elevation and real expert-labelled tiles. It's a drop-in replacement for `world/stub_world.build_stub_world`.

## You own
- `world/builder.py`
- `world/DEMO_SEEDS.md`
- `backend/tests/test_builder.py`
- `$RR_ARTIFACTS/runs/worlds/*.png`

## Inputs
`world.terrain.load_terrain()`, `world/tiles.npz` (`ids`, `label20`), `world/usable_tiles.json`, `backend/app/contracts.py` (`World`, `GRID`, `TILE`, `N_TILES`).

## Spec
```python
def build_world(seed: int, sand_bias: float = 0.0) -> World
def render_world(world, path) -> None    # PNG: hillshade + truth classes + tile grid + tile ids
```
- `rng = np.random.default_rng(seed)`. Choose 10 × 10 tile ids from the usable list (sample with replacement if fewer than 100 usable).
- `truth[tr*20:(tr+1)*20, tc*20:(tc+1)*20] = label20[id]`
- Optionally apply a random 0/90/180/270° rotation per tile for variety. **If you do, record it** (for example `tile_image[tr][tc] = "m3_0042@r90"`), and expose `parse_tile_ref(ref) -> (id, k_rot)` so A07 rotates predictions identically. Simpler alternative: no rotation. **Pick one and document it at the top of builder.py.**
- `sand_bias > 0`: weight sampling toward tiles with more sand and big rock, so demo scenarios are interesting.
- Elevation, slope and roughness come straight from `load_terrain()`. They're identical for every seed; only the surface tiles vary.
- `geo_obstacle` all False. `seed` stored on the World.
- Deterministic: the same seed gives byte-identical arrays.

**Demo seeds:** use a default start of (190, 10) (south-west) and goal of (10, 120) (north, staying off the eastern scarp). Adjust if they're blocked. Search seeds 0 to 500 for 3 where the straight line between start and goal crosses at least 2 sand tiles and the goal is reachable around them. Write the seeds, start and goal, and why each is interesting into `world/DEMO_SEEDS.md`. Render each to PNG.

## Tests
- `build_world(0)` returns a `World` with the right shapes and dtypes; `truth` values in {0,1,2,3,255}
- determinism: two calls with the same seed give equal arrays; different seeds give different tile layouts
- every `tile_image` ref resolves to a valid id

## Acceptance
- [ ] Tests pass; 3 demo seeds documented with PNGs
- [ ] `backend/app/api.py` still runs if `build_stub_world` is swapped for `build_world`. **Don't edit api.py** (B owns it); just check it in a scratch script.

## Hand back
The demo seeds, PNG paths, and the exact one-line swap for Stream B:
`from world.builder import build_world` in place of `build_stub_world`.
