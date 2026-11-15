# A02: Terrain grids from the real Jezero DTM

**Wave 1, start immediately.** CPU only, about 1 hour.

## Goal
Turn `terrain/jezero_crop.tif` into the elevation, slope and roughness grids that `World` needs.

## You own
- `world/terrain.py`
- `world/terrain.npz` (committed, about 0.5 MB)
- `backend/tests/test_terrain.py`
- `$RR_ARTIFACTS/runs/terrain/*.png`

## Inputs
- `terrain/jezero_crop.tif`: 58 × 58 at 1 m, **4 m padding** on each side of the 50 m patch, elevation in metres. Location 18.43910°N, 77.40880°E: a drivable plateau with the delta-front scarp along the east edge.
- `world/paths.py`: `DTM_CROP`, `TERRAIN_NPZ`, `RUNS_DIR`
- `backend/app/contracts.py`: `GRID=200`, `RES_M=0.25`

## Spec
`world/terrain.py`:
```python
def build_terrain(path=DTM_CROP) -> dict[str, np.ndarray]   # keys elevation, slope_deg, roughness; each (200,200) float32
def load_terrain() -> dict[str, np.ndarray]                  # reads TERRAIN_NPZ (fast path used by builder)
if __name__ == "__main__": build, save TERRAIN_NPZ, save PNG, print stats
```
1. Read with rasterio. Keep the padding during processing, so gradients and windows at the edges are valid.
2. **elevation:** bilinear upsample from 1 m to 0.25 m (`scipy.ndimage.zoom(order=1)` or `cv2.resize INTER_LINEAR`), then trim the padding (16 cells each side) to exactly (200, 200).
3. **slope_deg:** `np.degrees(np.arctan(np.hypot(*np.gradient(z_upsampled, RES_M))))`, trimmed the same way.
4. **roughness: compute at native 1 m, NOT on the upsampled grid.** A bilinear upsample is locally planar, so its plane-fit residual is about 0. At 1 m, for each pixel fit a plane `z = ax + by + c` to its 5 × 5 neighbourhood (least squares; it's vectorisable, since the design matrix is identical for every window) and take the std of the residual. Upsample to 0.25 m, trim, then normalize: `clip(r / p99(r), 0, 1)`.
5. Save `np.savez_compressed(TERRAIN_NPZ, elevation=..., slope_deg=..., roughness=...)`.
6. PNG: 3 panels (elevation with hillshade, slope with a 20° contour, roughness) to `$RR_ARTIFACTS/runs/terrain/terrain.png`. This becomes a slide.

## Tests (`backend/tests/test_terrain.py`)
- shapes (200, 200), dtype float32, no NaN
- roughness in [0, 1] and **not degenerate** (std > 0.05)
- slope: 2% to 25% of cells above 20° (the scarp exists) and median below 10° (mostly drivable)
- plane-fit sanity: a synthetic tilted plane gives roughness ≈ 0 before normalization

## Acceptance
- [ ] `$PY world/terrain.py` prints stats and writes the npz and PNG
- [ ] The PNG looks like terrain: the scarp is visible in slope and roughness on the east side
- [ ] Tests pass

## Hand back
Stats (elevation range, slope median/p95/share over 20°, roughness std) and the PNG path.
