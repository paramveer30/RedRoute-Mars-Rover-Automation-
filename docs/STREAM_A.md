# Stream A: Perception, World, and Evaluation

**You own:** everything that turns real NASA data into the rover's picture of the world, and the numbers that prove it works.

**Your files:** `ml/`, `world/`, `eval/`, `backend/app/perception.py`, `backend/app/mapping.py`, plus the 3D view and vision-panel components in `frontend/src/` (P1).

**Do not edit without Stream B:** `backend/app/contracts.py`. Stream B owns `planner.py`, `rover.py`, `link.py`, `faults.py`, `api.py`.

**Git:** everyone works directly on `main`, no branches. Pull before you start, commit small, push often. Stick to your own files to avoid conflicts.

---

## What already exists (from setup)

| File | State |
|---|---|
| `backend/app/contracts.py` | `World`, `Belief`, `Pose`, `CostConfig`, `SensorConfig`, `ObserveFlags`, `ObserveResult`, grid constants |
| `backend/app/perception.py` | `observe()` with **oracle mode** working (sensor cone, geometric channel, drop and corrupt flags). `mode="model"` raises `NotImplementedError`: that's yours |
| `backend/app/mapping.py` | `cost_grid()` reference implementation of PRD 7.3, including blob filtering before inflation. Tune and test it |
| `world/stub_world.py` | Synthetic world that Stream B uses until your real `build_world()` lands |
| `world/crop_dtm.py` | Crops the Jezero DTM over HTTP (no 1.8 GB download) and prints slope stats |
| `ml/check_dataset.py` | Loads AI4Mars `test_min3` and streams training rows |
| `backend/tests/test_contracts.py` | Shape and cone tests. Keep them green |

Run the tests with `.venv/Scripts/python -m pytest` from the repo root.

---

## Hour-by-hour

### Hour 0 to 1: contract and start training
- [ ] 30 min contract meeting with B. Read `contracts.py` together and change anything now, not later.
- [x] Dataset verified in setup (`python ml/check_dataset.py`). What it showed:
  - Columns: `image`, `label_mask`, `rover_mask`, `range_mask`, `has_masks`, `has_labels`. Every array is **1024 × 1024 uint8**.
  - **Some training rows have `has_labels=False` and `label_mask=None`. Filter them out** or training crashes.
  - **There is no image-id column.** Use the `test_min3` row index as the id (`"m3_0042"`), which is what `World.tile_image` stores.
  - On Windows, the HF stream sometimes throws `WinError 10038` and retries itself. If streaming is flaky, download the split once with `load_dataset(..., split="train[:6000]")`.
- [ ] Write `ml/train.py` and start it **in the background**:
  - `smp.Unet("resnet34", encoder_weights="imagenet", classes=4)`
  - Grayscale to 3 channels, **256 px** first (go to 512 only if time allows)
  - Ignore label 255. Also set to 255 any pixel where `rover_mask` is on or `range_mask` marks beyond 30 m
  - Loss = weighted CE (`ignore_index=255`) + Dice. Starting weights: soil 1, bedrock 1, sand 2, **big rock 10 to 20**
  - About 6000 streamed training rows; hold out about 500 as **validation** (you need these later for temperature scaling, so never tune on `test_min3`)
  - AdamW, lr 3e-4, mixed precision, batch 16 at 256 px fits in 8 GB
  - Save `ml/checkpoints/unet_best.pt` (gitignored) by val mIoU, and log loss and mIoU per epoch
- **Done:** loss goes down in the log.

> **GPU:** this laptop has an RTX 5060 (8 GB, CUDA 12.8 build installed), so you can train locally. Kaggle is the backup.

### Hour 1 to 3: terrain grids from the real DTM
- [x] **DTM crop already done in setup:** `terrain/jezero_crop.tif` (58 × 58 at 1 m including 4 m padding each side) at **18.43910°N, 77.40880°E**. It's a drivable plateau (median slope 4.5°) with the steep delta-front scarp along the east edge (about 11% of the 50 m patch is over 20°, 16 m relief). The landing site was tested and rejected: only 1 m of relief and a 4° maximum slope, so the slope constraint would never fire. Re-crop with `world/crop_dtm.py --lat .. --lon ..` if you want another spot.
- [ ] `world/terrain.py`: load the crop, trim the padding to exactly 50 m.
  - **elevation:** bilinear upsample from 1 m to 0.25 m, giving (200, 200).
  - **slope:** `np.degrees(np.arctan(np.hypot(*np.gradient(z, 0.25))))` on the upsampled grid.
  - **roughness: compute at native 1 m, not 0.25 m.** A bilinear upsample is locally planar, so a 3×3 plane fit on it leaves almost zero residual. Use the plane-fit residual std over 3×3 or 5×5 at 1 m, then upsample and normalize to 0..1 with a fixed percentile clip (p99) so values compare across seeds.
  - Save `world/terrain.npz` and a matplotlib PNG of slope and roughness for a sanity check (and later for the slides).
- **Done:** heatmaps look like real terrain, not noise and not flat.

### Hour 3 to 6: tile warping and world builder
- [ ] `world/warp.py`: one fixed ground-plane homography (inverse perspective mapping).
  - Pick 4 source points on an image's near field (a trapezoid about 2 to 10 m ahead) and map them to a 5 m × 5 m top-down square of 20 × 20 cells (warp at a higher resolution, for example 160 × 160 px, for the camera overlay, then downsample to 20 × 20 for the grid).
  - Tune the trapezoid by eye on 5 to 10 images. Use `range_mask` to stay inside the labelled near field.
  - Warp the label with `INTER_NEAREST` and images or probabilities with `INTER_LINEAR`. **Use the same matrix for everything.**
  - Drop images where the warped label is more than 40% 255. Expect to lose a fair share. Save the list of usable ids.
- [ ] Cache warped data to `terrain/tiles/<id>.npz` (`label20`, `image160`, gitignored, regenerable).
- [ ] `world/builder.py::build_world(seed) -> World`:
  - Seeded RNG picks 100 usable ids into the 10 × 10 grid (sample with replacement if fewer than 100 survive)
  - `truth` = the warped labels stitched together; elevation, slope and roughness from `terrain.npz`; `geo_obstacle` all False
  - Optional: bias tile choice so sand and rock tiles appear near the straight-line route, which makes demo seeds interesting
- [ ] Pick 2 or 3 **demo seeds** where the straight route crosses sand. Write them in `world/DEMO_SEEDS.md`.
- **HANDOFF at hour 6:** tell B "swap `build_stub_world` for `build_world`". The signature and return type are identical.

### Hour 6 to 9: evaluate and precompute
- [ ] `ml/evaluate.py` on **`test_min3`**: mIoU, per-class IoU (report big rock separately), confusion matrix, CPU latency per frame at your input size. Save `eval/seg_metrics.json`.
- [ ] If big-rock IoU is poor: raise its weight and retrain in the background. Either way, report it honestly. The geometric channel is the safety backup.
- [ ] `ml/precompute.py`: softmax probabilities for all 322 images, warped with the same homography, saved to `terrain/preds/<id>.npy` as (20, 20, 4) float16, plus logits for calibration.
- **Done:** metrics table saved, preds cached.

### Hour 9 to 12: `observe()` model mode and `cost_grid()`
- [ ] `perception.py` `mode == "model"`:
  - For each tile in the sensor cone, load the cached `(20,20,4)` probabilities and write them into the cone cells of that tile.
  - **Fusion:** keep a per-cell weight sum. New `p = (w_old*p_old + w_new*p_new) / (w_old + w_new)` with `w_new = 1 - U_new`.
  - `U = -Σ p ln p / ln 4` after fusion.
  - **`flags.blur > 0` must run the model LIVE** on a Gaussian-blurred or noised version of the source image (the cache can't show uncertainty rising). Keep the model loaded once in a module-level singleton on CPU. Report `latency_ms`.
  - `mode == "geometry_only"` or `"offline"`: mark known, no semantics (already stubbed).
  - Set `frame_image_id` so B's camera panel shows the real NAVCAM frame.
- [ ] `mapping.py`: tests for the hard constraints (slope, big rock, geo obstacle), inflation radius (6 cells), blob filter, the `use_semantics=False` path, and `cost >= 1` everywhere (A*'s heuristic depends on it).
- [ ] Watch for this: if noisy big-rock pixels plus a 6-cell inflation block huge areas and the rover keeps SAFE_STOPping, raise `min_blob_cells` or apply a morphological opening first.
- **Done:** unit tests pass; B's planner runs on your real belief.

### Hour 12: INTEGRATION 1, gate G3 (most important milestone)
- [ ] Switch the default mode from `"oracle"` to `"model"`. The ORACLE banner disappears.
- [ ] Both of you watch a demo seed: the **route changes because of model output** (detours around predicted sand).
- Don't move on until this works.

### Hour 12 to 16: evaluation harness
- [ ] `eval/harness.py`: headless missions, no FastAPI. It needs B's `Rover` class to step without the server. Agree on `Rover(world, mode, cfg).run_to_completion(start, goal, max_steps)` with B at hour 12.
- [ ] Planners to compare:

  | Name | Setting |
  |---|---|
  | Geometry-only | `CostConfig(use_semantics=False)` |
  | RedRoute without U | `CostConfig(w_u=0)` |
  | RedRoute | default |
  | Oracle | `mode="oracle"` |

- [ ] Metrics per mission: success (within 1 m, no hazard entry), hazard entries (rover footprint on truth sand, big rock, or slope over the limit), route length ratio, replanning latency, steps.
- [ ] `multiprocessing.Pool` with an **`if __name__ == "__main__":` guard** (Windows uses spawn). Load the model once per worker in the pool `initializer`, not once per mission. Batch frames where you can.
- [ ] Record **throughput**: missions/min, frames/s.
- [ ] Save `eval/results.json` and a markdown table. B exposes it at `GET /api/eval/results`.
- **Done:** a real results table. Fill the slide only with measured numbers.

### Hour 16: P0 freeze
Every P0 feature works end to end. Nothing new until then.

### Hour 16 to 19: calibration and decision card (P1)
- [ ] Temperature scaling: fit a single T on your **validation** split (LBFGS on NLL). Report ECE before and after (15 bins). Apply T inside `observe()`.
- [ ] Route decision card data: `{chosen_len_m, shortest_len_m, hazard_avoided: "sand", mean_conf_on_avoided, extra_m}`. Agree on the field names with B, who draws it.
- [ ] Overlay PNGs for the camera panel: the NAVCAM frame with the class-colour mask at 40% alpha, served from `terrain/overlays/<id>.png`.

### Hour 19 to 21: 3D view (P1, cut first if behind)
- [ ] `frontend/src/Terrain3D.tsx` with React Three Fiber: a `PlaneGeometry(50, 50, 199, 199)`, vertex z from `/api/world/{seed}` elevation (exaggerate it about 3×), vertex colours from belief classes, rover mesh, path as a `Line` from drei.

### Hour 21 to 24: slides and rehearsal
- [ ] Slides: data sources (AI4Mars, USGS DTM, Horizons), architecture, segmentation metrics, results table vs baselines, throughput.
- [ ] **Honesty notes on a slide:** the Curiosity NAVCAM vs Perseverance mismatch, the flat-ground homography, and that the geometric obstacles are simulated.
- [ ] Rehearse with B three times.

---

## Cut order if behind
Multi-rover (B's), Horizons (B's), then **3D view**, then **calibration**. Never cut G3 or the harness.

## Interfaces you promise to B

```python
build_world(seed: int) -> World                                  # hour 6
observe(world, belief, pose, sensor, flags, mode) -> ObserveResult  # oracle now, model at hour 12
cost_grid(world, belief, cfg) -> (cost float32 >= 1, blocked bool)  # now (reference version)
GET-able files: eval/results.json, eval/seg_metrics.json, terrain/overlays/<id>.png
```

## Gotchas
- `test_min3` is both your test set and the world tiles. **Never train or tune on it.** Calibrate on your validation split.
- Check that no `test_min3` image id appears in the training subset.
- Big rock is about 0.5% of pixels. Without class weights the model will predict it close to never.
- Coordinates are `(row, col)`. Heading 0 is north (row decreasing), clockwise. `sensor_cone()` already follows this.
