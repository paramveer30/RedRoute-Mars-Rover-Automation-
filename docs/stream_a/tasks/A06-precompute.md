# A06: Precompute warped predictions for all tiles  `[GPU]`

**Wave 3.** Needs A01 weights and A03 (`world/warp.py`, `world/tiles.npz`) on main. About 30 min.

## Goal
Cache the model's per-cell class probabilities for every `test_min3` tile, so `observe()` runs instantly during demos and in the parallel harness.

## You own
- `ml/precompute.py`
- `world/preds20.npz` (committed, about 1 to 2 MB)
- `$RR_ARTIFACTS/logits/test_logits.npz`, `$RR_ARTIFACTS/logits/val_logits.npz`

## Inputs
`ml/model.py`, `ml/ai4mars.py`, `world/warp.py` (`warp_probs`), `world/tiles.npz` (`ids`, `H`).

## Spec
- For each of the 322 test images: logits at `INPUT_SIZE`, then `softmax`, then `warp_probs(probs, H[i])` → (20, 20, 4).
- Save `np.savez_compressed(PREDS_NPZ, ids=..., probs=<(322,20,20,4) float16>, temperature=1.0)`.
- **Calibration inputs for A10** (don't apply them here): save raw logits, downsampled to 64 × 64 per image together with matching cleaned labels, for the **val** split (from `train_val_split`) and the test split. Store as `logits (N,4,64,64) float16`, `labels (N,64,64) uint8`.
- `--temperature T` flag: if `ml/calibration.json` exists, also write the probabilities computed at that T (A10 asks for a rerun after it lands).
- Sanity check: for usable tiles, report agreement between `argmax(preds20)` and `label20` over labelled cells. This is the tile-level accuracy that actually drives the planner.

## Acceptance
- [ ] `preds20.npz` loads; shape (322, 20, 20, 4); each cell sums to 1 (±1e-2)
- [ ] Tile-level agreement printed (expect roughly the pixel accuracy from A05)
- [ ] Logit files written for A10

## Hand back
The tile-level agreement number and the file sizes.
