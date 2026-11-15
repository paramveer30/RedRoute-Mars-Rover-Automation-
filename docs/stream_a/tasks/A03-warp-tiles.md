# A03: Ground-plane warp and the tile library

**Wave 1, start immediately.** CPU only, about 2 to 3 hours. It's the riskiest geometry step, so start it early.

## Goal
Turn each of the 322 `test_min3` NAVCAM images into a 5 m × 5 m top-down tile (20 × 20 cells) of expert labels, with one homography per image that every later task reuses exactly.

## You own
- `world/warp.py`: the homography and warp functions. **Everyone else imports these. Push an early version within about 1 hour.**
- `world/make_tiles.py`
- `world/tiles.npz`, `world/usable_tiles.json` (committed, small)
- `backend/tests/test_warp.py`
- `$RR_ARTIFACTS/tiles_img/<id>.png` (160 × 160 top-down image patches), `$RR_ARTIFACTS/runs/warp/*.png`

## Inputs
- `ml/ai4mars.py`: `load_test()`, `tile_id_for(i)` (ids are `m3_0000` to `m3_0321`), `clean_label()`
- Facts verified in setup: images are 1024 × 1024 uint8. `rover_mask==1` is the rover body at the **bottom** of the frame (18 to 44% of pixels, varies per image). `range_mask==1` is beyond 30 m at the **top**. 73 to 85% of each image is label 255; **only a band just above the rover is labelled.**

## Spec
`world/warp.py`:
```python
OUT_PX = 160            # top-down patch resolution for images and probabilities
CELLS = 20              # TILE from contracts
def homography_for(rover_mask: np.ndarray) -> np.ndarray      # 3x3 float64, image px -> top-down px
def warp_label(label, H) -> np.ndarray                         # (20,20) uint8; INTER_NEAREST then 8x8 majority vote ignoring 255
def warp_image(img, H) -> np.ndarray                           # (160,160) uint8; INTER_LINEAR
def warp_probs(probs_hw4, H) -> np.ndarray                     # (20,20,4) float32; INTER_LINEAR per channel then 8x8 mean; renormalize
```
- Base trapezoid: 4 source points on the image's near-field ground (bottom edge wide, top edge narrow), mapped to the 160 × 160 square. Top of the patch = far, bottom = near. Tune the base trapezoid **by eye** on about 10 images using contact sheets.
- **Per-image vertical offset:** find the rover body's top edge in the central columns (`rover_mask[:, 300:724]`, the topmost row where most pixels are 1). Shift the trapezoid so its bottom edge sits a few px above it. This is the only per-image parameter. `homography_for` computes it from the mask, so it stays deterministic.
- `warp_probs` must accept probabilities at **any resolution** (A06 feeds model output at 256 px). Scale `H` for that resolution internally (`H @ diag(1024/s, 1024/s, 1)`).

`world/make_tiles.py`:
- For all 322 rows: H, label20, image160, and the labelled fraction of label20 (cells not 255).
- **usable** = labelled fraction ≥ 0.6 (tune: you want at least 60 usable. If you get fewer, loosen to 0.5 and say so).
- Save `np.savez_compressed(TILES_NPZ, ids=<str array>, H=(N,3,3), label20=(N,20,20) uint8, valid_frac=(N,))` for **all 322** (A06/A11 need H for all), and `usable_tiles.json` = sorted list of usable ids.
- Contact sheet: 12 random images showing the original with the trapezoid drawn, the warped image, and the warped label in class colours, saved to `$RR_ARTIFACTS/runs/warp/contact_*.png`. **Look at it.** Ground should look like ground seen from above, and labels should line up with the image.
- Print the class distribution across usable tiles. At least some tiles must contain sand and some big rock; if not, report it.

## Tests (`backend/tests/test_warp.py`)
- `homography_for` on a synthetic mask (rover top at row 700) maps the trapezoid corners to the patch corners
- `warp_label` output is (20,20) uint8 with values in {0,1,2,3,255}
- `warp_probs` rows sum to 1, and it works on 256 px and 1024 px inputs with the same H
- an image that's all one class warps to all that class

## Acceptance
- [ ] Contact sheets look right (include one in the hand-back)
- [ ] At least 60 usable tiles, with sand present in several
- [ ] Tests pass; `tiles.npz` + `usable_tiles.json` committed

## Hand back
Usable count, valid_frac threshold, class distribution, contact-sheet path, and any images where the warp is clearly wrong.

## Honesty note for the slides
Flat-ground inverse perspective mapping with no camera intrinsics. It's an approximation, and the slides say so.
