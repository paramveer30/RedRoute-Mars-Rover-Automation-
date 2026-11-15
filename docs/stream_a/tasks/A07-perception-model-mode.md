# A07: `observe()` model mode: fusion, uncertainty, and live inference

Two parts:
- **A07a (Wave 2):** model mode driven by cached predictions. Start once A03 is on main. If A06 isn't done yet, generate **fake** `preds20` for development (one-hot of `label20` blended with Dirichlet noise) in a throwaway script. Never commit the fake file over the real one.
- **A07b (Wave 3):** live inference for the blur/noise fault. Needs A01 weights.

It's the core of gate G3 (hour 12). About 3 hours total.

## You own
- `backend/app/perception.py` (the oracle branch already exists; keep it working)
- `ml/infer.py`: model singleton and the live path
- `backend/tests/test_perception.py`

## Inputs
`backend/app/contracts.py` (`Belief`, `ObserveFlags`, `ObserveResult`, `SensorConfig`, `PerceptionMode`), `world/preds20.npz`, `world/tiles.npz` (H), `world/warp.py`, `ml/model.py`, `ml/ai4mars.py`, `world/builder.py` (`parse_tile_ref` if A04 used rotations), `ml/calibration.json` (optional, A10).

## Contract change needed (non-breaking, ask the human first)
Fusion needs a per-cell accumulated weight. Propose adding to `Belief`:
```python
weight: np.ndarray | None = None   # (200,200) float32, sum of confidence weights; None -> zeros
```
with `Belief.empty()` initialising it to zeros. **Stream B must agree** before you push a contracts.py change. Tell the orchestrator. If it's refused, keep a module-level `WeakKeyDictionary` keyed by Belief instead.

## Spec: model mode in `observe()`
1. Compute the `cone` as now. Geometric channel unchanged.
2. For each tile intersecting the cone: look up its source id from `world.tile_image[tr][tc]`, take `p_new = preds20[id]` (apply the tile's rotation if any), shape (20, 20, 4).
3. `U_new = -Σ p ln p / ln 4` per cell, and `w_new = 1 - U_new` (floor it at 0.05 so a fully uncertain frame still counts a little).
4. For cells in `cone ∩ tile`: `p = (W·p_old + w_new·p_new) / (W + w_new)`, `W += w_new`. On first sight (`W == 0`), `p = p_new`.
5. Recompute `belief.unc` on the touched cells from the fused p. Set `known = True`. `version += 1`.
6. `frame_image_id` = the source id of the tile straight ahead of the rover (pose cell + about 4 m along heading, clamped), falling back to the rover's own tile. That's what B's camera panel shows.
7. `latency_ms` measured.

**Temperature:** if `ml/calibration.json` exists, use probabilities at that T. The simplest way is for A06 to have written calibrated `preds20`. Otherwise use T = 1 and say so.

## Spec: live path (A07b), `ml/infer.py`
```python
def get_model()                                            # lazy CPU singleton via ml.model.load_weights
def predict_tile(tile_id: str, blur: float = 0.0, noise: float = 0.0) -> np.ndarray  # (20,20,4)
```
- Load the source image by index from `load_test()` (cache the dataset object), degrade it (`blur` → Gaussian sigma up to about 8 px; `noise` → Gaussian noise up to sigma 40), run the model at `INPUT_SIZE`, softmax with T, then `warp_probs(probs, H[idx])`.
- In `observe()`: if `flags.blur > 0`, use `predict_tile(id, blur=flags.blur)` for the tiles in view instead of the cache. **Uncertainty must visibly rise**, and a test checks it.
- `flags.model_offline` or `mode == "offline"`: no semantic update, geometry only (already stubbed).
- Inference must not crash the sim. On exception, return `ObserveResult(ok=False, reason="model_error: ...")`.

## Tests (`backend/tests/test_perception.py`)
- model mode on `build_world(seed)`: belief becomes known only inside the cone; probabilities sum to 1; `unc` in [0, 1]
- fusion: observing the same tile twice with identical predictions leaves p unchanged and increases W
- confident beats unsure: fusing a confident and a uniform prediction lands closer to the confident one
- blur: mean U over the viewed cells is higher with `blur=0.8` than with `blur=0` (skip if weights are missing)
- oracle mode still works (the existing tests stay green)
- latency: cached observe < 20 ms; live one-tile predict < 1 s on CPU

## Acceptance (= Stream A's half of gate G3)
- [ ] With `mode="model"` on a demo seed, B's planner routes differently than with uniform belief (detours predicted sand)
- [ ] The blur fault raises U on the uncertainty heatmap
- [ ] All tests pass

## Hand back
Cached and live latency numbers, whether the Belief contract change was made, and how to switch modes (one line for B).
