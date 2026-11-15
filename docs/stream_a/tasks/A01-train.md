# A01: Train the terrain segmentation model  `[GPU]` `[long-running]`

**Wave 1, start immediately.** It's on the critical path: A05, A06, A07b, A10 and A11 all wait for its weights.

## Goal
A U-Net that segments AI4Mars NAVCAM images into soil, bedrock, sand and big rock, saved as deployable weights.

## You own (create or edit only these)
- `ml/model.py`: `build_model()`, `preprocess()`, `load_weights()`. **Commit and push this file first, within about 30 min.** A07b and A06 import it.
- `ml/train.py`
- `ml/weights/unet_r34_fp16.pt` (committed, must be under 50 MB)
- Anything under `$RR_ARTIFACTS/checkpoints/` and `$RR_ARTIFACTS/runs/train/`

## Do not touch
`backend/app/contracts.py`, any other task's files.

## Inputs
- `ml/ai4mars.py`: `cache_train_subset()`, `train_val_split()`, `clean_label()`. Use these and don't write your own loader.
- `world/paths.py`: `WEIGHTS`, `CKPT_DIR`, `RUNS_DIR`

## Spec
`ml/model.py` (the single definition, shared by training and inference):
```python
INPUT_SIZE = 256                       # bump to 512 only if time allows; keep it a constant here
def build_model() -> torch.nn.Module   # smp.Unet("resnet34", encoder_weights="imagenet", classes=4)
def preprocess(img_u8: np.ndarray, size: int = INPUT_SIZE) -> torch.Tensor  # (3,H,W) float, ImageNet norm, gray->3ch
def load_weights(path=WEIGHTS, device="cpu") -> torch.nn.Module  # eval mode, fp32 on CPU
```
`ml/train.py`:
- `ds = cache_train_subset(6500)`, then `train, val = train_val_split(ds, 500)`. The fixed val split matters because A10 calibrates on the same 500 rows.
- Labels: `clean_label(label, rover_mask, range_mask)`, resized with **nearest** to `INPUT_SIZE`. Images: `preprocess`.
- Augmentation: horizontal flip, small brightness/contrast jitter. No vertical flip (it breaks the sky-to-ground layout).
- Loss: `CrossEntropyLoss(weight=w, ignore_index=255)` + `smp.losses.DiceLoss("multiclass", ignore_index=255)`. Compute starting weights from inverse sqrt class frequency over the train subset, clip to [1, 20], print them. Big rock should end up around 10 to 20.
- AdamW lr 3e-4, cosine schedule, AMP (`torch.autocast("cuda")`), batch 16, about 15 to 25 epochs or until val mIoU plateaus. Use `num_workers=0` on Windows if DataLoader workers hang.
- Every epoch: log train loss, val mIoU and per-class IoU to `$RR_ARTIFACTS/runs/train/log.csv` and stdout. Save the best by val mIoU to `$RR_ARTIFACTS/checkpoints/best.pt`.
- At the end: save `model.half().state_dict()` to `ml/weights/unet_r34_fp16.pt` and check the size is under 50 MB.
- CLI: `--epochs`, `--limit N` (smoke test on N rows), `--size`.

## Run
```bash
$PY ml/train.py --limit 64 --epochs 1      # smoke test first (< 3 min)
$PY ml/train.py                            # real run, in the background; tail log.csv
```
The first `cache_train_subset` call downloads several GB. If HF streaming or download is flaky on Windows (WinError 10038), retry. It caches to `$RR_ARTIFACTS/data/` once.

## Acceptance
- [ ] Smoke run completes and writes the log and a checkpoint
- [ ] Real run: loss falls; val mIoU is logged per epoch; big-rock IoU is reported separately
- [ ] `ml/weights/unet_r34_fp16.pt` exists and is under 50 MB; `load_weights()` + `preprocess()` run on one test image on CPU
- [ ] No `test_min3` data touched anywhere in training or model selection

## Hand back
Final val mIoU, per-class IoU, epochs, wall time and class weights used. Push `ml/model.py`, `ml/train.py` and the fp16 weights.
If big-rock IoU < 0.15: report it. The orchestrator decides whether to retrain with a higher weight (in the background while A05/A06 use the current weights).
