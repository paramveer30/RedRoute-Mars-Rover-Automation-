# A05: Segmentation metrics on the expert test set  `[GPU]`

**Wave 3.** Needs A01 weights on main (`ml/weights/unet_r34_fp16.pt`). About 30 to 45 min.

## Goal
Honest numbers for the slides: mIoU, per-class IoU, a confusion matrix, and CPU latency.

## You own
- `ml/evaluate.py`
- `eval/seg_metrics.json` (committed)
- `$RR_ARTIFACTS/runs/eval/confusion.png`

## Inputs
`ml/model.py` (`load_weights`, `preprocess`, `INPUT_SIZE`), `ml/ai4mars.py` (`load_test`, `clean_label`).

## Spec
- Run the model on all 322 `test_min3` images. Upsample logits to 1024 (bilinear) and compare with `clean_label(...)`, ignoring 255.
- Accumulate one 4 × 4 confusion matrix over all pixels, then compute per-class IoU, mIoU, pixel accuracy and per-class pixel share.
- **Latency:** CPU, batch 1, `torch.inference_mode()`, model loaded once, median and p95 over 50 images at `INPUT_SIZE`. Also GPU throughput (images/s at batch 16) for the scalability slide.
- Also report mIoU on the **val** split from A01 (via `train_val_split`), so the gap between val and test is visible.
- Write JSON:
```json
{"split":"test_min3","n_images":322,"input_size":256,"miou":0.0,
 "iou":{"soil":0,"bedrock":0,"sand":0,"big_rock":0},"pixel_acc":0,"class_share":{...},
 "val_miou":0,"cpu_latency_ms":{"p50":0,"p95":0},"gpu_imgs_per_s":0,"weights":"ml/weights/unet_r34_fp16.pt"}
```
- Confusion matrix PNG (row-normalized) for the slides.

## Acceptance
- [ ] JSON and PNG written; numbers printed as a markdown table
- [ ] Never uses test results to change training (that would be reported as leakage)

## Hand back
The markdown table. **If big-rock IoU < 0.15**, say so explicitly. The orchestrator may relaunch A01 with a higher big-rock weight.
