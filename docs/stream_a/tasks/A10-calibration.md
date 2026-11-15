# A10: Temperature scaling and ECE  (P1)

**Wave 3 or 4.** Needs A06's logit files. CPU, about 45 min.

## You own
- `ml/calibrate.py`
- `ml/calibration.json` (committed)
- `$RR_ARTIFACTS/runs/calib/reliability.png`

## Spec
- Load `$RR_ARTIFACTS/logits/val_logits.npz` (**val only for fitting**) and `test_logits.npz` (reporting only).
- Fit a scalar T > 0 by minimising NLL over the non-255 pixels (`torch.optim.LBFGS` on `log T`).
- ECE with 15 equal-width confidence bins, before and after, on val and on test.
- Reliability diagram (before vs after) PNG for the slides.
- `calibration.json`: `{"T": 1.37, "ece_val_before":..., "ece_val_after":..., "ece_test_before":..., "ece_test_after":..., "nll_before":..., "nll_after":...}`
- Then ask the orchestrator to rerun `ml/precompute.py --temperature T` (A06), so `preds20.npz` ships calibrated.

## Acceptance
- [ ] T fitted on val only; ECE on test improves (or report honestly if it doesn't)
- [ ] JSON and PNG written

## Hand back
T and the before/after ECE table.
