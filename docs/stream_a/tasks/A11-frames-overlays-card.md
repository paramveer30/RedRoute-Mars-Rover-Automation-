# A11: Camera frames, overlays, and route decision card data

**Wave 3 or 4.** Frames and overlays need A01 weights. The decision card needs B's planner and A07. About 2 hours.

## You own
- `ml/make_overlays.py`
- `terrain/frames/<id>.jpg`, `terrain/overlays/<id>.png` (committed; 384 px, keep the total under about 15 MB)
- `eval/decision_card.py`
- `backend/tests/test_decision_card.py`

## Part 1: frames and overlays (for B's camera panel)
- For all 322 test images: `terrain/frames/<id>.jpg` = the NAVCAM image resized to 384 px, JPEG quality 85.
- `terrain/overlays/<id>.png` = an RGBA class mask from the model prediction at 384 px, alpha 0 where the prediction is low-confidence (U > 0.7) or under the rover mask, alpha 110 elsewhere. Colours (keep them identical to B's map): soil `#c8a165`, bedrock `#8a8f98`, sand `#f2d16b`, big rock `#e5484d`.
- B serves these as static files. Tell B the paths.

## Part 2: route decision card data
```python
def decision_card(world, belief, cfg, start, goal, chosen_path) -> dict
```
- `shortest`: `plan()` on a cost map with the semantic terms removed (`use_semantics=False`, `w_u=0`), so it's the geometry-shortest route.
- Return:
```json
{"chosen_len_m":0, "shortest_len_m":0, "extra_m":0, "extra_pct":0,
 "avoided": {"class":"sand", "cells":0, "mean_p":0.0, "mean_conf":0.0},
 "chosen_max_unc":0.0, "summary":"Took a 6.2 m (14%) longer route to avoid 31 cells of predicted sand (mean P=0.83)."}
```
- `avoided` = belief cells on the shortest path but not on the chosen path, grouped by argmax class; report the dominant hazard class. Use `mapping.explain_cell` from A08 if it's there.
- B renders the card. Agree on the field names with B before finishing.

## Acceptance
- [ ] Frames and overlays generated, total size checked
- [ ] Test: on a synthetic world with a sand band, the card reports sand avoided and extra_m > 0
- [ ] Summary sentence reads naturally

## Hand back
Paths, total size, a sample card JSON.
