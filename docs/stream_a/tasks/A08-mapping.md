# A08: Harden and test `cost_grid()`

**Wave 1, start immediately.** CPU, about 1 hour. Small, so it suits a quick subagent.

## Goal
`backend/app/mapping.py` already implements PRD 7.3 (it was written during setup). Prove it with tests, fix the edge cases, and make it fast.

## You own
- `backend/app/mapping.py`
- `backend/tests/test_mapping.py`

## Inputs
`backend/app/contracts.py` (`Belief`, `CostConfig`, `World`, `RES_M`), `world/stub_world.py` (until A04 lands).

## Spec and tests
- **cost floor:** `cost >= 1.0` everywhere, for any belief. A*'s heuristic (Stream B) relies on this. Add `np.maximum(cost, 1.0)` defensively.
- **slope block:** a cell with slope 25° is blocked; one at 15° is not (before inflation).
- **big rock:** a known cell with P(big rock) = 0.9 in a blob of at least `min_blob_cells` is blocked; a single isolated cell is filtered as noise; an unknown cell is never semantically blocked.
- **geo obstacle:** only blocks once `belief.geo_seen`, never from `world.geo_obstacle` directly (the rover can't see through rocks it hasn't sensed).
- **inflation:** one blocked cell grows into a disk of radius `round(1.5 / 0.25) = 6` cells; check the count.
- **unknown:** unknown cells get `U = 0.5` and no S term.
- **geometry-only** (`use_semantics=False`): belief probabilities have no effect on cost or blocked.
- **ablation** (`w_u = 0`): uncertainty has no effect on cost.
- **sand dearer than bedrock:** for confident one-hot belief, cost(sand) > cost(soil) > cost(bedrock).
- **performance:** `cost_grid` on 200 × 200 in under 15 ms (pre-compute the disk structure once at module level; the tests time it).
- Return dtypes: cost float32, blocked bool.

Also add `explain_cell(world, belief, cfg, cell) -> dict` returning each term (S, R, U, slope, blocked reason). The route decision card and UI tooltips use it.

## Acceptance
- [ ] All of the above as tests; they pass
- [ ] No change to the function signature (Stream B calls it)

## Hand back
The test list and timing.
