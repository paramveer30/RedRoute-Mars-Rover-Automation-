# Track A: plan to finish

Status on 2026-11-15, origin/main `8cea8e4`.

**Done:** A01 to A08 and A11a. That covers the trained model, terrain, tiles, world builder, cached and live perception, mapping, segmentation metrics and camera assets.
**Still to do:** five items, listed below. Two of them (real A09 runs and the G3 route-change gate) need Stream B's `planner.plan` and `Rover`, which are still stubs (`backend/app/planner.py:17`, `backend/app/rover.py:19`).

The approach is to build everything A can build now against interfaces we define ourselves. When B's code lands, we plug it in, and the integration step becomes small.

---

## 1. Your items mapped to tasks

| # | Your item | Task | Blocked by B? |
|---|---|---|---|
| 1 | Metrics + synthetic tests (hazard entries, success, route length, CIs) | **A09a** | No |
| 2 | Harness around the agreed Rover interface | **A09a** (against a fake rover) then **A09b** (real runs) | A09b only |
| 3 | Blur behaviour + unresolved acceptance test | **A07c** | No |
| 4 | Decision-card calculations + tests | **A11b** (planner injected) then **A11c** (wire to B's planner) | A11c only |
| 5 | Calibration (optional) | **A10** (fit) then **A10b** (apply: regenerate cache + overlays) | No |

A12 (3D view) stays P1 and is the first thing to cut.

## 2. Subagents, models and worktrees

All worktrees are created detached from `origin/main` under `C:/Pranoy/Machacks/wt/`.

| Task | Model | Why this model | Worktree | Wave | GPU |
|---|---|---|---|---|---|
| A09a metrics + harness | **opus** | Subtle correctness (footprint overlap, run counting, bootstrap, Windows multiprocessing) | `wt/A09` | 1 | no |
| A07c blur investigation | **opus** | Open-ended diagnosis; must not "manufacture" entropy | `wt/A07c` | 1 | no (CPU inference) |
| A11b decision card | **sonnet** | Well-specified calculation + one synthetic test | `wt/A11b` | 1 | no |
| A10 calibration fit | **sonnet** | Textbook temperature scaling + ECE | `wt/A10` | 1 | no |
| A10b apply calibration | **sonnet** | Rerun `ml/precompute.py --temperature T` + `ml/make_overlays.py`, check consistency | `wt/A10b` | 2 (after A07c lands) | **yes** |
| A09b real mission runs | **opus** | Interpreting results honestly, debugging integration | `wt/A09` (reuse) | 3 (after B's Rover) | no |
| A11c decision card on real planner | **sonnet** | Small wiring + real-mission validation | `wt/A11b` (reuse) | 3 (after B's planner) | no |
| A12 3D view (P1) | **sonnet** | Frontend, spec is concrete | `wt/A12` | 4 (after P0 freeze) | no |
| Docs sync | **haiku** | Update WAVE1_EXECUTION / README status from this plan | main checkout, me | 0 | no |

Max 4 run at once (wave 1). GPU rule: only A10b uses the GPU, and it runs alone.

## 3. Waves

```
Wave 0 (now, 10 min, orchestrator)
  remove stale clean worktrees wt/A05 wt/A06 wt/A07 wt/A11 (all published)
  agree two interfaces with B (section 4), docs sync

Wave 1 (parallel, ~2-3 h)            Wave 2              Wave 3 (needs B)          Wave 4
A09a  metrics + harness + fake rover ─────────────────▶ A09b real runs N=20→100
A07c  blur investigation ────────────▶ A10b apply T ──▶ (re-verify caches)
A11b  decision card (injected planner) ───────────────▶ A11c real planner          A12 3D (P1)
A10   calibration fit ───────────────▶ A10b
                                                       G3 gate (human + B)
```

A10b waits for A07c to land because both touch `ml/infer.py` and the prediction cache that the blur tests read.

## 4. Decisions to settle with B before wave 1 (human)

1. **Rover result fields** (A09 builds against this as a `Protocol` in `eval/`; it doesn't edit B's files):
   ```python
   Rover(world, mode, cost_cfg, sensor).run_to_completion(start, goal, max_steps) -> MissionResult
   MissionResult: reached: bool, final_state: str, trajectory: list[Cell]   # EVERY traversed cell, not one per tick
                  path_len_m: float, steps: int, replans: int,
                  replan_latencies_ms: list[float], observe_calls: int
   ```
   `trajectory` and `observe_calls` are missing from B's list. A09 needs both: the trajectory to catch hazard crossings between ticks, and `observe_calls` for throughput. Modes: `geometry_only | model | oracle`.
2. **Decision-card JSON fields.** Use the A11 brief's schema unchanged unless B wants to change it.
3. **Route-ratio denominator** (review finding 4). Recommendation: shortest **hazard-free** 8-connected route on truth (sand, big rock and slope over the limit blocked, footprint 2 cells), computed by an eval-only Dijkstra in `eval/`. That keeps the denominator consistent with the success definition and doesn't depend on B's planner. Scenarios whose goal is unreachable under that rule are resampled at generation time, with the same seed each time.

## 5. Per-task briefs (handed to each subagent with its brief file)

Every subagent gets the following: its brief file (if one exists), this section, worktree path, `PY=C:/Pranoy/Machacks/RedRoute-Mars-Rover-Automation-/.venv/Scripts/python.exe`, `RR_ARTIFACTS=C:/Pranoy/Machacks/rr-artifacts`. It also gets this instruction: "Follow AGENTS.md. Edit only the files listed under 'You own'. Never read `World.truth` outside eval / sim hazard checks / ORACLE. Report measured numbers only. Finish with: fetch, rebase origin/main, pytest green, `git push origin HEAD:main`. Commit messages are plain sentences, with no `A##:` prefix and no AI attribution."

### A09a: metrics, harness, fake rover (opus, `wt/A09`)
- **Own:** `eval/metrics.py`, `eval/harness.py`, `eval/report.py`, `eval/rover_api.py` (Protocol + `FakeRover` for tests), `backend/tests/test_eval_metrics.py`, `backend/tests/test_eval_harness.py`.
- **Metrics:** hazard entries from the trajectory (footprint radius 2 cells overlapping truth sand, big rock or steep slope; truth 255 is not a hazard; contiguous runs count once), success (reached within 1 m and 0 entries), route ratio against the section 4.3 denominator, 95% bootstrap CIs (seeded, 10k resamples) on success rate and hazard entries.
- **Harness:** scenario generator (seeded starts and goals ≥ 30 m apart on non-blocked truth cells, identical across configs), 4 configs as in the A09 brief, `multiprocessing.Pool` with `__main__` guard + initializer, CLI `--n --workers --configs --rover {fake,real}`. `--rover real` imports `backend.app.rover.Rover` and fails with a clear message while it's a stub.
- **Tests:** synthetic trajectories (contiguous run = 1, footprint overlap, 255 ignored, a diagonal crossing between ticks is caught), success logic, ratio = 1 for the optimal path, bootstrap determinism, and a `--rover fake --n 4` end-to-end smoke test.
- **Hand back:** metric definitions, fake-run timing, open questions for B.

### A07c: blur behaviour (opus, `wt/A07c`)
- **Own:** `backend/app/perception.py`, `ml/infer.py`, `backend/tests/test_perception.py`.
- **Context:** `test_real_demo_blur_uncertainty_increase` is xfail. On a fresh belief, blur=0.8 measured mean U 0.000818 → 0.000660. Cached-clear followed by live-blur rose to 0.01304 (review finding 2).
- **Investigate:** what U is (aggregation from 64×64 logits to 20×20 cells, entropy vs 1−max-p, temperature), how blur is applied (kernel, before or after resize), U per blur level 0 to 1, across many tiles rather than one seed, and the actual rover sequence (accumulated belief over consecutive ticks).
- **Fix:** change code only if there's a real defect, such as blur applied at the wrong stage or U computed in a way that hides disagreement. Then replace the xfail with an acceptance criterion grounded in measurement, e.g. "on the rover's tick sequence, fused U over blurred tiles rises vs. the clear run on ≥ X% of N tiles", with X set from data and reported. If blur genuinely doesn't raise model entropy, document that and make the test assert the honest property. Never add artificial entropy.
- **Hand back:** the U-vs-blur table, the root cause, and the new criterion with its measured pass rate.

### A11b: decision card (sonnet, `wt/A11b`)
- **Own:** `eval/decision_card.py`, `backend/tests/test_decision_card.py`.
- Implement `decision_card(world, belief, cfg, start, goal, chosen_path, planner=None)` according to the A11 brief. `planner` defaults to `backend.app.planner.plan`. The tests inject a small Dijkstra, because B's planner is a stub. Uses `mapping.cost_grid` / `explain_cell` (A08). Belief-only: never touch `world.truth`.
- **Tests:** a synthetic sand band reports `avoided.class == "sand"` and `extra_m > 0`; identical paths give `extra_m == 0` with a sensible summary; the summary sentence reads naturally.
- **Hand back:** a sample card JSON.

### A10: calibration fit (sonnet, `wt/A10`)
- Follow `docs/stream_a/tasks/A10-calibration.md` exactly. Fit on val only, report test.
- **Own:** `ml/calibrate.py`, `ml/calibration.json`, `backend/tests/test_calibrate.py` (ECE on synthetic perfectly-calibrated data ≈ 0; T = 1 when logits are already calibrated).
- Do **not** rerun precompute. That's A10b.
- **Hand back:** T and the ECE/NLL before/after table. If ECE on test doesn't improve, report that and recommend not applying T.

### A10b: apply calibration (sonnet, `wt/A10b`, GPU, after A07c lands and only if A10 improved test ECE)
- Rerun `ml/precompute.py --temperature T` and `ml/make_overlays.py`. Confirm `test_published_cache_matches_live_cpu_predictions` runs (not skipped) and passes, and that overlays stay under 15 MB.
- **Own:** the regenerated `world/preds20.npz`, `terrain/overlays/*`.

### A09b: real runs (opus, `wt/A09`, after B's Rover lands + G3 passes)
- Adapt `eval/rover_api.py` to B's real names if they differ. Run N=20, sanity-check, then N ≥ 100. Write `eval/results.json` and `eval/results.md`.
- If RedRoute doesn't beat geometry-only, report it, then investigate the cost weights as a separate, clearly labelled experiment. Never edit numbers.

### A11c: real planner (sonnet, `wt/A11b`, after B's planner)
- Remove the test-only planner default where appropriate, validate one card on a real seed-1 mission, and hand B the schema and frame paths.

## 6. Human checkpoints (Pranoy)

| When | You |
|---|---|
| Before wave 1 | Settle section 4 with B (Rover fields, card fields, denominator) |
| After A07c | Read the U-vs-blur table and approve the new acceptance criterion |
| After A10 | Decide whether to apply T (only if test ECE improves) |
| G3 | With B: real world, model mode, route changes because of model output |
| After A09b | Sanity-check the results table before slides |

## 7. Acceptance for "Track A finished"

- [ ] `eval/metrics.py` + harness + tests green; `python -m eval.harness --rover fake --n 4` works
- [ ] Blur xfail replaced by a measured, honest criterion that passes
- [ ] `eval/decision_card.py` + synthetic test green
- [ ] (P1) `ml/calibration.json` committed; cache regenerated only if ECE improved
- [ ] After B: G3 witnessed, `eval/results.{json,md}` from N ≥ 100, card validated on a real mission
- [ ] Full `pytest -q` green, `npx tsc -b` green
