# Stream A: Perception, World, and Evaluation (orchestration playbook)

**Owner:** Pranoy (human), working with an orchestrator agent (Codex) that spawns subagents, one git worktree each.

**Stream A owns:** everything that turns real NASA data into the rover's picture of the world, plus the numbers that prove it works.
Files: `ml/`, `world/`, `eval/`, `backend/app/perception.py`, `backend/app/mapping.py`, Stream A tests in `backend/tests/`, `frontend/src/Terrain3D.tsx`.
**Never edit without Stream B agreeing:** `backend/app/contracts.py`. Stream B owns `planner.py`, `rover.py`, `link.py`, `faults.py`, `api.py` and the rest of `frontend/`.

Task briefs live in [`docs/stream_a/tasks/`](stream_a/tasks/). Each one is self-contained: hand it to a subagent unchanged. Repo-wide agent rules: [`AGENTS.md`](../AGENTS.md).

---

## 1. Rules for the orchestrator

1. **Main only, no branches.** Worktrees are created **detached** (`--detach`). Subagents push with `git push origin HEAD:main`. Never create a named branch, never push one.
2. **No AI attribution.** No `Co-Authored-By` or "Generated with" lines in commits. Only the human appears as author.
3. **One task = one subagent = one worktree.** A subagent edits only the files its brief lists under "You own". If it needs another file changed, it stops and reports.
4. **`contracts.py` changes need the human's OK** (who checks with Stream B). A07 proposes one (a `Belief.weight` field), so escalate it rather than pushing it.
5. **One GPU job at a time.** Tasks tagged `[GPU]` (A01, A05, A06) never overlap. While A01 trains, A05/A06 wait.
6. **Tests green before every push:** `$PY -m pytest -q` in the worktree after rebasing on `origin/main`.
7. **Honest numbers only.** Results, metrics and ECE are whatever the code measured. A disappointing number gets reported, never massaged.
8. Commit message format: `A03: ground-plane warp and tile library`.

## 2. Environment (every worktree)

```bash
# Shared Python env (lives in the main checkout; worktrees reuse it, no extra venvs)
PY=C:/Pranoy/Machacks/RedRoute-Mars-Rover-Automation-/.venv/Scripts/python.exe
# Shared artifacts dir. Worktrees do NOT share gitignored files, so all big outputs go here
RR_ARTIFACTS=C:/Pranoy/Machacks/rr-artifacts        # set as a Windows user env var
```
- The Hugging Face cache (`~/.cache/huggingface`) is shared automatically; `test_min3` is already cached.
- `world/paths.py` is the only place paths are defined. `ml/ai4mars.py` is the only dataset loader. Both exist already.
- Run Python from the worktree root (pytest uses `pythonpath = .`).

## 3. Worktree lifecycle

```bash
MAIN=C:/Pranoy/Machacks/RedRoute-Mars-Rover-Automation-
# spawn
git -C $MAIN fetch origin
git -C $MAIN worktree add --detach C:/Pranoy/Machacks/wt/A03 origin/main
# ...subagent works inside C:/Pranoy/Machacks/wt/A03...
# finish (run by the subagent)
git add <owned files only>
git commit -m "A03: ground-plane warp and tile library"
git fetch origin && git rebase origin/main && $PY -m pytest -q && git push origin HEAD:main
#   push rejected (someone pushed first)? -> fetch, rebase, test, push again
# cleanup (orchestrator)
git -C $MAIN worktree remove C:/Pranoy/Machacks/wt/A03
git -C $MAIN pull --rebase
```
Long-running tasks (A01 training) can push code early (`ml/model.py` first) and keep their worktree alive while training runs.

## 4. Task graph

```
Wave 1 (t=0, parallel)        Wave 2 (~h3)            Wave 3 (after weights, ~h6-9)       Gate      Wave 4 (h12+)
A01 train [GPU, long] ──────────────────────────────▶ A05 evaluate [GPU]                            A09 harness ◀─ B's Rover
   └─ ml/model.py early ─────────────────────────────▶ A06 precompute [GPU] ─▶ A10 calibrate (P1)    A11b decision card
A02 terrain ───────────┐                               A07b live inference                          A12 3D view (P1)
A03 warp + tiles ──────┼─▶ A04 world builder ──▶ B     A11a frames + overlays                       slides (human)
                       └─▶ A07a perception (cached) ─────────────────────────────────▶ G3 (h12)
A08 mapping tests
```

| ID | Task | Wave | Depends on | GPU | Est. | Brief |
|---|---|---|---|---|---|---|
| A01 | Train U-Net | 1 | n/a | yes, long | 30 min setup, 2 to 4 h train | [A01](stream_a/tasks/A01-train.md) |
| A02 | Terrain grids | 1 | n/a | | 1 h | [A02](stream_a/tasks/A02-terrain.md) |
| A03 | Warp + tile library | 1 | n/a | | 2 to 3 h | [A03](stream_a/tasks/A03-warp-tiles.md) |
| A08 | Cost map tests | 1 | n/a | | 1 h | [A08](stream_a/tasks/A08-mapping.md) |
| A04 | World builder (**hand to B**) | 2 | A02, A03 | | 1.5 h | [A04](stream_a/tasks/A04-world-builder.md) |
| A07a | observe() model mode, cached | 2 | A03 (+ fake preds until A06) | | 2 h | [A07](stream_a/tasks/A07-perception-model-mode.md) |
| A05 | Segmentation metrics | 3 | A01 | yes | 45 min | [A05](stream_a/tasks/A05-evaluate.md) |
| A06 | Precompute preds + logits | 3 | A01, A03 | yes | 30 min | [A06](stream_a/tasks/A06-precompute.md) |
| A07b | Live inference for blur | 3 | A01, A07a | | 1 h | [A07](stream_a/tasks/A07-perception-model-mode.md) |
| A11a | Frames + overlays | 3 | A01 | | 45 min | [A11](stream_a/tasks/A11-frames-overlays-card.md) |
| A10 | Calibration (P1) | 3/4 | A06 | | 45 min | [A10](stream_a/tasks/A10-calibration.md) |
| A09 | Eval harness | 4 | B's `Rover`, A04, A07 | | 3 to 4 h | [A09](stream_a/tasks/A09-harness.md) |
| A11b | Decision card data | 4 | B's planner, A07, A08 | | 1 h | [A11](stream_a/tasks/A11-frames-overlays-card.md) |
| A12 | 3D view (P1) | 4 | B's `/api/world` | | 2 h | [A12](stream_a/tasks/A12-terrain-3d.md) |

**Max useful parallelism:** 4 subagents in Wave 1, 2 to 3 afterwards. More than that just adds merge noise.

**Spawning a subagent:** give it (1) the brief file contents, (2) its worktree path, (3) the `PY` and `RR_ARTIFACTS` values, (4) "Follow AGENTS.md; edit only the files your brief lists under 'You own'; finish with the push sequence in STREAM_A.md section 3; reply with the brief's 'Hand back' items."

## 5. Human checkpoints (Pranoy, not agents)

| When | What you do |
|---|---|
| Hour 0 | 20-min contract meeting with the B teammate. Start the orchestrator on Wave 1 |
| After A03 | **Look at the warp contact sheets.** If ground doesn't look like ground from above, the whole world is wrong |
| After A04 (~h6) | Tell B: swap `build_stub_world` for `world.builder.build_world`. Pick the demo seed together |
| A07 contract change | Approve the `Belief.weight` field with B |
| After A05 | Read the metrics. Big-rock IoU < 0.15 → have the orchestrator relaunch A01 with a higher weight, in the background |
| **Hour 12, gate G3** | With B: switch to `mode="model"` and watch the route change because of model output. **Nothing else matters until this works** |
| Hour 16 | P0 freeze: no new features until every P0 works end to end |
| After A09 | Sanity-check the results table before it goes on a slide |
| Hours 21 to 24 | Slides (data sources, architecture, metrics, results, honesty notes), rehearse with B 3× |

## 6. Hour plan (wall clock)

| Hours | Orchestrator runs | Milestone |
|---|---|---|
| 0 to 1 | A01, A02, A03, A08 in parallel | Training running, loss logged |
| 1 to 3 | A02 and A08 land; A03 continues | `world/terrain.npz` committed |
| 3 to 6 | A04, A07a (fake preds) | **h6: `build_world` handed to B** |
| 6 to 9 | A05 → A06 (GPU, in sequence), A07b, A11a | Metrics + `preds20.npz` committed |
| 9 to 12 | Finish A07; integration fixes | **h12: G3, route changes on model output** |
| 12 to 16 | A09 (once B's Rover lands), A10 | `eval/results.md` |
| 16 to 19 | A11b, A10 rerun of A06 with T | Decision card has real numbers |
| 19 to 21 | A12 (cut if behind) | 3D view |
| 21 to 24 | Human: slides, rehearsal | Demo locked |

**Cut order if behind:** A12 (3D), then A10 (calibration). Never cut A07/G3 or A09.

## 7. Interfaces Stream A promises Stream B

```python
world.builder.build_world(seed) -> World                                    # h6, drop-in for build_stub_world
backend.app.perception.observe(world, belief, pose, sensor, flags, mode)    # oracle now; "model" at h12
backend.app.mapping.cost_grid(world, belief, cfg) -> (cost >= 1 float32, blocked bool)   # now
backend.app.mapping.explain_cell(world, belief, cfg, cell) -> dict          # A08
eval.decision_card.decision_card(world, belief, cfg, start, goal, path) -> dict  # A11b
files: eval/results.json, eval/seg_metrics.json, terrain/frames/<id>.jpg, terrain/overlays/<id>.png
```
And what Stream A needs from B: `planner.plan()` (around h4) and a headless `Rover(...).run_to_completion(...)` (around h12) for A09.

## 8. Facts already verified (don't rediscover them)

- **DTM crop** `terrain/jezero_crop.tif`: 18.43910°N 77.40880°E, 58 × 58 at 1 m with 4 m padding. Plateau with median slope 4.5°, delta-front scarp on the east edge (about 11% over 20°), 16 m relief. The landing site was rejected as too flat.
- **AI4Mars** (`hassanjbara/AI4MARS`): 1024 × 1024 uint8; columns `image, label_mask, rover_mask, range_mask, has_masks, has_labels`. Some train rows have no labels (filter `has_labels`). No id column, so ids are `m3_<row>`. `rover_mask==1` is the rover body at the bottom (18 to 44%); `range_mask==1` is beyond 30 m at the top. 73 to 85% of each test image is 255.
- **Light delay** today: 10.9 min one-way (`terrain/light_delay.json`).
- **GPU:** RTX 5060 Laptop, 8 GB, torch 2.11 + cu128 working.

## 9. Honesty notes (go on a slide)

- The AI4Mars images are Curiosity NAVCAM; the simulated rover is Perseverance at Jezero.
- The top-down tiles come from a flat-ground homography with no camera intrinsics.
- The geometric obstacles from the fault panel are simulated.
- The world stitches unrelated images into tiles, so tile edges are discontinuous by construction.
