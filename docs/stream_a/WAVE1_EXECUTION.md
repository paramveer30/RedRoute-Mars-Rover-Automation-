# Wave 1 execution record

Scope: launch and complete A01, A02, A03, and A08 using the unchanged task
briefs and the lifecycle in `docs/STREAM_A.md` section 3. Stop downstream
work at the human checkpoints in section 5.

## Execution plan

1. Read `AGENTS.md`, the full Stream A playbook, and all four task briefs.
   Verify a clean `main`, fetch `origin/main`, and run the baseline tests.
2. Create detached worktrees at `C:/Pranoy/Machacks/wt/A01`, `A02`, `A03`,
   and `A08`. Use the main checkout's Python environment and
   `RR_ARTIFACTS=C:/Pranoy/Machacks/rr-artifacts` in every task process.
3. Assign each task to a separate subagent. Start A01, A02, and A03 first;
   start A08 when a slot opens. The session permits three active subagents
   alongside the orchestrator. A01 has exclusive use of the GPU.
4. Each task edits only its brief's owned files, generates its actual
   artifacts, runs its acceptance checks, and commits incrementally.
   Publish A01's model interface and A03's warp interface early.
5. Before each push, fetch, rebase onto `origin/main`, and run the full
   Python suite. Push only `HEAD:main`; retry this sequence after a race.
   Never create branches, stage unrelated files, or edit `contracts.py`.
6. Report each task's measured results and commits when it finishes.
   Inspect the terrain plot and warp contact sheets. Request Pranoy's
   contact-sheet review after A03; do not start A04 or A07 until that
   checkpoint has been resolved.
7. Remove only clean, finished worktrees. Update the main checkout, run
   the integrated tests, and record completed results and outstanding
   work here. Retain any worktree with active training or unfinished work.

## Baseline

- Starting commit: `5a7131d`.
- Main checkout initially clean; all four worktrees created detached.
- Baseline Python suite: 4 passed in 0.61 seconds.

## Results

| Task | Status | Published commits |
| --- | --- | --- |
| A01 | Complete; trained weights published | `18b581b`, `a64cf3d`, `7d05496` |
| A02 | Complete | `e4b5ae6` |
| A03 | Complete; human visual review approved | `134e9ff`, `4318808` |
| A08 | Complete | `9a22dc8` |

### A01: training

- GPU smoke run completed in 58.03 seconds, with a checkpoint, CSV log,
  49,007,035-byte fp16 export, and successful offline CPU fp32 inference.
- Smoke used 55 training rows and 7 validation rows. These are smoke
  checks, not evidence of final model quality.
- Default full training started with 20 requested epochs, batch 16,
  input size 256, and the fixed 500-row validation split. No test set is
  used for training or checkpoint selection.
- Full training completed 20 epochs in 1,400.396 seconds (23.34 minutes).
  Training loss fell from 0.855600 to 0.117096.
- Selected checkpoint: epoch 16, validation mIoU 0.795561; per-class IoU
  soil 0.923963, bedrock 0.937332, sand 0.874320, big rock 0.446628.
  These use 256 px validation labels; A05's native-resolution evaluation
  has a different measurement procedure.
- Class weights: [1.651625, 1.406552, 2.897715, 10.625541].
- Deployable weights: 49,007,035 bytes. Final offline fp32 CPU execution
  passed; weights were pushed after rebasing and 44 passing tests.
- Training process PID `43164` exited successfully; GPU released.
- Logs: `C:/Pranoy/Machacks/rr-artifacts/runs/train/stdout.log`,
  `stderr.log`, `log.csv`, and `summary.json`.
- Completed A01 worktree removed after the final weights push.

### A02: real terrain

- Elevation range: -2489.841553 to -2477.150146 m.
- Slope median: 3.854343 degrees; p95: 29.208776 degrees;
  share above 20 degrees: 10.9625%.
- Roughness standard deviation: 0.180279.
- Task and orchestrator inspected the plot; the eastern scarp is visible.
- Plot: `C:/Pranoy/Machacks/rr-artifacts/runs/terrain/terrain.png`.
- Full suite after task rebase: 10 passed.

### A03: warp and tile library

- All 322 source ids, homographies, and label grids are in `world/tiles.npz`;
  all 322 image patches are under the shared artifact directory.
- 149 usable tiles at the unchanged labelled-fraction threshold of 0.6.
- Class cell counts across usable tiles: soil 28,257; bedrock 8,997;
  sand 15,120; big rock 8.
- Usable tiles containing each class: soil 79; bedrock 39; sand 51;
  big rock 1. Big-rock diversity is very sparse.
- Clear arm/wheel failures `m3_0000`, `m3_0275`, and `m3_0020` fail the
  usable threshold. Some masked rover-edge intrusion remains.
- Task and orchestrator inspected all three review sheets. This does not
  replace the required human review.
- Review sheets under `C:/Pranoy/Machacks/rr-artifacts/runs/warp/`:
  `contact_random_01.png`, `contact_random_02.png`, `contact_edge_cases.png`.
- The warp remains a flat-ground approximation without camera intrinsics
  or calibrated metric scale, as required by the honesty notes.
- Full suite after final task rebase: 44 passed.

### A08: mapping

- Added a defensive cost floor, cached inflation footprints, faster
  semantic blob filtering, and JSON-serializable `explain_cell()` output.
- 18 mapping checks cover all brief requirements, including the hidden
  truth/unsensed-geometry boundary and the 113-cell inflation footprint.
- Measured 200 x 200 runtime: median 3.749 ms, p95 6.110 ms over 25
  calls after 5 warmups.
- Full suite after task rebase: 43 passed.
- No shared contract changes.

## Human checkpoints and follow-on work

- After A03: inspect the warp contact sheets and confirm that the ground
  and labels look plausible before downstream world construction.
- After A04: tell Stream B to use `world.builder.build_world`; agree on a
  demo seed together. A04 is outside this Wave 1 launch.
- Before A07's proposed `Belief.weight` change: obtain agreement from both
  humans. This execution does not authorize any contract change.
- After A05: review measured metrics; big-rock IoU below 0.15 triggers a
  human-reviewed retraining decision. A05 follows completed A01 weights.
- Gate G3: with Stream B, switch to model mode and verify that predictions
  change the route. Later checkpoints remain as listed in section 5.

## Resumed execution after warp approval

The human explicitly approved the warps and requested continuation of the
plan. A04 and A07 were launched in separate detached worktrees. A05 follows
completed A01 on the GPU; A06 waits for A05 to release it. The remaining
sequence is A07 live inference, A11 frame assets, and optional A10
calibration, with the section 5 checkpoints retained.

The shared contract remains unchanged unless both humans approve a change.
A07 can maintain internal fusion weights without modifying `Belief`.

At resumption, Stream B's `planner.plan()` still raises
`NotImplementedError`, and `Rover` is not implemented. Actual G3 route-change
verification and A09 mission results depend on those implementations. A09
metric code and synthetic tests may be prepared early as its brief permits;
synthetic results must never be presented as measured rover performance.
