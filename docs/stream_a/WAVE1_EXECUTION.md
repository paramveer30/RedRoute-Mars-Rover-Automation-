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

Pending task completion.

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
