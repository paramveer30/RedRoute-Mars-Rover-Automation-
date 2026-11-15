# A09: Evaluation harness and baselines

**Wave 4, after gate G3.** It needs Stream B's headless `Rover.run_to_completion` (promised around hour 12). Start the metric code and tests earlier against a fake `MissionResult`. About 3 to 4 hours.

## Goal
Measured evidence for the main claim: ML-aware planning enters fewer hazards than geometry-only, at a modest route-length cost, and gets close to the oracle. It also produces the scalability numbers.

## You own
- `eval/harness.py`, `eval/metrics.py`, `eval/report.py`
- `eval/results.json`, `eval/results.md` (committed)
- `backend/tests/test_eval_metrics.py`

## Inputs
- Stream B (agreed interface; check `backend/app/rover.py` for the actual names):
  ```python
  Rover(world, mode, cost_cfg, sensor).run_to_completion(start, goal, max_steps) -> MissionResult
  # MissionResult: reached, hazard_entries, path_len_m, steps, replans, replan_latencies_ms, final_state, trajectory
  ```
- `world/builder.build_world`, `backend/app/contracts.CostConfig`, `backend/app/planner.plan` (to compute the shortest feasible path on truth for the route ratio)

## Spec
Configurations:

| name | mode | CostConfig |
|---|---|---|
| geometry_only | `geometry_only` | `use_semantics=False` |
| redroute_no_u | `model` | `w_u=0` |
| redroute | `model` | default |
| oracle | `oracle` | default |

- Scenarios: seeds 0 to N-1 (default N = 100). Per seed, sample start and goal at least 30 m apart on non-blocked truth cells (seeded), **identical across configs**.
- **Hazard entry** (computed here from the trajectory, so it's consistent whatever the rover reports): a step where the rover footprint (radius 0.5 m = 2 cells) overlaps truth sand, truth big rock, or slope above the limit. Truth 255 is not a hazard. Count the entries (contiguous runs count once).
- **Success** = reached within 1 m AND 0 hazard entries.
- **Route length ratio** = path_len_m / shortest feasible path on the *truth* cost map (oracle `cost_grid` + `plan`).
- Parallelism: `multiprocessing.Pool(processes=os.cpu_count()-1)` with **`if __name__ == "__main__":`** (Windows uses spawn). Use the pool `initializer` to load `preds20.npz`, the terrain and the tiles once per worker. Workers never import FastAPI.
- Throughput: wall time, missions/min, observe() calls/s.
- `results.json`: per config `{success_rate, hazard_entries_mean, route_ratio_mean, replan_latency_ms_p50/p95, n}` + throughput + git sha + seeds.
- `results.md`: the slide table (PRD section 8 format), with 95% bootstrap CIs on success rate and hazard entries.
- CLI: `--n 100 --workers 8 --configs all`.

## Tests
Metric functions on synthetic trajectories: hazard counting (contiguous runs count once, footprint overlap), success logic, route ratio = 1 for the optimal path.

## Acceptance
- [ ] `$PY -m eval.harness --n 20` runs end to end in a few minutes
- [ ] Full run (N ≥ 100) writes results.json and results.md
- [ ] The numbers are what the harness measured. If RedRoute doesn't beat geometry-only, report that, then investigate cost weights with the orchestrator. Never edit the numbers.

## Hand back
The results table, throughput, and any surprises.
