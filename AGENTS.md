# Agent instructions (RedRoute)

These apply to every coding agent in this repo (Codex, Claude, subagents).

## Git
- Work on `main` only. **Never create or push branches.** Use detached worktrees for parallel work and push with `git push origin HEAD:main`.
- **No AI attribution:** no `Co-Authored-By`, "Generated with", or similar lines in commits or PRs.
- `git fetch origin && git rebase origin/main` before pushing; tests must pass first.
- Commit only files you own (see below). Never `git add -A` in a shared checkout.

## Ownership
- Stream A (perception, world, eval): `ml/`, `world/`, `eval/`, `backend/app/perception.py`, `backend/app/mapping.py`, `frontend/src/Terrain3D.tsx`. Plan: `docs/STREAM_A.md`, task briefs in `docs/stream_a/tasks/`.
- Stream B (autonomy, reliability, UI): `backend/app/{planner,rover,link,faults,api}.py`, `frontend/` (except Terrain3D). Plan: `docs/STREAM_B.md`.
- **`backend/app/contracts.py` is shared.** Don't change it without both humans agreeing.

## Environment
- Python: `.venv/Scripts/python` in the main checkout (worktrees use its absolute path: `C:/Pranoy/Machacks/RedRoute-Mars-Rover-Automation-/.venv/Scripts/python.exe`).
- Big or regenerable outputs go under `$RR_ARTIFACTS` (see `world/paths.py`); never hardcode paths.
- Tests: `python -m pytest -q` from the repo root. Frontend: `cd frontend && npx tsc -b`.

## Conventions
- Grid cells are `(row, col)`; row 0 is north; heading 0 = north, clockwise. 200 × 200 cells at 0.25 m.
- AI4Mars classes: 0 soil, 1 bedrock, 2 sand, 3 big rock, 255 ignore. Tile ids are `m3_<row index in test_min3>`.
- The rover and planner must never read `World.truth`. Only the simulator's hazard check, the eval harness and the clearly labelled ORACLE perception mode may.
- Report measured numbers only.
