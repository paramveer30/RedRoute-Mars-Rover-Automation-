# Stream B: Autonomy, Reliability, and Mission Control

**You own:** everything that moves the rover, keeps it safe, and shows it to the operator.

**Your files:** `backend/app/planner.py`, `rover.py`, `link.py`, `faults.py`, `api.py`, `backend/tests/`, and `frontend/` (except A's 3D view and vision-panel components).

**Do not edit without Stream A:** `backend/app/contracts.py`. A owns `perception.py`, `mapping.py`, `world/`, `ml/`, `eval/`.

**Branch:** `stream-b`. Merge to `main` at every sync point (hours 6, 12, 16, 20).

---

## What already exists (from setup)

| File | State |
|---|---|
| `backend/app/contracts.py` | Shared dataclasses and grid constants. Read them first |
| `backend/app/api.py` | FastAPI app with `/api/health`, `/api/world/{seed}`, and a WebSocket **hello tick** that drives a straight line on the stub world using oracle perception |
| `backend/app/perception.py` | `observe()` oracle mode works now: use it |
| `backend/app/mapping.py` | `cost_grid()` works now (A tunes it later): use it |
| `world/stub_world.py` | `build_stub_world(seed)`: your world until A's real one lands at hour 6 |
| `planner.py`, `rover.py`, `link.py`, `faults.py` | Stubs with docstrings: yours to write |
| `frontend/` | Vite + React + TS + Tailwind v4 + zustand + three/R3F. `App.tsx` shows the ORACLE banner, connection status, a placeholder canvas map and a telemetry list. `/api` and `/ws` are proxied to :8000 |
| `scripts/fetch_light_delay.py` | Fetches the real light delay from JPL Horizons into `terrain/light_delay.json` |

**Run it:**
```bash
# terminal 1 (repo root)
.venv/Scripts/python -m uvicorn backend.app.api:app --reload --port 8000
# terminal 2
cd frontend && npm run dev        # open http://localhost:5173
# tests
.venv/Scripts/python -m pytest
```

---

## Hour-by-hour

### Hour 0 to 1: contract and hello tick
- [ ] 30 min contract meeting with A. Lock `contracts.py`, the WebSocket schema (`frontend/src/types.ts` mirrors it), and branch names.
- [ ] Run both servers and confirm the browser shows ticks and a moving yellow dot. **Done:** this already works from setup, so move straight on.

### Hour 1 to 4: A*, simulator, state machine
- [ ] `planner.py::plan(cost, blocked, start, goal)`:
  - 8-connected, `heapq`. **Units are metres:** step cost = `0.5*(cost[a]+cost[b]) * L`, with L = 0.25 for straight moves and 0.25·√2 for diagonals. The heuristic is the Euclidean distance in metres. Because every cell costs at least 1, the heuristic is admissible and the path optimal.
  - No corner-cutting: a diagonal move needs both orthogonal neighbours unblocked.
  - Return `None` if the start or goal is blocked or unreachable.
  - Should run in well under 100 ms on 200 × 200 (use NumPy arrays, a closed-set bool array and a g-score float array).
- [ ] `backend/tests/test_planner.py`: straight line on uniform cost, wall with a gap, no path → None, prefers the cheap detour over the expensive straight line, path never touches blocked cells, diagonal corner rule.
- [ ] `rover.py`: a `Rover` class that knows nothing about FastAPI (A's eval harness drives it headlessly):
  ```python
  Rover(world, mode="oracle", cost_cfg=CostConfig(), sensor=SensorConfig())
  .set_target(goal) ; .step() -> list[Event] ; .snapshot() -> dict (tick payload)
  .run_to_completion(start, goal, max_steps) -> MissionResult   # for A's harness
  ```
  - Fixed tick. Each DRIVING tick moves 1 m (4 cells) along the path.
  - States IDLE → PLANNING → DRIVING ↔ SCANNING (every 2 m or on entering a new tile) → … → IDLE on arrival (within 1 m of the goal). REPLANNING and SAFE_STOP as below.
  - Track battery as a simple linear drain (no energy model), odometer, distance since Earth contact.
- **Done:** the rover drives a planned path on the stub world in the UI.

### Hour 4 to 7: 2D mission map
- [ ] Canvas layers, each toggleable:
  1. Elevation hillshade (fetch `/api/world/{seed}` once)
  2. Belief classes (soil tan, bedrock grey, sand yellow, big rock red, unknown dark)
  3. Uncertainty heatmap
  4. Blocked and inflated cells (hatched)
  5. Path (current route_version)
  6. Rover with a heading arrow and the sensor cone
  7. Debug-only toggle: truth layer (label it TRUTH, operator never sees it by default)
- [ ] Sending the full belief every tick is 160k floats, which is too heavy. Send `belief_delta` per tile: for each changed tile, the 20 × 20 argmax class and a 20 × 20 uncertainty (uint8 0..255) as base64 or small arrays. The frontend keeps a full 200 × 200 buffer.
- [ ] Click on the map to choose a target. Event feed panel. Telemetry panel (state, battery, odometer, mission time, link, perception mode).
- [ ] **Hour 6:** A hands over `build_world(seed)`. Swap one import, and the map now shows real Jezero.

### Hour 7 to 10: replanning, safety, logging
- [ ] Replanning trigger, checked every tick: any remaining path cell is now `blocked`, OR the remaining-path cost rose by more than 50% since the plan was made. Bump `route_version` on every new plan.
- [ ] Observation watchdog: newest successful observation older than `T_stale` (5 s of sim time) → SAFE_STOP.
- [ ] No route → SAFE_STOP. Autonomy limit: more than 30 m driven without Earth contact → SAFE_STOP.
- [ ] Model offline → `mode="geometry_only"`, half speed, SAFE_STOP if the geometry can't clear the path.
- [ ] SAFE_STOP only exits on an operator command.
- [ ] SQLite (`sqlite3` stdlib, file gitignored): tables `mission`, `telemetry`, `route` (versioned), `event`.
- [ ] `test_rover.py`: one test per state transition.

### Hour 10 to 12: comms link
- [ ] `link.py`: an uplink and a downlink queue, each message stamped `{mission_time, seq, route_version}`, and released after `delay_s / compression`.
  - **Time compression:** the real delay (10.9 min today, cached) ÷ factor (for example 60) gives about 11 s on screen. **The UI shows both** the real and compressed delay.
  - **Blackout:** downlink messages buffer onboard; uplink is dropped or queued. The rover keeps driving within the autonomy boundary. On restore, flush the buffer to the ground feed **sorted by mission_time**, with each late arrival tagged `late: true`.
  - **Stale command:** every command carries a precondition `{expected_state, route_version}`. On arrival, if `route_version` doesn't match the rover's current one, reject with an event like `"Rejected: route v3 superseded by v5"`.
- [ ] Ground clock vs rover clock both on screen. They diverge during a blackout.
- **Done:** the blackout demo works end to end.

### Hour 12: INTEGRATION 1, gate G3 (with A)
- [ ] A switches perception to `"model"`. The ORACLE banner disappears automatically (it's driven by `perception_mode`).
- [ ] Watch a demo seed together: the route changes because of model output.
- [ ] Agree on the `Rover.run_to_completion` signature for A's harness.

### Hour 12 to 15: fault panel (all seven)
`POST /api/missions/{id}/faults {"type": ...}`

| Fault | How to implement | Expected |
|---|---|---|
| `hidden_rock` | Set `world.geo_obstacle` blob on the path a few metres ahead | Detected in range → blocked → replan |
| `drop_frame` | `ObserveFlags(drop_frame=True)` for N ticks | Keep the last map; watchdog → SAFE_STOP |
| `corrupt_frame` | `ObserveFlags(corrupt_frame=True)` once | Red log line, frame rejected, rescan |
| `low_confidence` | `ObserveFlags(blur=0.8)` for the next frames | U rises, detour or rescan |
| `model_offline` | `mode="geometry_only"` | Banner, reduced speed, SAFE_STOP if needed |
| `blackout` | Link status BLACKOUT for N s | Clocks diverge, buffer, ordered replay |
| `stale_command` | Send a command with an old `route_version` through the delayed link | Rejected with reason |

- [ ] `test_faults.py`: each fault produces its expected state or event.

### Hour 15 to 17: camera panel and security (P1)
- [ ] Camera panel: show the real NAVCAM frame for `frame_image_id`, with A's overlay PNG and an opacity slider.
- [ ] HMAC-SHA256 signed commands: `hmac(key, json(cmd) + seq)`; reject a bad signature or a replayed or old `seq`, and log it. Roles: an `operator` token can command, `viewer` is read-only. Include a "tamper" button to demo a rejected command.

### Hour 16: P0 freeze

### Hour 17 to 20: P1 extras
- [ ] Horizons light delay: read `terrain/light_delay.json` (already cached from setup), try a live refresh on startup, fall back to the cache. Show "real delay: 10.9 min (JPL Horizons, 2026-11-15)".
- [ ] Three rovers at once: one `Rover` per mission in an asyncio task, three different seeds, a selector in the UI. (Cut first if behind.)
- [ ] `GET /api/eval/results` serves A's `eval/results.json`. Draw the route decision card from A's data.
- [ ] UI polish.

### Hour 20 to 22: harden and record
- [ ] Run the full demo script 5 times and fix whatever breaks. Add a "reset mission" button.
- [ ] **Record the backup video** of the whole 3-minute script.

### Hour 22 to 24: rehearse with A
Script under 3 minutes, no slides during the live part.

---

## Cut order if behind
**Multi-rover**, then **Horizons live** (keep the cached value), then A's 3D view and calibration. Never cut G3 or the fault panel.

## Interfaces you promise to A
```python
plan(cost, blocked, start, goal) -> list[Cell] | None                    # hour 4
Rover(world, mode, cost_cfg, sensor).run_to_completion(start, goal, max_steps) -> MissionResult  # hour 12
# MissionResult: reached, hazard_entries, path_len_m, steps, replans, replan_latencies_ms, final_state
```

## WebSocket tick (owned by you; keep `frontend/src/types.ts` in sync)
```json
{ "type": "tick", "mission_time": 182.5, "seq": 913, "rover_id": "rover-01",
  "pose": {"x": 51, "y": 33, "heading": 45}, "state": "DRIVING", "battery": 84,
  "route_version": 5, "path": [[51,33],[52,34]], "belief_delta": {"tiles": ["t_3_4"]},
  "perception_mode": "model", "link": {"status": "CONNECTED", "delay_s": 9.6}, "events": [] }
```
`pose.x` is the column and `pose.y` is the row (grid cells). Note `path` in the hello tick is empty until the planner exists.

## Gotchas
- The backend is the single source of truth. The frontend only draws what it's told.
- The rover/planner must **never read `world.truth`**. Only the simulator's hazard check uses it.
- Keep `Rover` free of FastAPI and asyncio, otherwise A's parallel harness can't use it.
