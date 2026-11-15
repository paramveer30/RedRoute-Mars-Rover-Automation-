"""Rover interface the eval harness codes against, plus a TEST-ONLY fake rover.

Agreed with Stream B (docs/stream_a/FINISH_PLAN.md section 4)::

    Rover(world, mode, cost_cfg, sensor).run_to_completion(start, goal, max_steps) -> MissionResult

``FakeRover`` exists only so the harness and its tests can run before Stream
B's ``backend.app.rover.Rover`` lands. It is a test double: its numbers are
NOT RedRoute performance and must never be reported as such. It behaves like
a rover is required to (observe -> belief cost grid -> plan -> drive 1 m) and
never reads ``World.truth`` itself; only ``observe(mode="oracle")``, the
labelled ORACLE perception path, touches truth.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Callable, Literal, Protocol

import numpy as np
from scipy.sparse.csgraph import dijkstra

from backend.app.contracts import (Belief, Cell, CostConfig, ObserveFlags, Pose,
                                   SensorConfig, World)
from backend.app.mapping import cost_grid
from backend.app.perception import observe
from eval.metrics import GOAL_TOL_M, grid_graph, path_length_m

Mode = Literal["geometry_only", "model", "oracle"]
MODES: tuple[str, ...] = ("geometry_only", "model", "oracle")


@dataclass
class MissionResult:
    reached: bool
    final_state: str
    trajectory: list[Cell]               # EVERY traversed cell, start first
    path_len_m: float
    steps: int
    replans: int
    replan_latencies_ms: list[float] = field(default_factory=list)
    observe_calls: int | None = 0        # None: the rover does not report it


REQUIRED_FIELDS = ("reached", "final_state", "trajectory", "path_len_m", "steps",
                   "replans", "replan_latencies_ms")
# Agreed but not yet in Stream B's MissionResult (rover.py at fb2d60a); recorded as None.
OPTIONAL_FIELDS = ("observe_calls",)


class RoverLike(Protocol):
    def run_to_completion(self, start: Cell, goal: Cell, max_steps: int) -> MissionResult: ...


# Rover(world, mode, cost_cfg, sensor)
RoverFactory = Callable[[World, str, CostConfig, SensorConfig], RoverLike]


def as_mission_result(obj) -> MissionResult:
    """Normalise whatever the rover returned into ``MissionResult``.

    Fails loudly, naming missing fields, rather than guessing defaults.
    """
    if isinstance(obj, MissionResult):
        return obj
    missing = [name for name in REQUIRED_FIELDS if not hasattr(obj, name)]
    if missing:
        raise TypeError(f"rover result {type(obj).__name__} lacks agreed fields: {missing}")
    return MissionResult(
        reached=bool(obj.reached), final_state=str(getattr(obj.final_state, "value", obj.final_state)),
        trajectory=[(int(r), int(c)) for r, c in obj.trajectory],
        path_len_m=float(obj.path_len_m), steps=int(obj.steps), replans=int(obj.replans),
        replan_latencies_ms=[float(x) for x in obj.replan_latencies_ms],
        observe_calls=(int(obj.observe_calls) if getattr(obj, "observe_calls", None) is not None
                       else None))


# ---- TEST DOUBLE ---------------------------------------------------------------
def _fake_plan(cost: np.ndarray, blocked: np.ndarray, start: Cell, goal: Cell) -> list[Cell] | None:
    """Test-only Dijkstra on the belief cost grid (stand-in for Stream B's A*)."""
    blocked = blocked.copy()
    blocked[start] = False
    graph = grid_graph(blocked, cost)
    m = cost.shape[1]
    src, dst = start[0] * m + start[1], goal[0] * m + goal[1]
    dist, pred = dijkstra(graph, directed=False, indices=src, return_predecessors=True)
    if not np.isfinite(dist[dst]):
        return None
    out, node = [], dst
    while node != src:
        out.append((node // m, node % m))
        node = pred[node]
    out.append(tuple(start))
    return [(int(r), int(c)) for r, c in out[::-1]]


class FakeRover:
    """TEST DOUBLE ONLY. Not Stream B's rover; its results are not performance.

    Loop per tick: observe (perception ``mode``) -> ``mapping.cost_grid`` on the
    belief -> test-only Dijkstra -> drive up to ``cells_per_tick`` cells. If the
    belief blocks every route (e.g. the goal sits inside 1.5 m inflation), it
    retries with blocked cells turned into a large cost instead of stopping, so
    the harness always gets a trajectory to score.
    """

    perception_for_mode = {"geometry_only": "geometry_only", "model": "model", "oracle": "oracle"}

    def __init__(self, world: World, mode: str, cost_cfg: CostConfig | None = None,
                 sensor: SensorConfig | None = None, cells_per_tick: int = 4):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.world = world
        self.mode = mode
        self.cfg = cost_cfg or CostConfig()
        self.sensor = sensor or SensorConfig()
        self.cells_per_tick = cells_per_tick

    def _plan(self, belief: Belief, cur: Cell, goal: Cell) -> list[Cell] | None:
        cost, blocked = cost_grid(self.world, belief, self.cfg)
        path = _fake_plan(cost, blocked, cur, goal)
        if path is None:
            path = _fake_plan(np.where(blocked, cost + 1000.0, cost).astype(np.float32),
                              np.zeros_like(blocked), cur, goal)
        return path

    def run_to_completion(self, start: Cell, goal: Cell, max_steps: int = 300) -> MissionResult:
        start, goal = (int(start[0]), int(start[1])), (int(goal[0]), int(goal[1]))
        heading = math.degrees(math.atan2(goal[1] - start[1], -(goal[0] - start[0]))) % 360
        pose = Pose(float(start[0]), float(start[1]), heading)
        belief = Belief.empty()
        trajectory = [start]
        latencies: list[float] = []
        replans = observe_calls = steps = 0
        remaining: list[Cell] | None = None
        state = "MAX_STEPS"
        perception = self.perception_for_mode[self.mode]

        def at_goal(cell: Cell) -> bool:
            return math.hypot(cell[0] - goal[0], cell[1] - goal[1]) * 0.25 <= GOAL_TOL_M

        while steps < max_steps:
            cur = trajectory[-1]
            if at_goal(cur):
                state = "IDLE"
                break
            observe(self.world, belief, pose, self.sensor, ObserveFlags(), perception)
            observe_calls += 1
            t0 = time.perf_counter()
            path = self._plan(belief, cur, goal)
            latencies.append((time.perf_counter() - t0) * 1000.0)
            if path is None:
                state = "SAFE_STOP"
                break
            if remaining is not None and path[:len(remaining)] != remaining[:len(path)]:
                replans += 1
            for cell in path[1:1 + self.cells_per_tick]:
                trajectory.append(cell)
            prev, cur = trajectory[-2], trajectory[-1]
            pose = Pose(float(cur[0]), float(cur[1]),
                        math.degrees(math.atan2(cur[1] - prev[1], -(cur[0] - prev[0]))) % 360)
            remaining = path[self.cells_per_tick:]
            steps += 1
        else:
            if at_goal(trajectory[-1]):
                state = "IDLE"
        return MissionResult(
            reached=state == "IDLE", final_state=state, trajectory=trajectory,
            path_len_m=path_length_m(trajectory), steps=steps, replans=replans,
            replan_latencies_ms=latencies, observe_calls=observe_calls)
