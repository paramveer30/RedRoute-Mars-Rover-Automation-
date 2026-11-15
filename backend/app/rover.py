"""Rover simulator + safety state machine (Stream B owns). See docs/STREAM_B.md.

States: IDLE, SCANNING, PLANNING, DRIVING, REPLANNING, SAFE_STOP (PRD 7.5).

The Rover knows nothing about FastAPI or asyncio: A's eval harness drives it headlessly via
``run_to_completion``; the API drives it one ``step()`` per tick.

Only the simulator's hazard check (``_hazard_at``) reads ``world.truth``. Planning and
replanning only see the belief through ``cost_grid``.
"""
from __future__ import annotations

import base64
import math
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any

import numpy as np

from .contracts import (BIG_ROCK, GRID, RES_M, SAND, TILE, Belief, Cell, CostConfig,
                        ObserveFlags, PerceptionMode, Pose, SensorConfig, World,
                        cell_to_tile, tile_id)
from .mapping import cost_grid
from .perception import observe
from .planner import plan

TICK_S = 0.5                 # sim seconds per step()
DRIVE_M_PER_TICK = 1.0       # 1 m (4 cells) per DRIVING tick
SCAN_EVERY_M = 2.0
ARRIVE_M = 1.0
T_STALE_S = 5.0              # observation watchdog
AUTONOMY_LIMIT_M = 30.0      # max driving without Earth contact
REPLAN_COST_RISE = 1.5       # remaining-path cost rose by more than 50%
CHEAPER_ROUTE = 0.9          # after a scan, switch if a new route is >= 10% cheaper
BATTERY_PER_M = 0.2          # % per metre driven (linear, no energy model)
BATTERY_PER_TICK = 0.005     # % idle drain per tick
FOOTPRINT_CELLS = 2          # rover radius 0.5 m, for the simulator hazard check
START_CLEAR_CELLS = 3        # let the rover plan out of its own inflation buffer


class State(str, Enum):
    IDLE = "IDLE"
    SCANNING = "SCANNING"
    PLANNING = "PLANNING"
    DRIVING = "DRIVING"
    REPLANNING = "REPLANNING"
    SAFE_STOP = "SAFE_STOP"


@dataclass
class Event:
    mission_time: float
    level: str        # "info" | "warn" | "error"
    kind: str         # machine-readable, e.g. "replan", "safe_stop", "frame_rejected"
    msg: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"mission_time": round(self.mission_time, 2), "level": self.level,
                "kind": self.kind, "msg": self.msg, **({"data": self.data} if self.data else {})}


@dataclass
class MissionResult:
    reached: bool
    hazard_entries: int
    path_len_m: float
    steps: int
    replans: int
    replan_latencies_ms: list[float]
    final_state: str
    trajectory: list[Cell]


def _disk(radius: int) -> np.ndarray:
    yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return (xx * xx + yy * yy) <= radius * radius


_FOOTPRINT = _disk(FOOTPRINT_CELLS)
_START_CLEAR = _disk(START_CLEAR_CELLS)


def _stamp(mask: np.ndarray, center: Cell, disk: np.ndarray, value: bool) -> None:
    r = disk.shape[0] // 2
    r0, c0 = center
    rs, re = max(r0 - r, 0), min(r0 + r + 1, mask.shape[0])
    cs, ce = max(c0 - r, 0), min(c0 + r + 1, mask.shape[1])
    sub = disk[rs - (r0 - r):re - (r0 - r), cs - (c0 - r):ce - (c0 - r)]
    mask[rs:re, cs:ce][sub] = value


def _step_len(a: Cell, b: Cell) -> float:
    return RES_M * (math.sqrt(2) if a[0] != b[0] and a[1] != b[1] else 1.0)


def _heading(a: Cell, b: Cell) -> float:
    dr, dc = b[0] - a[0], b[1] - a[1]
    return math.degrees(math.atan2(dc, -dr)) % 360


class Rover:
    def __init__(self, world: World, mode: PerceptionMode = "oracle",
                 cost_cfg: CostConfig | None = None, sensor: SensorConfig | None = None,
                 rover_id: str = "rover-01", start: Cell = (190, 10)):
        self.world = world
        self.mode: PerceptionMode = mode
        self.base_mode: PerceptionMode = mode
        self.cost_cfg = cost_cfg or CostConfig()
        self.sensor = sensor or SensorConfig()
        self.rover_id = rover_id
        self.belief = Belief.empty()
        self.flags = ObserveFlags()

        self.pose = Pose(row=float(start[0]), col=float(start[1]), heading_deg=0.0)
        self.state = State.IDLE
        self.goal: Cell | None = None
        self.path: list[Cell] = []
        self.path_idx = 0
        self.route_version = 0
        self._plan_remaining: np.ndarray | None = None  # remaining cost from each path index
        self._cost: np.ndarray | None = None
        self._blocked: np.ndarray | None = None
        self._cost_version = -1

        self.mission_time = 0.0
        self.seq = 0
        self.battery = 100.0
        self.odometer_m = 0.0
        self.since_contact_m = 0.0
        self.since_scan_m = 0.0
        self.last_obs_time = 0.0
        self.link_up = True
        self.frame_image_id: str | None = None

        self.replans = 0
        self.replan_latencies_ms: list[float] = []
        self.hazard_entries = 0
        self._in_hazard = False
        self.trajectory: list[Cell] = [self.pose.cell]
        self.safe_stop_reason = ""

        self._events: list[Event] = []
        self._changed_tiles: set[str] = set()
        # Fault hooks (faults.py sets these): number of upcoming observations affected.
        self.drop_frames = 0
        self.corrupt_frames = 0
        self.blur_frames = 0
        self.blur_level = 0.0

    # ------------------------------------------------------------------ helpers
    def _emit(self, level: str, kind: str, msg: str, **data) -> Event:
        ev = Event(self.mission_time, level, kind, msg, data)
        self._events.append(ev)
        return ev

    def _set_state(self, new: State) -> None:
        if new != self.state:
            self._emit("info", "state", f"{self.state.value} -> {new.value}",
                       old=self.state.value, new=new.value)
            self.state = new

    def safe_stop(self, reason: str) -> None:
        self.safe_stop_reason = reason
        self._set_state(State.SAFE_STOP)
        self._emit("error", "safe_stop", f"SAFE_STOP: {reason}")

    @property
    def cell(self) -> Cell:
        return self.pose.cell

    def _effective_cfg(self) -> CostConfig:
        if self.mode in ("geometry_only", "offline") and self.cost_cfg.use_semantics:
            return replace(self.cost_cfg, use_semantics=False)
        return self.cost_cfg

    def _grids(self) -> tuple[np.ndarray, np.ndarray]:
        if self._cost is None or self._cost_version != self.belief.version:
            self._cost, self._blocked = cost_grid(self.world, self.belief, self._effective_cfg())
            self._cost_version = self.belief.version
        return self._cost, self._blocked

    def _invalidate_grids(self) -> None:
        self._cost = None

    def _speed_m(self) -> float:
        return DRIVE_M_PER_TICK * (0.5 if self.mode == "geometry_only" else 1.0)

    def _dist_to_goal_m(self) -> float:
        if self.goal is None:
            return math.inf
        return math.hypot(self.pose.row - self.goal[0], self.pose.col - self.goal[1]) * RES_M

    # ------------------------------------------------------------------ sensing
    def _observe(self) -> bool:
        flags = ObserveFlags(
            drop_frame=self.drop_frames > 0,
            corrupt_frame=self.corrupt_frames > 0 and self.drop_frames == 0,
            blur=self.blur_level if self.blur_frames > 0 else 0.0,
            model_offline=self.mode in ("geometry_only", "offline"),
        )
        if self.drop_frames > 0:
            self.drop_frames -= 1
        elif self.corrupt_frames > 0:
            self.corrupt_frames -= 1
        elif self.blur_frames > 0:
            self.blur_frames -= 1
        self.flags = flags
        res = observe(self.world, self.belief, self.pose, self.sensor, flags, self.mode)
        if res.frame_image_id:
            self.frame_image_id = res.frame_image_id
        self._changed_tiles.update(res.tiles_seen)
        self.since_scan_m = 0.0
        if res.ok:
            self.last_obs_time = self.mission_time
            if flags.blur > 0:
                self._emit("warn", "low_confidence", f"Blurred frame (blur {flags.blur:.1f})")
            return True
        if res.reason == "corrupt_frame":
            self._emit("error", "frame_rejected", "Frame rejected: checksum failed, rescanning")
        elif res.reason == "dropped":
            self._emit("warn", "frame_dropped", "No frame received")
        else:
            self._emit("error", "perception_error", f"Perception failed: {res.reason}")
        return False

    # ------------------------------------------------------------------ planning
    def _plan(self, reason: str) -> bool:
        t0 = time.perf_counter()
        cost, blocked = self._grids()
        start = self.cell
        if blocked[start]:
            blocked = blocked.copy()
            _stamp(blocked, start, _START_CLEAR, False)
        path = plan(cost, blocked, start, self.goal)
        latency = (time.perf_counter() - t0) * 1000
        if path is None:
            self.path, self.path_idx, self._plan_remaining = [], 0, None
            self.safe_stop("no route to goal")
            return False
        self.path, self.path_idx = path, 0
        seg = np.array([0.5 * (cost[a] + cost[b]) * _step_len(a, b)
                        for a, b in zip(path, path[1:])] + [0.0])
        self._plan_remaining = np.cumsum(seg[::-1])[::-1]
        self.route_version += 1
        if reason != "initial":
            self.replans += 1
            self.replan_latencies_ms.append(latency)
        self._emit("info", "route", f"Route v{self.route_version} ({reason}), "
                   f"{len(path)} cells, {latency:.0f} ms",
                   route_version=self.route_version, reason=reason, latency_ms=round(latency, 1))
        return True

    def _replan_reason(self) -> str | None:
        if not self.path:
            return "no path"
        cost, blocked = self._grids()
        rest = self.path[self.path_idx + 1:]
        if not rest:
            return None
        idx = np.array(rest)
        if blocked[idx[:, 0], idx[:, 1]].any():
            return "path blocked"
        now = sum(0.5 * (float(cost[a]) + float(cost[b])) * _step_len(a, b)
                  for a, b in zip(self.path[self.path_idx:], rest))
        then = float(self._plan_remaining[self.path_idx])
        if then > 0 and now > REPLAN_COST_RISE * then:
            return f"path cost +{(now / then - 1) * 100:.0f}%"
        return None

    # ------------------------------------------------------------------ simulator
    def _hazard_at(self, cell: Cell) -> bool:
        """Simulator ground truth (the ONLY place the rover module reads world.truth)."""
        r = FOOTPRINT_CELLS
        r0, c0 = cell
        rs, re = max(r0 - r, 0), min(r0 + r + 1, GRID)
        cs, ce = max(c0 - r, 0), min(c0 + r + 1, GRID)
        sub = _FOOTPRINT[rs - (r0 - r):re - (r0 - r), cs - (c0 - r):ce - (c0 - r)]
        truth = self.world.truth[rs:re, cs:ce][sub]
        slope = self.world.slope_deg[rs:re, cs:ce][sub]
        geo = self.world.geo_obstacle[rs:re, cs:ce][sub]
        return bool(((truth == SAND) | (truth == BIG_ROCK)).any()
                    or (slope > self.cost_cfg.max_slope_deg).any() or geo.any())

    def _enter(self, cell: Cell) -> None:
        self.trajectory.append(cell)
        hazard = self._hazard_at(cell)
        if hazard and not self._in_hazard:
            self.hazard_entries += 1
            self._emit("error", "hazard", f"Entered hazard at {cell}", cell=list(cell))
        self._in_hazard = hazard

    def _drive(self) -> None:
        budget = self._speed_m()
        old_tile = cell_to_tile(self.cell)
        moved = 0.0
        while self.path_idx < len(self.path) - 1:
            a, b = self.path[self.path_idx], self.path[self.path_idx + 1]
            length = _step_len(a, b)
            if moved + length > budget + 1e-9:
                break
            moved += length
            self.path_idx += 1
            self.pose = Pose(row=float(b[0]), col=float(b[1]), heading_deg=_heading(a, b))
            self._enter(b)
        self.odometer_m += moved
        self.since_scan_m += moved
        self.battery = max(0.0, self.battery - BATTERY_PER_M * moved)
        if not self.link_up:
            self.since_contact_m += moved
        if cell_to_tile(self.cell) != old_tile:
            self._entered_new_tile = True

    # ------------------------------------------------------------------ operator API
    def set_target(self, goal: Cell) -> None:
        self.goal = (int(goal[0]), int(goal[1]))
        self.safe_stop_reason = ""
        self._emit("info", "target", f"New target {self.goal}", goal=list(self.goal))
        self.pose.heading_deg = _heading(self.cell, self.goal) if self.goal != self.cell \
            else self.pose.heading_deg
        self._set_state(State.PLANNING)

    def resume(self) -> bool:
        """Operator command: the only way out of SAFE_STOP."""
        if self.state != State.SAFE_STOP:
            return False
        self.safe_stop_reason = ""
        self.last_obs_time = self.mission_time
        self.since_contact_m = 0.0
        self._emit("info", "resume", "Operator resumed")
        self._set_state(State.PLANNING if self.goal is not None else State.IDLE)
        return True

    def stop(self) -> None:
        self.goal = None
        self.path = []
        self._set_state(State.IDLE)

    def set_mode(self, mode: PerceptionMode) -> None:
        if mode == self.mode:
            return
        self._emit("warn" if mode != self.base_mode else "info", "mode",
                   f"Perception mode {self.mode} -> {mode}", old=self.mode, new=mode)
        self.mode = mode
        self._invalidate_grids()

    def earth_contact(self) -> None:
        self.since_contact_m = 0.0

    # ------------------------------------------------------------------ main loop
    def step(self) -> list[Event]:
        """Advance one fixed tick. Returns the events emitted during this tick."""
        self._events = []
        self._entered_new_tile = False
        self.mission_time += TICK_S
        self.seq += 1
        self.battery = max(0.0, self.battery - BATTERY_PER_TICK)
        if self.link_up:
            self.since_contact_m = 0.0

        s = self.state
        if s in (State.IDLE, State.SAFE_STOP):
            pass
        elif s in (State.PLANNING, State.REPLANNING):
            if s == State.PLANNING:
                self._observe()
            if self._plan("initial" if s == State.PLANNING else self._pending_reason):
                self._set_state(State.DRIVING)
        elif s == State.SCANNING:
            if self._observe():
                self._after_scan()
            elif self.mission_time - self.last_obs_time > T_STALE_S:
                self.safe_stop(f"no valid observation for {T_STALE_S:.0f} s")
            # else: stay SCANNING and retry next tick
        elif s == State.DRIVING:
            self._drive_tick()
        return list(self._events)

    _pending_reason = "replan"

    def _after_scan(self) -> None:
        if self._dist_to_goal_m() <= ARRIVE_M:
            self._arrive()
            return
        reason = self._replan_reason() or self._cheaper_route_reason()
        if reason:
            self._pending_reason = reason
            self._emit("warn", "replan_trigger", f"Replanning: {reason}")
            self._set_state(State.REPLANNING)
        else:
            self._set_state(State.DRIVING)

    def _cheaper_route_reason(self) -> str | None:
        """After a scan: is there now a clearly cheaper route than the one we're on?"""
        if not self.path or self.goal is None:
            return None
        cost, blocked = self._grids()
        current = self.path[self.path_idx:]
        now = sum(0.5 * (float(cost[a]) + float(cost[b])) * _step_len(a, b)
                  for a, b in zip(current, current[1:]))
        start = self.cell
        if blocked[start]:
            blocked = blocked.copy()
            _stamp(blocked, start, _START_CLEAR, False)
        alt = plan(cost, blocked, start, self.goal)
        if alt is None:
            return None
        alt_cost = sum(0.5 * (float(cost[a]) + float(cost[b])) * _step_len(a, b)
                       for a, b in zip(alt, alt[1:]))
        if alt_cost < CHEAPER_ROUTE * now:
            return f"cheaper route -{(1 - alt_cost / now) * 100:.0f}%"
        return None

    def _arrive(self) -> None:
        self._emit("info", "arrived", f"Arrived at {self.goal}")
        self.path = []
        self._set_state(State.IDLE)

    def _drive_tick(self) -> None:
        if self.mission_time - self.last_obs_time > T_STALE_S:
            self.safe_stop(f"observation stale > {T_STALE_S:.0f} s")
            return
        if self.since_contact_m > AUTONOMY_LIMIT_M:
            self.safe_stop(f"{AUTONOMY_LIMIT_M:.0f} m driven without Earth contact")
            return
        reason = self._replan_reason()
        if reason:
            self._pending_reason = reason
            self._emit("warn", "replan_trigger", f"Replanning: {reason}")
            self._set_state(State.REPLANNING)
            return
        self._drive()
        if self._dist_to_goal_m() <= ARRIVE_M:
            self._arrive()
        elif self.path_idx >= len(self.path) - 1:
            self._set_state(State.SCANNING)  # end of path but not there: rescan + replan
        elif self.since_scan_m >= SCAN_EVERY_M or self._entered_new_tile:
            self._set_state(State.SCANNING)

    # ------------------------------------------------------------------ headless
    def run_to_completion(self, start: Cell, goal: Cell, max_steps: int = 2000) -> MissionResult:
        self.pose = Pose(row=float(start[0]), col=float(start[1]), heading_deg=0.0)
        self.trajectory = [self.cell]
        self._in_hazard = self._hazard_at(self.cell)
        self.set_target(goal)
        steps = 0
        reached = False
        while steps < max_steps:
            self.step()
            steps += 1
            if self.state == State.IDLE:
                reached = self._dist_to_goal_m() <= ARRIVE_M
                break
            if self.state == State.SAFE_STOP:
                break
        return MissionResult(
            reached=reached, hazard_entries=self.hazard_entries,
            path_len_m=round(self.odometer_m, 3), steps=steps, replans=self.replans,
            replan_latencies_ms=list(self.replan_latencies_ms),
            final_state=self.state.value, trajectory=list(self.trajectory))

    # ------------------------------------------------------------------ telemetry
    def belief_delta(self) -> dict:
        """Changed tiles since the last call: 20x20 argmax class (255 unknown) and
        uncertainty (uint8 0..255), each base64 of 400 row-major bytes."""
        tiles = {}
        for tid in sorted(self._changed_tiles):
            _, tr, tc = tid.split("_")
            tr, tc = int(tr), int(tc)
            region = np.s_[tr * TILE:(tr + 1) * TILE, tc * TILE:(tc + 1) * TILE]
            cls = self.belief.probs[region].argmax(axis=-1).astype(np.uint8)
            cls[~self.belief.known[region]] = 255
            unc = np.clip(self.belief.unc[region] * 255, 0, 255).astype(np.uint8)
            tiles[tid] = {
                "cls": base64.b64encode(cls.tobytes()).decode(),
                "unc": base64.b64encode(unc.tobytes()).decode(),
            }
        self._changed_tiles.clear()
        return {"tiles": tiles}

    def blocked_cells(self) -> list[list[int]]:
        _, blocked = self._grids()
        rows, cols = np.nonzero(blocked)
        return np.stack([rows, cols], axis=1).tolist()

    def snapshot(self, events: list[Event] | None = None, include_belief: bool = True) -> dict:
        return {
            "type": "tick",
            "mission_time": round(self.mission_time, 2),
            "seq": self.seq,
            "rover_id": self.rover_id,
            "pose": {"x": round(self.pose.col, 2), "y": round(self.pose.row, 2),
                     "heading": round(self.pose.heading_deg, 1)},
            "state": self.state.value,
            "safe_stop_reason": self.safe_stop_reason,
            "battery": round(self.battery, 1),
            "odometer_m": round(self.odometer_m, 2),
            "since_contact_m": round(self.since_contact_m, 2),
            "route_version": self.route_version,
            "path": [[c, r] for r, c in self.path[self.path_idx:]],
            "goal": None if self.goal is None else [self.goal[1], self.goal[0]],
            "belief_delta": self.belief_delta() if include_belief else {"tiles": {}},
            "perception_mode": self.mode,
            "frame_image_id": self.frame_image_id,
            "link": {"status": "CONNECTED" if self.link_up else "BLACKOUT", "delay_s": 0.0},
            "events": [e.to_dict() for e in (events or [])],
        }


__all__ = ["Rover", "State", "Event", "MissionResult", "TICK_S", "tile_id"]
