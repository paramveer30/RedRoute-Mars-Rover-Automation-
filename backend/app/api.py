"""FastAPI app (Stream B owns). Run from repo root:

    .venv/Scripts/python -m uvicorn backend.app.api:app --reload --port 8000

One ``Mission`` per id: a Rover stepped by an asyncio task at a fixed tick. Everything between
ground and rover goes through the simulated ``Link``:

  operator command --REST--> link.send_up --(delay)--> precondition check --> Rover
  Rover.snapshot --> link.send_down --(delay / blackout buffer)--> WebSocket --> operator

Reset, perception-mode and link-compression changes are sim/debug controls and act at once.
The Rover itself has no FastAPI/asyncio in it.
"""
from __future__ import annotations

import asyncio
import base64
import contextlib
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Literal

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from world.builder import build_world

from . import faults
from .contracts import GRID, N_TILES, TILE, CostConfig, World, tile_id
from .link import Link, check_precondition
from .missionlog import MissionLog
from .rover import TICK_S, Rover

log = logging.getLogger("redroute")


@contextlib.asynccontextmanager
async def _lifespan(_app):
    yield
    for m in MISSIONS.values():
        if m.task:
            m.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await m.task


app = FastAPI(title="RedRoute", lifespan=_lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

DEFAULT_SEED = 1
DEFAULT_START = (190, 10)   # (row, col), see world/DEMO_SEEDS.md
DEFAULT_GOAL = (10, 120)

_mission_log: MissionLog | None = None


def mission_log() -> MissionLog:
    global _mission_log
    if _mission_log is None:
        _mission_log = MissionLog()
    return _mission_log


@lru_cache(maxsize=8)
def _world_layers(seed: int) -> dict:
    w = build_world(seed)
    return {
        "seed": seed,
        "grid": GRID,
        "elevation": np.round(w.elevation, 3).tolist(),
        "slope_deg": np.round(w.slope_deg, 2).tolist(),
        "tile_image": w.tile_image,
    }


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/world/{seed}")
def world_layers(seed: int):
    """Static layers for the map (elevation for shading). Truth is debug-only."""
    return _world_layers(seed)


@app.get("/api/world/{seed}/truth")
def world_truth(seed: int):
    """DEBUG ONLY: ground-truth classes for the TRUTH overlay. The rover never sees this."""
    w = build_world(seed)
    return {"seed": seed, "truth": base64.b64encode(w.truth.astype(np.uint8).tobytes()).decode()}


# ---------------------------------------------------------------- missions
def _pack_bits(mask: np.ndarray) -> str:
    return base64.b64encode(np.packbits(mask.astype(bool), axis=None).tobytes()).decode()


@dataclass
class Mission:
    id: str
    seed: int
    world: World
    rover: Rover
    link: Link
    run_id: int
    subscribers: set[WebSocket] = field(default_factory=set)
    task: asyncio.Task | None = None
    blocked_version: int = -2
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def rover_payload(self, events, full: bool = False) -> dict[str, Any]:
        """What the rover transmits this tick (before the link)."""
        r = self.rover
        if full:
            r._changed_tiles.update(
                tile_id(tr, tc) for tr in range(N_TILES) for tc in range(N_TILES)
                if r.belief.known[tr*TILE:(tr+1)*TILE, tc*TILE:(tc+1)*TILE].any())
        snap = r.snapshot(events)
        snap["mission_id"] = self.id
        snap["seed"] = self.seed
        _, blocked = r._grids()
        if full or r._cost_version != self.blocked_version:
            snap["blocked"] = _pack_bits(blocked)
            self.blocked_version = r._cost_version
        return snap

    def ground_view(self, body: dict, late: bool = False) -> dict:
        """Stamp a delivered payload with the ground's view of the link and clocks."""
        out = dict(body)
        out["link"] = self.link.info(self.rover.mission_time)
        out["late"] = late
        if late:
            out["events"] = [{**e, "late": True} for e in body.get("events", [])]
        return out


MISSIONS: dict[str, Mission] = {}


def _new_mission(mission_id: str, seed: int, mode: str = "oracle",
                 compression: float | None = None) -> Mission:
    world = build_world(seed)
    rover = Rover(world, mode=mode, cost_cfg=CostConfig(), start=DEFAULT_START)
    link = Link() if compression is None else Link(compression=compression)
    try:
        run_id = mission_log().start(mission_id, seed, mode)
    except Exception:
        log.exception("mission log unavailable")
        run_id = -1
    return Mission(id=mission_id, seed=seed, world=world, rover=rover, link=link, run_id=run_id)


def get_mission(mission_id: str, seed: int | None = None) -> Mission:
    m = MISSIONS.get(mission_id)
    if m is None:
        m = MISSIONS[mission_id] = _new_mission(
            mission_id, seed if seed is not None else DEFAULT_SEED)
    return m


async def _broadcast(m: Mission, payload: dict) -> None:
    dead = []
    for ws in list(m.subscribers):
        try:
            await ws.send_json(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        m.subscribers.discard(ws)


async def _ground_event(m: Mission, level: str, msg: str, **data) -> None:
    """Something the ground knows immediately (not via downlink), e.g. an uplink drop."""
    await _broadcast(m, {"type": "ground_event", "level": level, "msg": msg,
                         "mission_time": round(m.link.ground_time, 2), "data": data})


def apply_command(rover: Rover, cmd: dict) -> None:
    """Run a command that has arrived at the rover. Rejections become rover events."""
    check = check_precondition(cmd, rover.route_version, rover.state.value)
    kind = cmd.get("type")
    if not check.accepted:
        rover._emit("error", "command_rejected", f"Rejected {kind}: {check.reason}",
                    command=kind, reason=check.reason)
        return
    if kind == "target":
        rover.set_target((int(cmd["y"]), int(cmd["x"])))
    elif kind == "resume":
        if not rover.resume():
            rover._emit("warn", "command_rejected",
                        f"Rejected resume: rover is {rover.state.value}, not SAFE_STOP")
    elif kind == "stop":
        rover.stop()
        rover._emit("info", "command", "Operator stop")
    else:
        rover._emit("warn", "command_rejected", f"Rejected unknown command {kind!r}")


def tick(m: Mission) -> tuple[list[dict], dict]:
    """One sim tick (sync; runs in a worker thread). Returns (delivered payloads, rover payload)."""
    r, link = m.rover, m.link
    now = r.mission_time + 1e-9
    r.link_up = not link.in_blackout(now)
    for msg in link.recv_up(now):
        apply_command(r, msg.body)
    events = r.step()
    if m.run_id >= 0:
        try:
            mission_log().record(m.run_id, r, events)
        except Exception:
            log.exception("mission log write failed")
    payload = m.rover_payload(events)
    link.send_down(r.mission_time, payload, r.route_version)
    delivered = [m.ground_view(msg.body, msg.late) for msg in link.recv_down(r.mission_time)]
    return delivered, payload


async def _run(m: Mission) -> None:
    loop = asyncio.get_running_loop()
    try:
        while m.subscribers:
            t0 = loop.time()
            async with m.lock:
                try:
                    delivered, _ = await asyncio.to_thread(tick, m)
                except Exception as exc:  # never let one bad tick kill the mission loop
                    log.exception("tick failed")
                    m.rover.safe_stop(f"internal error: {exc}")
                    delivered = []
                status = m.link.info(m.rover.mission_time)
            for p in delivered:
                await _broadcast(m, p)
            # Clocks and link status update every tick, even when nothing is delivered.
            await _broadcast(m, {"type": "link", "link": status})
            await asyncio.sleep(max(0.0, TICK_S - (loop.time() - t0)))
    finally:
        m.task = None


@app.websocket("/ws/missions/{mission_id}")
async def mission_ws(ws: WebSocket, mission_id: str, seed: int | None = None):
    await ws.accept()
    m = get_mission(mission_id, seed)
    m.subscribers.add(ws)
    async with m.lock:
        # A new console gets the current picture once, then the delayed stream.
        await ws.send_json(m.ground_view(m.rover_payload([], full=True)) | {"initial": True})
    if m.task is None:
        m.task = asyncio.create_task(_run(m))
    try:
        while True:
            await ws.receive_text()   # commands go over REST; this just detects disconnect
    except WebSocketDisconnect:
        pass
    finally:
        m.subscribers.discard(ws)


class Command(BaseModel):
    type: Literal["target", "resume", "stop", "reset", "mode", "link"]
    x: int | None = None          # column
    y: int | None = None          # row
    mode: Literal["oracle", "model", "geometry_only"] | None = None
    seed: int | None = None
    compression: float | None = None
    route_version: int | None = None    # precondition: the route the operator was looking at
    expected_state: str | None = None   # precondition


class Fault(BaseModel):
    type: Literal["hidden_rock", "drop_frame", "corrupt_frame", "low_confidence",
                  "model_offline", "blackout", "stale_command"]
    duration_s: float | None = None
    frames: int | None = None


@app.get("/api/missions/{mission_id}")
def mission_info(mission_id: str):
    m = get_mission(mission_id)
    r = m.rover
    return {"id": m.id, "seed": m.seed, "run_id": m.run_id, "state": r.state.value,
            "route_version": r.route_version, "link": m.link.info(r.mission_time),
            "default_goal": [DEFAULT_GOAL[1], DEFAULT_GOAL[0]], "faults": list(faults.FAULTS)}


@app.post("/api/missions/{mission_id}/command")
async def mission_command(mission_id: str, cmd: Command):
    m = get_mission(mission_id)
    async with m.lock:
        r, link = m.rover, m.link
        now = r.mission_time
        if cmd.type in ("target", "resume", "stop"):
            if cmd.type == "target" and (cmd.x is None or cmd.y is None
                                         or not (0 <= cmd.x < GRID and 0 <= cmd.y < GRID)):
                raise HTTPException(400, "target needs x (col) and y (row) inside the grid")
            body = cmd.model_dump(exclude_none=True)
            pre = {k: body.pop(k) for k in ("route_version", "expected_state") if k in body}
            if pre:
                body["precondition"] = pre
            sent = link.send_up(now, body)
            if sent is None:
                await _ground_event(m, "error", f"Uplink dropped ({cmd.type}): link in BLACKOUT")
                raise HTTPException(409, "link in BLACKOUT: command dropped")
            await _ground_event(m, "info", f"Sent {cmd.type}, arrives in {link.delay_s:.1f} s",
                                arrive_at=round(sent.arrive_at, 2))
            return {"ok": True, "queued": True, "arrive_in_s": round(link.delay_s, 2)}
        if cmd.type == "mode":
            if cmd.mode is None:
                raise HTTPException(400, "mode command needs 'mode'")
            r.base_mode = cmd.mode
            r.set_mode(cmd.mode)
        elif cmd.type == "link":
            if cmd.compression is None or cmd.compression <= 0:
                raise HTTPException(400, "link command needs compression > 0")
            link.compression = float(cmd.compression)
        elif cmd.type == "reset":
            seed = cmd.seed if cmd.seed is not None else m.seed
            fresh = _new_mission(m.id, seed, cmd.mode or r.base_mode, link.compression)
            m.seed, m.world, m.rover, m.link = seed, fresh.world, fresh.rover, fresh.link
            m.run_id, m.blocked_version = fresh.run_id, -2
            await _broadcast(m, m.ground_view(m.rover_payload([], full=True)) | {"reset": True})
    return {"ok": True, "state": m.rover.state.value, "route_version": m.rover.route_version}


@app.post("/api/missions/{mission_id}/faults")
async def mission_fault(mission_id: str, fault: Fault):
    m = get_mission(mission_id)
    params = fault.model_dump(exclude_none=True)
    kind = params.pop("type")
    async with m.lock:
        result = faults.inject(m.rover, m.link, kind, **params)
    await _ground_event(m, "warn", f"Injected {kind}: {result['msg']}",
                        **{k: v for k, v in result.items() if k != "msg"})
    return result


@app.get("/api/missions/{mission_id}/log")
def mission_events(mission_id: str, limit: int = 200):
    m = get_mission(mission_id)
    rows = mission_log().query(
        "SELECT mission_time, level, kind, msg FROM event WHERE run_id=? "
        "ORDER BY id DESC LIMIT ?", (m.run_id, limit))
    routes = mission_log().query(
        "SELECT route_version, mission_time, reason, latency_ms FROM route WHERE run_id=? "
        "ORDER BY route_version", (m.run_id,))
    return {"run_id": m.run_id,
            "events": [dict(zip(("mission_time", "level", "kind", "msg"), r)) for r in rows],
            "routes": [dict(zip(("route_version", "mission_time", "reason", "latency_ms"), r))
                       for r in routes]}


