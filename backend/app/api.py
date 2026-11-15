"""FastAPI app (Stream B owns). Run from repo root:

    .venv/Scripts/python -m uvicorn backend.app.api:app --reload --port 8000

One ``Mission`` per id: a Rover stepped by an asyncio task at a fixed tick, broadcasting the
tick payload to every WebSocket subscriber. The Rover itself has no FastAPI/asyncio in it.
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

from .contracts import GRID, CostConfig, World
from .rover import TICK_S, Rover

log = logging.getLogger("redroute")
app = FastAPI(title="RedRoute")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

DEFAULT_SEED = 1
DEFAULT_START = (190, 10)   # (row, col), see world/DEMO_SEEDS.md
DEFAULT_GOAL = (10, 120)


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
    subscribers: set[WebSocket] = field(default_factory=set)
    task: asyncio.Task | None = None
    blocked_version: int = -2
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def tick_payload(self, events, full: bool = False) -> dict[str, Any]:
        r = self.rover
        if full:
            r._changed_tiles.update(f"t_{tr}_{tc}" for tr in range(10) for tc in range(10)
                                    if r.belief.known[tr*20:(tr+1)*20, tc*20:(tc+1)*20].any())
        snap = r.snapshot(events)
        snap["mission_id"] = self.id
        snap["seed"] = self.seed
        _, blocked = r._grids()
        if full or r._cost_version != self.blocked_version:
            snap["blocked"] = _pack_bits(blocked)
            self.blocked_version = r._cost_version
        return snap


MISSIONS: dict[str, Mission] = {}


def _new_mission(mission_id: str, seed: int, mode: str = "oracle") -> Mission:
    world = build_world(seed)
    rover = Rover(world, mode=mode, cost_cfg=CostConfig(), start=DEFAULT_START)
    return Mission(id=mission_id, seed=seed, world=world, rover=rover)


def get_mission(mission_id: str, seed: int | None = None) -> Mission:
    m = MISSIONS.get(mission_id)
    if m is None:
        m = MISSIONS[mission_id] = _new_mission(mission_id, seed if seed is not None else DEFAULT_SEED)
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


async def _run(m: Mission) -> None:
    loop = asyncio.get_running_loop()
    try:
        while m.subscribers:
            t0 = loop.time()
            async with m.lock:
                try:
                    events = await asyncio.to_thread(m.rover.step)
                except Exception as exc:  # never let one bad tick kill the mission loop
                    log.exception("rover step failed")
                    m.rover.safe_stop(f"internal error: {exc}")
                    events = m.rover._events
                payload = m.tick_payload(events)
            await _broadcast(m, payload)
            await asyncio.sleep(max(0.0, TICK_S - (loop.time() - t0)))
    finally:
        m.task = None


@app.websocket("/ws/missions/{mission_id}")
async def mission_ws(ws: WebSocket, mission_id: str, seed: int | None = None):
    await ws.accept()
    m = get_mission(mission_id, seed)
    m.subscribers.add(ws)
    async with m.lock:
        await ws.send_json(m.tick_payload([], full=True))
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
    type: Literal["target", "resume", "stop", "reset", "mode"]
    x: int | None = None          # column
    y: int | None = None          # row
    mode: Literal["oracle", "model", "geometry_only"] | None = None
    seed: int | None = None


@app.get("/api/missions/{mission_id}")
def mission_info(mission_id: str):
    m = get_mission(mission_id)
    r = m.rover
    return {"id": m.id, "seed": m.seed, "state": r.state.value, "route_version": r.route_version,
            "default_goal": [DEFAULT_GOAL[1], DEFAULT_GOAL[0]]}


@app.post("/api/missions/{mission_id}/command")
async def mission_command(mission_id: str, cmd: Command):
    m = get_mission(mission_id)
    async with m.lock:
        r = m.rover
        if cmd.type == "target":
            if cmd.x is None or cmd.y is None or not (0 <= cmd.x < GRID and 0 <= cmd.y < GRID):
                raise HTTPException(400, "target needs x (col) and y (row) inside the grid")
            r.set_target((cmd.y, cmd.x))
        elif cmd.type == "resume":
            if not r.resume():
                raise HTTPException(409, f"resume only valid in SAFE_STOP (state {r.state.value})")
        elif cmd.type == "stop":
            r.stop()
        elif cmd.type == "mode":
            if cmd.mode is None:
                raise HTTPException(400, "mode command needs 'mode'")
            r.base_mode = cmd.mode
            r.set_mode(cmd.mode)
        elif cmd.type == "reset":
            seed = cmd.seed if cmd.seed is not None else m.seed
            fresh = _new_mission(m.id, seed, cmd.mode or r.base_mode)
            m.seed, m.world, m.rover, m.blocked_version = seed, fresh.world, fresh.rover, -2
            await _broadcast(m, m.tick_payload([], full=True) | {"reset": True})
    return {"ok": True, "state": m.rover.state.value, "route_version": m.rover.route_version}


@app.on_event("shutdown")
async def _shutdown():
    for m in MISSIONS.values():
        if m.task:
            m.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await m.task
