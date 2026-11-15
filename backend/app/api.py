"""FastAPI app (Stream B owns). Run from repo root:

    .venv/Scripts/python -m uvicorn backend.app.api:app --reload --port 8000

Right now: a "hello" tick loop. The rover drives a straight line on the stub world with the
oracle perception stub, so the frontend can render real-shaped messages from minute one.
Replace the loop body with the Rover state machine (rover.py) once it exists.
"""
from __future__ import annotations

import asyncio
import math

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from world.stub_world import build_stub_world

from .contracts import GRID, Belief, Pose
from .perception import observe

app = FastAPI(title="RedRoute")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

TICK_S = 0.5


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/world/{seed}")
def world_layers(seed: int):
    """Static layers for the map (elevation for shading). Truth is debug-only."""
    w = build_stub_world(seed)
    return {
        "grid": GRID,
        "elevation": np.round(w.elevation, 3).tolist(),
        "slope_deg": np.round(w.slope_deg, 2).tolist(),
        "tile_image": w.tile_image,
    }


@app.websocket("/ws/missions/{mission_id}")
async def mission_ws(ws: WebSocket, mission_id: str):
    await ws.accept()
    world = build_stub_world(0)
    belief = Belief.empty()
    pose = Pose(row=180.0, col=20.0, heading_deg=45.0)
    seq, t = 0, 0.0
    try:
        while True:
            res = observe(world, belief, pose, mode="oracle")
            # 1 m step = 4 cells along heading (0 = north, clockwise)
            h = math.radians(pose.heading_deg)
            pose.row = float(np.clip(pose.row - 4 * math.cos(h), 0, GRID - 1))
            pose.col = float(np.clip(pose.col + 4 * math.sin(h), 0, GRID - 1))
            if pose.row <= 0 or pose.col >= GRID - 1:
                pose = Pose(row=180.0, col=20.0, heading_deg=45.0)
            seq += 1
            t += TICK_S
            await ws.send_json({
                "type": "tick", "mission_time": t, "seq": seq, "rover_id": "rover-01",
                "pose": {"x": round(pose.col, 2), "y": round(pose.row, 2),
                         "heading": pose.heading_deg},
                "state": "DRIVING", "battery": 100, "route_version": 0, "path": [],
                "belief_delta": {"tiles": res.tiles_seen},
                "perception_mode": "oracle",
                "link": {"status": "CONNECTED", "delay_s": 0.0},
                "events": [],
            })
            await asyncio.sleep(TICK_S)
    except WebSocketDisconnect:
        pass
