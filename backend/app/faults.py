"""Fault injection (Stream B owns). Seven faults, PRD 7.7:
hidden_rock, drop_frame, corrupt_frame, low_confidence, model_offline, blackout, stale_command.
Most map to contracts.ObserveFlags or World.geo_obstacle.

Faults are environment events, so they act on the rover/world/link directly (not through
the delayed uplink). ``stale_command`` is the exception: it is a ground command and goes
through the link like any other, then gets rejected on arrival.
"""
from __future__ import annotations

import math

import numpy as np

from .contracts import GRID, RES_M
from .link import Link
from .rover import Rover

FAULTS = ("hidden_rock", "drop_frame", "corrupt_frame", "low_confidence",
          "model_offline", "blackout", "stale_command")

HIDDEN_ROCK_AHEAD_M = 4.0
HIDDEN_ROCK_RADIUS_CELLS = 3
DROP_FRAMES = 14          # > T_STALE at the scan cadence -> watchdog SAFE_STOP
BLUR_FRAMES = 6
BLUR_LEVEL = 0.8
BLACKOUT_S = 15.0


def _cell_ahead(rover: Rover, ahead_m: float) -> tuple[int, int]:
    """A cell on the remaining path ~ahead_m ahead, else straight ahead on the heading."""
    if rover.path and rover.path_idx < len(rover.path) - 1:
        travelled = 0.0
        for a, b in zip(rover.path[rover.path_idx:], rover.path[rover.path_idx + 1:]):
            travelled += RES_M * (math.sqrt(2) if a[0] != b[0] and a[1] != b[1] else 1.0)
            if travelled >= ahead_m:
                return b
        # Short remaining path: don't drop the rock on the goal itself.
        mid = rover.path_idx + (len(rover.path) - 1 - rover.path_idx) // 2
        return rover.path[mid]
    h = math.radians(rover.pose.heading_deg)
    cells = ahead_m / RES_M
    r = int(np.clip(round(rover.pose.row - cells * math.cos(h)), 0, GRID - 1))
    c = int(np.clip(round(rover.pose.col + cells * math.sin(h)), 0, GRID - 1))
    return r, c


def inject(rover: Rover, link: Link, fault: str, now: float | None = None,
           **params) -> dict:
    """Apply one fault. Returns {"fault", "msg", ...details} and logs an event on the rover."""
    now = rover.mission_time if now is None else now
    if fault == "hidden_rock":
        center = params.get("cell") or _cell_ahead(rover, float(params.get("ahead_m", HIDDEN_ROCK_AHEAD_M)))
        r0, c0 = int(center[0]), int(center[1])
        rad = int(params.get("radius_cells", HIDDEN_ROCK_RADIUS_CELLS))
        yy, xx = np.ogrid[:GRID, :GRID]
        blob = (yy - r0) ** 2 + (xx - c0) ** 2 <= rad * rad
        rover.world.geo_obstacle |= blob
        msg = f"Hidden rock placed at ({r0}, {c0})"
        detail = {"cell": [r0, c0], "radius_cells": rad}
    elif fault == "drop_frame":
        n = int(params.get("frames", DROP_FRAMES))
        rover.drop_frames = n
        msg = f"Camera dropping the next {n} frames"
        detail = {"frames": n}
    elif fault == "corrupt_frame":
        rover.corrupt_frames = int(params.get("frames", 1))
        msg = "Next frame will be corrupted"
        detail = {"frames": rover.corrupt_frames}
    elif fault == "low_confidence":
        rover.blur_frames = int(params.get("frames", BLUR_FRAMES))
        rover.blur_level = float(params.get("blur", BLUR_LEVEL))
        note = "" if rover.mode == "model" else " (no effect: blur only degrades model perception)"
        msg = f"Camera blurred ({rover.blur_level:.1f}) for {rover.blur_frames} frames{note}"
        detail = {"frames": rover.blur_frames, "blur": rover.blur_level}
    elif fault == "model_offline":
        rover.set_mode("geometry_only")
        msg = "Terrain model offline: geometry-only perception at half speed"
        detail = {}
    elif fault == "blackout":
        dur = float(params.get("duration_s", BLACKOUT_S))
        link.start_blackout(now, dur)
        msg = f"Comms blackout for {dur:.0f} s"
        detail = {"duration_s": dur}
    elif fault == "stale_command":
        stale_rv = int(params.get("route_version", rover.route_version - 1))
        cmd = {"type": "resume" if rover.state.value == "SAFE_STOP" else "stop",
               "precondition": {"route_version": stale_rv}, "origin": "fault:stale_command"}
        sent = link.send_up(now, cmd)
        msg = (f"Sent '{cmd['type']}' built on route v{stale_rv} "
               f"(arrives in {link.delay_s:.1f} s)" if sent else
               "Stale command dropped: link in blackout")
        detail = {"route_version": stale_rv, "sent": sent is not None}
    else:
        raise ValueError(f"unknown fault {fault!r}; expected one of {FAULTS}")
    rover._emit("warn", "fault", f"FAULT {fault}: {msg}", fault=fault, **detail)
    return {"fault": fault, "msg": msg, **detail}
