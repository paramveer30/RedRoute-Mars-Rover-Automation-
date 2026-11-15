"""Rover simulator + safety state machine (Stream B owns). See docs/STREAM_B.md.

States: IDLE, SCANNING, PLANNING, DRIVING, REPLANNING, SAFE_STOP (PRD 7.5).
"""
from __future__ import annotations

from enum import Enum


class State(str, Enum):
    IDLE = "IDLE"
    SCANNING = "SCANNING"
    PLANNING = "PLANNING"
    DRIVING = "DRIVING"
    REPLANNING = "REPLANNING"
    SAFE_STOP = "SAFE_STOP"


# TODO Stream B: class Rover with step(dt) driving the loop
#   observe() -> cost_grid() -> plan() -> move 1 m -> check replanning triggers -> watchdog
