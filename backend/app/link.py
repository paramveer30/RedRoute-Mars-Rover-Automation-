"""Earth<->rover comms link simulator (Stream B owns). See docs/STREAM_B.md, hours 10-12, 15-17.

delay with time compression, blackout, onboard buffer, replay ordered by mission_time,
stale-command rejection via route_version precondition, HMAC + seq numbers (P1).
Cached real light delay: terrain/light_delay.json (scripts/fetch_light_delay.py).

Pure and clock-driven: every call takes ``now`` (rover mission time, seconds), so it can be
unit-tested without asyncio. The API advances it once per tick.

  downlink: rover -> ground telemetry. Normal: arrives ``delay_s`` later. BLACKOUT: buffered
            onboard; on restore the buffer is sent sorted by mission_time, each tagged late.
  uplink:   ground -> rover commands. Normal: arrives ``delay_s`` later. BLACKOUT: dropped.
            On arrival a command's precondition {route_version, expected_state} is checked.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

FALLBACK_DELAY_S = 656.6       # 10.9 min, JPL Horizons 2026-11-15 (cached)
DEFAULT_COMPRESSION = 60.0


def load_light_delay(path: str | Path | None = None) -> dict:
    """Real one-way light time from the cached JPL Horizons file."""
    if path is None:
        from world.paths import LIGHT_DELAY
        path = LIGHT_DELAY
    try:
        data = json.loads(Path(path).read_text())
        return {"delay_s": float(data["one_way_light_time_s"]), "date": data.get("date"),
                "source": data.get("source", "JPL Horizons")}
    except (OSError, KeyError, ValueError):
        return {"delay_s": FALLBACK_DELAY_S, "date": None, "source": "fallback constant"}


@dataclass(order=True)
class Message:
    arrive_at: float
    seq: int
    mission_time: float = field(compare=False)
    route_version: int = field(compare=False)
    body: dict[str, Any] = field(compare=False)
    late: bool = field(default=False, compare=False)


@dataclass
class CommandResult:
    accepted: bool
    reason: str = ""


class Link:
    def __init__(self, real_delay_s: float | None = None, compression: float = DEFAULT_COMPRESSION,
                 source: str | None = None, date: str | None = None):
        if real_delay_s is None:
            info = load_light_delay()
            real_delay_s, source, date = info["delay_s"], info["source"], info["date"]
        self.real_delay_s = float(real_delay_s)
        self.compression = float(compression)
        self.source = source or "JPL Horizons"
        self.date = date
        self.blackout_until = -1.0
        self._down: list[Message] = []      # in flight to ground
        self._up: list[Message] = []        # in flight to rover
        self._onboard: list[Message] = []   # buffered during blackout
        self._down_seq = 0
        self._up_seq = 0
        self.ground_time = 0.0              # mission_time of newest telemetry the ground has
        self.dropped_uplink = 0
        self._was_blackout = False
        self.auth = CommandAuth()

    # ---------------------------------------------------------------- state
    @property
    def delay_s(self) -> float:
        return self.real_delay_s / max(self.compression, 1e-9)

    def in_blackout(self, now: float) -> bool:
        return now < self.blackout_until

    def status(self, now: float) -> str:
        return "BLACKOUT" if self.in_blackout(now) else "CONNECTED"

    def start_blackout(self, now: float, duration_s: float) -> None:
        self.blackout_until = max(self.blackout_until, now + duration_s)

    def info(self, now: float) -> dict:
        return {
            "status": self.status(now),
            "delay_s": round(self.delay_s, 2),
            "real_delay_s": round(self.real_delay_s, 1),
            "real_delay_min": round(self.real_delay_s / 60, 2),
            "compression": self.compression,
            "source": self.source,
            "date": self.date,
            "blackout_remaining_s": round(max(0.0, self.blackout_until - now), 1),
            "buffered": len(self._onboard),
            "uplink_in_flight": len(self._up),
            "rover_time": round(now, 2),
            "ground_time": round(self.ground_time, 2),
        }

    # ---------------------------------------------------------------- downlink
    def send_down(self, now: float, body: dict, route_version: int = 0) -> None:
        self._down_seq += 1
        msg = Message(now + self.delay_s, self._down_seq, now, route_version, body)
        if self.in_blackout(now):
            self._onboard.append(msg)
        else:
            self._down.append(msg)

    def _flush_onboard(self, now: float) -> None:
        if not self._onboard:
            return
        for msg in sorted(self._onboard, key=lambda m: (m.mission_time, m.seq)):
            msg.arrive_at = now + self.delay_s
            msg.late = True
            self._down.append(msg)
        self._onboard.clear()

    def recv_down(self, now: float) -> list[Message]:
        """Telemetry arriving at the ground by ``now``, ordered by mission_time."""
        blackout = self.in_blackout(now)
        if self._was_blackout and not blackout:
            self._flush_onboard(now)
        self._was_blackout = blackout
        ready = [m for m in self._down if m.arrive_at <= now + 1e-9]
        if not ready:
            return []
        self._down = [m for m in self._down if m.arrive_at > now + 1e-9]
        ready.sort(key=lambda m: (m.mission_time, m.seq))
        self.ground_time = max(self.ground_time, ready[-1].mission_time)
        return ready

    # ---------------------------------------------------------------- uplink
    def send_up(self, now: float, command: dict) -> Message | None:
        """Queue a ground command. Returns None if dropped by a blackout."""
        if self.in_blackout(now):
            self.dropped_uplink += 1
            return None
        self._up_seq += 1
        msg = Message(now + self.delay_s, self._up_seq, now,
                      int(command.get("route_version") or 0), dict(command))
        self._up.append(msg)
        return msg

    def recv_up(self, now: float) -> list[Message]:
        ready = sorted(m for m in self._up if m.arrive_at <= now + 1e-9)
        self._up = [m for m in self._up if m.arrive_at > now + 1e-9]
        return ready


def check_precondition(command: dict, route_version: int, state: str) -> CommandResult:
    """Reject commands built on a superseded route or an unexpected rover state."""
    pre = command.get("precondition") or {
        k: command[k] for k in ("route_version", "expected_state") if command.get(k) is not None}
    expected_rv = pre.get("route_version")
    if expected_rv is not None and int(expected_rv) != int(route_version):
        return CommandResult(False, f"route v{expected_rv} superseded by v{route_version}")
    expected_state = pre.get("expected_state")
    if expected_state and expected_state != state:
        return CommandResult(False, f"expected state {expected_state}, rover is {state}")
    return CommandResult(True)


# ---------------------------------------------------------------- command security (P1)
import hashlib  # noqa: E402
import hmac  # noqa: E402
import os  # noqa: E402
import time as _time  # noqa: E402

OPERATOR_TOKEN = os.environ.get("RR_OPERATOR_TOKEN", "operator-demo")
VIEWER_TOKEN = os.environ.get("RR_VIEWER_TOKEN", "viewer-demo")


def role_for(token: str | None) -> str | None:
    if token and hmac.compare_digest(token, OPERATOR_TOKEN):
        return "operator"
    if token and hmac.compare_digest(token, VIEWER_TOKEN):
        return "viewer"
    return None


def canonical(cmd: dict) -> str:
    """Stable JSON of everything except the signature (the frontend mirrors this exactly)."""
    body = {k: v for k, v in cmd.items() if k != "sig" and v is not None}
    return json.dumps(body, sort_keys=True, separators=(",", ":"))


def sign(key: str, cmd: dict) -> str:
    """hmac_sha256(key, canonical(cmd) + seq), hex. ``seq`` is part of the signed body too."""
    msg = canonical(cmd) + str(cmd.get("seq"))
    return hmac.new(key.encode(), msg.encode(), hashlib.sha256).hexdigest()


class CommandAuth:
    """Rover-side verifier: valid HMAC and a strictly increasing seq (no replays)."""

    def __init__(self, key: str = OPERATOR_TOKEN):
        self.key = key
        self.last_seq = 0

    def sign(self, cmd: dict) -> dict:
        out = dict(cmd)
        out["seq"] = max(int(_time.time() * 1000), self.last_seq + 1)
        out["sig"] = sign(self.key, out)
        return out

    def verify(self, cmd: dict) -> CommandResult:
        sig, seq = cmd.get("sig"), cmd.get("seq")
        if not sig or seq is None:
            return CommandResult(False, "unsigned command")
        if not hmac.compare_digest(str(sig), sign(self.key, cmd)):
            return CommandResult(False, "bad signature (tampered?)")
        if int(seq) <= self.last_seq:
            return CommandResult(False, f"replayed or old seq {seq} (last {self.last_seq})")
        self.last_seq = int(seq)
        return CommandResult(True)
