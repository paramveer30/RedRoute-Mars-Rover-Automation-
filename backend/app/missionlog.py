"""SQLite mission log (Stream B owns). stdlib sqlite3; the file is gitignored (*.sqlite).

Tables: mission, telemetry, route (versioned), event. Written from the API tick loop;
the Rover itself stays free of persistence so A's harness can run it headlessly.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS mission (
    run_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    mission_id  TEXT NOT NULL,
    seed        INTEGER NOT NULL,
    mode        TEXT NOT NULL,
    started_utc REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS telemetry (
    run_id INTEGER NOT NULL, seq INTEGER NOT NULL, mission_time REAL NOT NULL,
    state TEXT, row REAL, col REAL, heading REAL, battery REAL, odometer_m REAL,
    route_version INTEGER, perception_mode TEXT, link_status TEXT,
    PRIMARY KEY (run_id, seq)
);
CREATE TABLE IF NOT EXISTS route (
    run_id INTEGER NOT NULL, route_version INTEGER NOT NULL, mission_time REAL NOT NULL,
    reason TEXT, latency_ms REAL, path_json TEXT NOT NULL,
    PRIMARY KEY (run_id, route_version)
);
CREATE TABLE IF NOT EXISTS event (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL, mission_time REAL NOT NULL, level TEXT, kind TEXT,
    msg TEXT, data_json TEXT
);
CREATE INDEX IF NOT EXISTS event_run ON event(run_id, mission_time);
"""


def default_path() -> Path:
    from world.paths import ARTIFACTS
    return Path(ARTIFACTS) / "missions.sqlite"


class MissionLog:
    def __init__(self, path: str | Path | None = None):
        path = Path(path) if path is not None else default_path()
        if str(path) != ":memory:":
            path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.executescript(SCHEMA)
        self._lock = threading.Lock()

    def start(self, mission_id: str, seed: int, mode: str) -> int:
        with self._lock, self._db:
            cur = self._db.execute(
                "INSERT INTO mission (mission_id, seed, mode, started_utc) VALUES (?,?,?,?)",
                (mission_id, seed, mode, time.time()))
            return int(cur.lastrowid)

    def record(self, run_id: int, rover, events) -> None:
        """Log one tick: telemetry row, any new route, and the tick's events."""
        p = rover.pose
        with self._lock, self._db:
            self._db.execute(
                "INSERT OR REPLACE INTO telemetry VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (run_id, rover.seq, rover.mission_time, rover.state.value, p.row, p.col,
                 p.heading_deg, rover.battery, rover.odometer_m, rover.route_version,
                 rover.mode, "CONNECTED" if rover.link_up else "BLACKOUT"))
            for e in events:
                if e.kind == "route":
                    self._db.execute(
                        "INSERT OR REPLACE INTO route VALUES (?,?,?,?,?,?)",
                        (run_id, e.data["route_version"], e.mission_time, e.data.get("reason"),
                         e.data.get("latency_ms"), json.dumps(rover.path)))
                self._db.execute(
                    "INSERT INTO event (run_id, mission_time, level, kind, msg, data_json) "
                    "VALUES (?,?,?,?,?,?)",
                    (run_id, e.mission_time, e.level, e.kind, e.msg,
                     json.dumps(e.data) if e.data else None))

    def query(self, sql: str, args: tuple = ()) -> list[tuple]:
        with self._lock:
            return self._db.execute(sql, args).fetchall()

    def close(self) -> None:
        self._db.close()
