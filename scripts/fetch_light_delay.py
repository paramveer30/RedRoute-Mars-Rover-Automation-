"""Fetch today's Earth->Mars one-way light time from JPL Horizons and cache it to
terrain/light_delay.json so the demo works offline.   (no API key needed)

    .venv/Scripts/python scripts/fetch_light_delay.py
"""
from __future__ import annotations

import datetime as dt
import json
import pathlib
import re

import httpx

OUT = pathlib.Path(__file__).resolve().parents[1] / "terrain" / "light_delay.json"


def fetch(day: dt.date) -> dict:
    params = {
        "format": "json", "COMMAND": "'499'", "CENTER": "'500@399'", "EPHEM_TYPE": "'OBSERVER'",
        "QUANTITIES": "'21'", "START_TIME": f"'{day}'", "STOP_TIME": f"'{day + dt.timedelta(days=1)}'",
        "STEP_SIZE": "'1d'",
    }
    r = httpx.get("https://ssd.jpl.nasa.gov/api/horizons.api", params=params, timeout=30)
    r.raise_for_status()
    text = r.json()["result"]
    body = text.split("$$SOE")[1].split("$$EOE")[0].strip()
    first = body.splitlines()[0]
    minutes = float(re.findall(r"[-+]?\d+\.\d+", first)[-1])  # 1-way light time, minutes
    return {"date": str(day), "one_way_light_time_min": minutes,
            "one_way_light_time_s": round(minutes * 60, 1), "source": "JPL Horizons",
            "raw_line": first, "fetched_utc": dt.datetime.now(dt.timezone.utc).isoformat()}


if __name__ == "__main__":
    data = fetch(dt.date.today())
    OUT.write_text(json.dumps(data, indent=2))
    print(json.dumps(data, indent=2))
