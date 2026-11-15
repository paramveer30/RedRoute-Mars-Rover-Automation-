"""Earth<->rover comms link simulator (Stream B owns). See docs/STREAM_B.md, hours 10-12, 15-17.

delay with time compression, blackout, onboard buffer, replay ordered by mission_time,
stale-command rejection via route_version precondition, HMAC + seq numbers (P1).
Cached real light delay: terrain/light_delay.json (scripts/fetch_light_delay.py).
"""
