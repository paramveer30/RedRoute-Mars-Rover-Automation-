"""HTTP-level checks: roles, signed commands through the delayed link, frames."""
from fastapi.testclient import TestClient

from backend.app import api
from backend.app.link import OPERATOR_TOKEN, VIEWER_TOKEN, sign

client = TestClient(api.app)
OP = {"Authorization": f"Bearer {OPERATOR_TOKEN}"}


def signed(m, **cmd):
    cmd["seq"] = m.link.auth.last_seq + 1000
    cmd["sig"] = sign(OPERATOR_TOKEN, cmd)
    return cmd


def drain(m, ticks=40):
    events = []
    for _ in range(ticks):
        _, payload = api.tick(m)
        events += payload["events"]
    return events


def test_viewer_and_anonymous_cannot_command():
    m = api.get_mission("api-roles")
    body = signed(m, type="stop")
    assert client.post("/api/missions/api-roles/command", json=body).status_code == 401
    r = client.post("/api/missions/api-roles/command", json=body,
                    headers={"Authorization": f"Bearer {VIEWER_TOKEN}"})
    assert r.status_code == 403
    assert client.get("/api/session", headers=OP).json() == {"role": "operator"}


def test_signed_target_is_applied_after_link_delay():
    m = api.get_mission("api-ok")
    m.link.compression = 600   # ~1.1 s
    r = client.post("/api/missions/api-ok/command", json=signed(m, type="target", x=60, y=150),
                    headers=OP)
    assert r.status_code == 200 and r.json()["queued"]
    assert m.rover.goal is None                 # not yet: still in flight
    drain(m, 4)
    assert m.rover.goal == (150, 60)


def test_tampered_command_rejected_on_rover():
    m = api.get_mission("api-tamper")
    m.link.compression = 600
    body = signed(m, type="target", x=60, y=150) | {"tamper": True}
    assert client.post("/api/missions/api-tamper/command", json=body, headers=OP).status_code == 200
    events = drain(m, 4)
    assert any("bad signature" in e["msg"] for e in events)
    assert m.rover.goal is None


def test_replayed_command_rejected_on_rover():
    m = api.get_mission("api-replay")
    m.link.compression = 600
    body = signed(m, type="target", x=60, y=150)
    client.post("/api/missions/api-replay/command", json=body, headers=OP)
    drain(m, 4)
    client.post("/api/missions/api-replay/command", json=body, headers=OP)
    events = drain(m, 4)
    assert any("replayed" in e["msg"] for e in events)


def test_frame_and_overlay_served_and_ids_validated():
    assert client.get("/api/frames/m3_0001").headers["content-type"] == "image/jpeg"
    assert client.get("/api/overlays/m3_0001").headers["content-type"] == "image/png"
    assert client.get("/api/frames/..%2Fsecret").status_code in (400, 404)
    assert client.get("/api/frames/m3_9999").status_code == 404
