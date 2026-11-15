"""Write eval/sample_card.json: decision card from a real Rover mission (belief only)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from backend.app.contracts import CostConfig
from backend.app.rover import Rover, State
from eval.decision_card import decision_card
from world.builder import build_world

SEED, START, GOAL = 1, (190, 10), (10, 120)


def main(mode: str = "model") -> dict:
    world = build_world(SEED)
    rover = Rover(world, mode=mode, cost_cfg=CostConfig(), start=START)
    rover.set_target(GOAL)
    cards = {}
    last_ver = 0
    for _ in range(2000):
        rover.step()
        if rover.route_version != last_ver and rover.path:
            last_ver = rover.route_version
            cards[f"route_v{last_ver}"] = {
                "pose": list(rover.cell), "known_cells": int(rover.belief.known.sum()),
                "card": decision_card(world, rover.belief, rover.cost_cfg, rover.cell, GOAL, rover.path)}
        if rover.state in (State.IDLE, State.SAFE_STOP):
            break
    out = {"seed": SEED, "mode": mode, "start": list(START), "goal": list(GOAL),
           "final_state": rover.state.value, "hazard_entries": rover.hazard_entries,
           "replans": rover.replans, "frame_image_id": rover.frame_image_id,
           "initial_route": cards.get("route_v1"),
           "final_route": cards[f"route_v{last_ver}"] if last_ver else None,
           "route_versions": len(cards),
           "full_trajectory": {
               "note": "driven trajectory vs geometry-only shortest on the final belief",
               "known_cells": int(rover.belief.known.sum()),
               "card": decision_card(world, rover.belief, rover.cost_cfg, START, GOAL, rover.trajectory)}}
    Path(__file__).with_name("sample_card.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    print(json.dumps(main(sys.argv[1] if len(sys.argv) > 1 else "model"), indent=1))
