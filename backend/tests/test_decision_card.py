"""Decision card: synthetic sand band, injected Dijkstra planner."""
import heapq
import json
import math

import pytest

from backend.app.contracts import GRID, Belief, CostConfig
from backend.app.mapping import cost_grid
from eval.decision_card import decision_card, path_length_m
from world.stub_world import build_stub_world


def dijkstra(cost, blocked, start, goal):
    dist = {start: 0.0}
    prev = {}
    pq = [(0.0, start)]
    while pq:
        d, u = heapq.heappop(pq)
        if u == goal:
            path = [u]
            while u in prev:
                u = prev[u]
                path.append(u)
            return path[::-1]
        if d > dist.get(u, 1e18):
            continue
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                v = (u[0] + dr, u[1] + dc)
                if (dr, dc) == (0, 0) or not (0 <= v[0] < GRID and 0 <= v[1] < GRID) or blocked[v]:
                    continue
                nd = d + math.hypot(dr, dc) * float(cost[v])
                if nd < dist.get(v, 1e18):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))
    return None


@pytest.fixture
def scene():
    world = build_stub_world(0)
    world.slope_deg.fill(0)
    world.roughness.fill(0)
    world.geo_obstacle.fill(False)
    belief = Belief.empty()
    belief.known.fill(True)
    belief.probs[:] = [0.9, 0.05, 0.03, 0.02]
    belief.unc[:] = 0.1
    belief.probs[95:106, 80:100] = [0.05, 0.02, 0.9, 0.03]  # sand band
    world.truth = None  # the card must be belief-only
    return world, belief, CostConfig(inflate_m=0)


START, GOAL = (100, 50), (100, 150)


def test_sand_band_avoided(scene):
    world, belief, cfg = scene
    chosen = dijkstra(*cost_grid(world, belief, cfg), START, GOAL)
    card = decision_card(world, belief, cfg, START, GOAL, chosen, planner=dijkstra)
    print(json.dumps(card, indent=1))
    assert card["avoided"]["class"] == "sand"
    assert card["avoided"]["cells"] > 0
    assert card["extra_m"] > 0
    assert card["avoided"]["mean_p"] == pytest.approx(0.9, abs=1e-5)
    assert card["avoided"]["mean_conf"] == pytest.approx(0.9, abs=1e-5)
    assert card["chosen_max_unc"] == pytest.approx(0.1, abs=1e-5)
    s = card["summary"]
    assert s.startswith("Took a ") and "longer route to avoid" in s
    assert f"{card['avoided']['cells']} cells of predicted sand (mean P=0.90)" in s
    json.dumps(card)


def test_identical_paths(scene):
    world, belief, cfg = scene
    belief.probs[95:106, 80:100] = [0.9, 0.05, 0.03, 0.02]
    same = dijkstra(*cost_grid(world, belief, cfg), START, GOAL)
    card = decision_card(world, belief, cfg, START, GOAL, same, planner=dijkstra)
    assert card["extra_m"] == 0
    assert card["avoided"]["cells"] == 0 and card["avoided"]["class"] is None
    assert "matches the shortest" in card["summary"]
    json.dumps(card)


def test_no_shortest_path(scene):
    world, belief, cfg = scene
    card = decision_card(world, belief, cfg, START, GOAL, [START, GOAL], planner=lambda *a: None)
    assert card["shortest_len_m"] is None
    json.dumps(card)


def test_path_length():
    assert path_length_m([(0, 0), (0, 1), (1, 2)]) == pytest.approx(0.25 + 0.25 * math.sqrt(2))


def test_sand_band_real_planner(scene):
    from backend.app.planner import plan
    world, belief, cfg = scene
    chosen = plan(*cost_grid(world, belief, cfg), START, GOAL)
    card = decision_card(world, belief, cfg, START, GOAL, chosen)  # default planner = real A*
    assert card["avoided"]["class"] == "sand"
    assert card["avoided"]["cells"] > 0
    assert card["extra_m"] > 0
    json.dumps(card)
