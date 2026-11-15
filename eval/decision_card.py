"""Route decision card: why the chosen route differs from the geometry-shortest one.

Belief-only: reads the belief and the cost map, never ``world.truth``.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

from backend.app.contracts import BIG_ROCK, RES_M, SAND, Belief, Cell, CostConfig, World
from backend.app.mapping import cost_grid

CLASS_NAMES = {0: "soil", 1: "bedrock", 2: "sand", 3: "big rock"}
HAZARD_CLASSES = (SAND, BIG_ROCK)


def path_length_m(path) -> float:
    """Length in metres: 0.25 m per axis step, sqrt(2)*0.25 m per diagonal."""
    total = 0.0
    for (r0, c0), (r1, c1) in zip(path[:-1], path[1:]):
        total += math.hypot(r1 - r0, c1 - c0) * RES_M
    return float(total)


def _dominant_class(classes: np.ndarray) -> int:
    """Dominant avoided class.

    Rule: if any avoided cell is a hazard class (sand, big rock), report the
    hazard class with the most cells (ties go to sand); otherwise report the
    most common of soil/bedrock.
    """
    counts = {k: int((classes == k).sum()) for k in CLASS_NAMES}
    hazards = {k: counts[k] for k in HAZARD_CLASSES if counts[k] > 0}
    pool = hazards or {k: v for k, v in counts.items() if v > 0}
    return max(pool, key=lambda k: (pool[k], k == SAND))


def _summary(extra_m, extra_pct, avoided, shortest_found) -> str:
    if not shortest_found:
        return "No geometry-only route exists to compare against."
    if extra_m < 0.05:
        return "Chosen route matches the shortest route; nothing needed avoiding."
    lead = f"Took a {extra_m:.1f} m ({extra_pct:.0f}%) longer route"
    if avoided["class"] is None:
        return lead + " to keep clear of uncertain or blocked terrain."
    n = avoided["cells"]
    return (f"{lead} to avoid {n} cell{'s' if n != 1 else ''} of predicted "
            f"{avoided['class']} (mean P={avoided['mean_p']:.2f}).")


def decision_card(world: World, belief: Belief, cfg: CostConfig, start: Cell, goal: Cell,
                  chosen_path: list[Cell], planner=None) -> dict:
    if planner is None:
        from backend.app.planner import plan as planner  # lazy: Stream B's module
    chosen = [(int(c[0]), int(c[1])) for c in chosen_path]
    chosen_len = path_length_m(chosen)
    chosen_max_unc = float(max(belief.unc[r, c] for r, c in chosen)) if chosen else 0.0

    geo_cfg = dataclasses.replace(cfg, use_semantics=False, w_u=0)
    cost, blocked = cost_grid(world, belief, geo_cfg)
    shortest = planner(cost, blocked, (int(start[0]), int(start[1])), (int(goal[0]), int(goal[1])))

    avoided = {"class": None, "cells": 0, "mean_p": 0.0, "mean_conf": 0.0}
    if not shortest:
        return {"chosen_len_m": round(chosen_len, 3), "shortest_len_m": None,
                "extra_m": None, "extra_pct": None, "avoided": avoided,
                "chosen_max_unc": chosen_max_unc, "summary": _summary(0, 0, avoided, False)}

    shortest = [(int(c[0]), int(c[1])) for c in shortest]
    shortest_len = path_length_m(shortest)
    extra_m = chosen_len - shortest_len
    if abs(extra_m) < 1e-6:
        extra_m = 0.0  # avoid -0.0
    extra_pct = 100.0 * extra_m / shortest_len if shortest_len > 0 else 0.0

    on_chosen = set(chosen)
    cells = [] if extra_m < 0.05 else [
        c for c in dict.fromkeys(shortest) if c not in on_chosen and belief.known[c]]
    if cells:
        rr = np.array([c[0] for c in cells])
        cc = np.array([c[1] for c in cells])
        probs = belief.probs[rr, cc]
        argmax = probs.argmax(axis=1)
        cls = _dominant_class(argmax)
        sel = argmax == cls
        avoided = {"class": CLASS_NAMES[cls], "cells": int(sel.sum()),
                   "mean_p": float(probs[sel, cls].mean()),
                   "mean_conf": float((1.0 - belief.unc[rr[sel], cc[sel]]).mean())}

    return {"chosen_len_m": round(chosen_len, 3), "shortest_len_m": round(shortest_len, 3),
            "extra_m": round(extra_m, 3), "extra_pct": round(extra_pct, 2),
            "avoided": avoided, "chosen_max_unc": chosen_max_unc,
            "summary": _summary(extra_m, extra_pct, avoided, True)}
