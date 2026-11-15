"""Mission evaluation harness (EVAL ONLY: scenario generation and scoring read truth).

    python -m eval.harness --rover fake --n 4 --workers 2      # test double, artifacts only
    python -m eval.harness --rover real --n 100 --configs all  # Stream B rover -> eval/results.*

Scenarios depend only on the seed, so every config sees the identical world,
start and goal. ``--rover fake`` results are written under
``$RR_ARTIFACTS/runs/eval_fake`` (or ``--out``) and refuse to go to ``eval/``,
so fake numbers can never be mistaken for real results.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

from backend.app.contracts import RES_M, Cell, CostConfig, SensorConfig
from eval import metrics
from eval.rover_api import as_mission_result
from world.paths import EVAL_RESULTS, PREDS_NPZ, REPO, RUNS_DIR

CONFIGS: dict[str, tuple[str, CostConfig]] = {
    "geometry_only": ("geometry_only", CostConfig(use_semantics=False)),
    "redroute_no_u": ("model", CostConfig(w_u=0.0)),
    "redroute": ("model", CostConfig()),
    "oracle": ("oracle", CostConfig()),
}
MIN_SEPARATION_M = 30.0
SCENARIO_SALT = 9001          # keeps scenario RNG independent of build_world(seed)'s RNG
MAX_ATTEMPTS = 200
DEFAULT_MAX_STEPS = 2000      # matches Stream B's run_to_completion default
FAKE_OUT = RUNS_DIR / "eval_fake"
REAL_OUT = EVAL_RESULTS.parent

DEFINITIONS = {
    "hazard cell": f"truth sand, truth big rock, or slope > {metrics.HAZARD_SLOPE_DEG:g} deg; "
                   "truth 255 is not a hazard",
    "hazard entry": f"footprint (disk radius {metrics.FOOTPRINT_CELLS} cells = "
                    f"{metrics.FOOTPRINT_CELLS * RES_M} m) overlaps a hazard cell; the trajectory "
                    "is densified by straight-line interpolation; each contiguous run counts once",
    "success": f"final trajectory cell within {metrics.GOAL_TOL_M:g} m of goal AND 0 hazard entries",
    "route ratio": "trajectory length / shortest hazard-free 8-connected route on truth "
                   "(same footprint mask, no corner cutting); reached missions only",
    "ci95": "percentile bootstrap of the mean, 10000 resamples, seed 0",
    "scenario": f"seeded start/goal >= {MIN_SEPARATION_M:g} m apart on non-hazard truth cells, "
                "goal reachable by the safe route; identical across configs",
}


@dataclass
class Scenario:
    seed: int
    start: Cell
    goal: Cell
    optimal_m: float
    straight_m: float
    attempts: int


def make_scenario(seed: int, world=None, min_sep_m: float = MIN_SEPARATION_M) -> Scenario:
    """Deterministic start/goal for ``seed`` (depends only on the seed)."""
    from world.builder import build_world
    world = build_world(seed) if world is None else world
    mask = metrics.hazard_footprint_mask(world)
    free = np.argwhere(~mask)
    if len(free) == 0:
        raise RuntimeError(f"seed {seed}: no hazard-free cells")
    rng = np.random.default_rng([SCENARIO_SALT, seed])
    min_cells = min_sep_m / RES_M
    for attempt in range(1, MAX_ATTEMPTS + 1):
        start = tuple(int(v) for v in free[rng.integers(len(free))])
        far = free[np.hypot(*(free - start).T) >= min_cells]
        if len(far) == 0:
            continue
        goal = tuple(int(v) for v in far[rng.integers(len(far))])
        routes = metrics.safe_routes_from(mask, start)
        optimal = routes.length_to(goal)
        if math.isfinite(optimal):
            return Scenario(seed, start, goal, optimal,
                            math.hypot(goal[0] - start[0], goal[1] - start[1]) * RES_M, attempt)
    raise RuntimeError(f"seed {seed}: no reachable scenario after {MAX_ATTEMPTS} attempts")


# ---- worker side -----------------------------------------------------------------
_FACTORY = None


def _resolve_factory(rover: str):
    if rover == "fake":
        from eval.rover_api import FakeRover
        return FakeRover
    try:
        from backend.app.rover import Rover
    except ImportError as exc:
        raise SystemExit(
            "backend.app.rover.Rover does not exist yet (Stream B stub). "
            "Use --rover fake for the test double, or wait for Stream B's Rover.") from exc
    return Rover


def _init_worker(rover: str) -> None:
    """Load the expensive shared assets once per process (Windows spawns workers)."""
    global _FACTORY
    from backend.app import perception
    from world import builder
    builder._assets()
    perception._predictions()
    _FACTORY = _resolve_factory(rover)


def run_seed(task: tuple[int, list[str], int]) -> list[dict]:
    """Generate the seed's scenario, then run every requested config on it."""
    from world.builder import build_world
    seed, config_names, max_steps = task
    scenario = make_scenario(seed)
    out = []
    for name in config_names:
        mode, cfg = CONFIGS[name]
        world = build_world(seed)       # fresh copy per config: rovers may mutate it
        mask = metrics.hazard_footprint_mask(world)
        rec = {"seed": seed, "config": name, "mode": mode, "start": list(scenario.start),
               "goal": list(scenario.goal), "optimal_m": scenario.optimal_m, "error": None}
        t0 = time.perf_counter()
        try:
            rover = _FACTORY(world, mode, cfg, SensorConfig())
            raw = rover.run_to_completion(scenario.start, scenario.goal, max_steps)
            res = as_mission_result(raw)
            rec["reported_hazard_entries"] = getattr(raw, "hazard_entries", None)
            score = metrics.score_mission(world, res.trajectory, scenario.goal, scenario.optimal_m, mask)
            rec.update(score)
            rec.update({"reported_reached": res.reached, "final_state": res.final_state,
                        "reported_path_len_m": res.path_len_m, "steps": res.steps,
                        "replans": res.replans, "replan_latencies_ms": res.replan_latencies_ms,
                        "observe_calls": res.observe_calls, "trajectory_cells": len(res.trajectory)})
        except Exception:
            rec.update({"error": traceback.format_exc(limit=5), "success": False,
                        "observe_calls": 0})
        rec["mission_wall_s"] = time.perf_counter() - t0
        out.append(rec)
    return out


# ---- driver ------------------------------------------------------------------------
def run(n: int, workers: int, configs: list[str], rover: str,
        max_steps: int = DEFAULT_MAX_STEPS) -> tuple[list[dict], dict]:
    seeds = list(range(n))
    tasks = [(s, configs, max_steps) for s in seeds]
    t0 = time.perf_counter()
    if workers <= 1:
        _init_worker(rover)
        chunks = [run_seed(t) for t in tasks]
    else:
        with mp.get_context("spawn").Pool(workers, initializer=_init_worker, initargs=(rover,)) as pool:
            chunks = list(pool.imap_unordered(run_seed, tasks))
    wall = time.perf_counter() - t0
    records = sorted((r for c in chunks for r in c),
                     key=lambda r: (r["seed"], configs.index(r["config"])))
    scenarios = {}
    for r in records:
        scenarios.setdefault(r["seed"], {"start": r["start"], "goal": r["goal"],
                                         "optimal_m": r["optimal_m"]})
    meta = {"rover": rover, "seeds": seeds, "workers": workers, "wall_s": wall,
            "scenarios": scenarios, "definitions": DEFINITIONS, "conditions": conditions()}
    return records, meta


def conditions() -> dict:
    """What the numbers depend on beyond the code: prediction cache and cost rules."""
    from backend.app import mapping
    cfg = CostConfig()
    cache = {"path": Path(PREDS_NPZ).relative_to(REPO).as_posix()}
    try:
        with np.load(PREDS_NPZ, allow_pickle=False) as data:
            for key in ("temperature", "weights_sha256", "tile_agreement"):
                if key in data:
                    value = data[key].item()
                    cache[key] = value if isinstance(value, str) else float(value)
    except OSError as exc:
        cache["error"] = str(exc)
    return {
        "prediction_cache": cache,
        "sand_blocking": (f"known cells with belief P(sand) >= {mapping.P_SAND_BLOCK:g} are blocked "
                          f"(blobs < {cfg.min_blob_cells} cells dropped), inflated by "
                          f"{mapping.SAND_INFLATE_M:g} m; big rock P > {cfg.p_big_rock_block:g}, "
                          f"slope > {cfg.max_slope_deg:g} deg and sensed geometry inflated by "
                          f"{cfg.inflate_m:g} m; sub-threshold sand keeps soft cost w_s*risk"),
        "configs": {name: {"mode": mode, "cost_cfg": {k: getattr(c, k) for k in c.__dataclass_fields__}}
                    for name, (mode, c) in CONFIGS.items()},
    }


def _parse_configs(text: str) -> list[str]:
    if text == "all":
        return list(CONFIGS)
    names = [c.strip() for c in text.split(",") if c.strip()]
    unknown = [c for c in names if c not in CONFIGS]
    if unknown or not names:
        raise SystemExit(f"unknown configs {unknown}; choose from {list(CONFIGS)} or 'all'")
    return names


def main(argv: list[str] | None = None) -> dict:
    from eval import report
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=100, help="scenarios (seeds 0..n-1)")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--configs", default="all", help="'all' or comma list of " + ",".join(CONFIGS))
    p.add_argument("--rover", choices=("fake", "real"), default="real")
    p.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    p.add_argument("--out", type=Path, default=None, help="output directory for results.json/.md")
    args = p.parse_args(argv)
    if args.n < 1:
        raise SystemExit("--n must be >= 1")
    configs = _parse_configs(args.configs)
    out = args.out or (FAKE_OUT if args.rover == "fake" else REAL_OUT)
    if args.rover == "fake" and Path(out).resolve() == REAL_OUT.resolve():
        raise SystemExit("refusing to write FakeRover results into eval/: they are not real results")
    _resolve_factory(args.rover)        # fail fast, before spawning workers
    records, meta = run(args.n, args.workers, configs, args.rover, args.max_steps)
    summary = report.summarize(records, configs, meta)
    js, md = report.write(summary, out)
    print(report.to_markdown({**summary, "missions": []}))
    print(f"wrote {js} and {md}", file=sys.stderr)
    return summary


if __name__ == "__main__":
    main()
