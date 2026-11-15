"""Aggregate scored missions into results.json / results.md.

Every number comes from the harness records; nothing is edited or rounded
before it is stored (markdown only formats for display).
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np

from eval.metrics import bootstrap_ci
from world.paths import REPO

FAKE_BANNER = ("FAKE ROVER TEST DOUBLE: these numbers come from eval.rover_api.FakeRover, "
               "not Stream B's rover. They are NOT RedRoute performance.")


def git_sha() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO,
                               capture_output=True, text=True, check=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def _pct(values, q):
    return float(np.percentile(values, q)) if len(values) else None


def summarize_config(records: list[dict], n_boot: int = 10_000, boot_seed: int = 0) -> dict:
    ok = [r for r in records if r.get("error") is None]
    success = [1.0 if r.get("success") else 0.0 for r in records]  # errors count as failures
    entries = [float(r["hazard_entries"]) for r in ok]
    ratios = [r["route_ratio"] for r in ok if r.get("route_ratio") is not None]
    lat = [x for r in ok for x in r["replan_latencies_ms"]]
    return {
        "n": len(records),
        "n_errors": len(records) - len(ok),
        "success_rate": float(np.mean(success)) if success else None,
        "success_rate_ci95": bootstrap_ci(success, n_boot, boot_seed) if success else None,
        "hazard_entries_mean": float(np.mean(entries)) if entries else None,
        "hazard_entries_ci95": bootstrap_ci(entries, n_boot, boot_seed) if entries else None,
        "reached_rate": float(np.mean([r["reached"] for r in ok])) if ok else None,
        "route_ratio_mean": float(np.mean(ratios)) if ratios else None,
        "route_ratio_n": len(ratios),
        "replans_mean": float(np.mean([r["replans"] for r in ok])) if ok else None,
        "replan_latency_ms_p50": _pct(lat, 50),
        "replan_latency_ms_p95": _pct(lat, 95),
        "reported_reached_mismatch": sum(r["reached"] != r["reported_reached"] for r in ok),
    }


def summarize(records: list[dict], configs: list[str], meta: dict) -> dict:
    by_cfg = {c: [r for r in records if r["config"] == c] for c in configs}
    obs = [r.get("observe_calls") for r in records if r.get("error") is None]
    total_obs = sum(obs) if obs and all(o is not None for o in obs) else None
    wall = meta["wall_s"]
    return {
        "rover": meta["rover"],
        "fake": meta["rover"] == "fake",
        "note": FAKE_BANNER if meta["rover"] == "fake" else "",
        "git_sha": git_sha(),
        "seeds": meta["seeds"],
        "scenarios": meta["scenarios"],
        "configs": {c: summarize_config(rs) for c, rs in by_cfg.items()},
        "throughput": {
            "wall_s": wall,
            "workers": meta["workers"],
            "missions": len(records),
            "missions_per_min": len(records) / wall * 60 if wall > 0 else None,
            "observe_calls": total_obs,
            "observe_calls_per_s": total_obs / wall if wall > 0 and total_obs is not None else None,
        },
        "definitions": meta["definitions"],
        "conditions": meta.get("conditions", {}),
        "missions": records,
    }


def _fmt(x, nd=2):
    return "n/a" if x is None else f"{x:.{nd}f}"


def _ci(ci, nd=2):
    return "" if ci is None else f" [{ci[0]:.{nd}f}, {ci[1]:.{nd}f}]"


def to_markdown(summary: dict) -> str:
    lines = []
    if summary["fake"]:
        lines += [f"> **{FAKE_BANNER}**", ""]
    lines += ["# Mission evaluation", "",
              f"Rover: `{summary['rover']}` | git `{summary['git_sha']}` | "
              f"seeds {summary['seeds'][0]}..{summary['seeds'][-1]} (n={len(summary['seeds'])})", "",
              "| Config | Success rate [95% CI] | Hazard entries / mission [95% CI] | "
              "Route ratio (reached) | Replan p50 / p95 ms | n (errors) |",
              "| --- | --- | --- | --- | --- | --- |"]
    for name, s in summary["configs"].items():
        lines.append(
            f"| {name} | {_fmt(s['success_rate'])}{_ci(s['success_rate_ci95'])} | "
            f"{_fmt(s['hazard_entries_mean'])}{_ci(s['hazard_entries_ci95'])} | "
            f"{_fmt(s['route_ratio_mean'], 3)} (n={s['route_ratio_n']}) | "
            f"{_fmt(s['replan_latency_ms_p50'], 1)} / {_fmt(s['replan_latency_ms_p95'], 1)} | "
            f"{s['n']} ({s['n_errors']}) |")
    t = summary["throughput"]
    lines += ["", f"Throughput: {t['missions']} missions in {t['wall_s']:.1f} s on {t['workers']} "
              f"worker(s) = {_fmt(t['missions_per_min'], 1)} missions/min, "
              f"{_fmt(t['observe_calls_per_s'], 1)} observe() calls/s.", "",
              "## Definitions", ""]
    lines += [f"- **{k}**: {v}" for k, v in summary["definitions"].items()]
    cond = summary.get("conditions") or {}
    if cond:
        cache = cond.get("prediction_cache", {})
        lines += ["", "## Conditions", "",
                  f"- **prediction cache**: `{cache.get('path')}`, temperature "
                  f"{cache.get('temperature', 'n/a')}, weights sha256 `{cache.get('weights_sha256', 'n/a')}`",
                  f"- **cost map blocking**: {cond.get('sand_blocking', 'n/a')}"]
    lines.append("")
    return "\n".join(lines)


def write(summary: dict, out_dir: Path) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    js, md = out_dir / "results.json", out_dir / "results.md"
    js.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md.write_text(to_markdown(summary), encoding="utf-8")
    return js, md
