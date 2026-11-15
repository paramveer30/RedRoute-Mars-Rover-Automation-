"""Measure how the live blur fault changes model uncertainty (CPU only).

Two studies, both on real weights and the cached AI4Mars test split:

1. ``tiles``: for every usable tile and blur level, the mean normalized
   entropy U of the warped 20x20 cells (what the belief stores), the mean
   per-pixel entropy inside the warped ground patch and over the whole
   image, mean max-probability, agreement with ``label20`` and the fraction
   of cells whose argmax changes against the clear prediction.
2. ``rover``: the actual observe() sequence. For each demo seed the rover
   drives north from its start; run A sees clear frames for every tick, run B
   sees the same clear frames and then the blur fault for the last ticks.
   Per tile seen during the blur phase, fused U at the end of B is compared
   with A.

Clear and blurred frames are both produced by the live CPU path at the same
temperature, so cache-vs-live differences (fp16 GPU, or a cache built at a
different T) cannot masquerade as a blur effect.

Usage: python -m ml.blur_study [--temperature T] [--seeds 20] [--out run.json]
eval/blur_study.json holds the T=1.0 and T=1.5187 runs side by side.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np

LEVELS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def _entropy(p: np.ndarray) -> np.ndarray:
    return -np.sum(p * np.log(np.maximum(p, 1e-12)), axis=-1) / np.log(4)


def tile_study(levels=LEVELS) -> dict:
    from ml import infer
    from world.paths import TILES_NPZ, USABLE_TILES
    from world.warp import OUT_PX, _matrix_for_shape, warp_probs

    usable = sorted(json.loads(Path(USABLE_TILES).read_text(encoding="utf-8")))
    with np.load(TILES_NPZ, allow_pickle=False) as data:
        lookup = {str(s): i for i, s in enumerate(data["ids"])}
        H, label20 = data["H"], data["label20"]
    rows = []
    for source in usable:
        h = H[lookup[source]]
        labels = label20[lookup[source]]
        valid = labels <= 3
        # Inverse-warp a ones image to find which model pixels feed the patch.
        support = cv2.warpPerspective(np.ones((OUT_PX, OUT_PX), np.float32),
                                      _matrix_for_shape(h, (256, 256)), (256, 256),
                                      flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP) > 0
        clear_arg = None
        row = {"id": source}
        for level in levels:
            probs = infer.predict_image(source, blur=level)
            cells = warp_probs(probs, h)
            arg = cells.argmax(-1)
            if clear_arg is None:
                clear_arg = arg
            pix_u = _entropy(probs)
            row[f"{level:.1f}"] = {
                "cell_u": float(_entropy(cells).mean()),
                "patch_pixel_u": float(pix_u[support].mean()),
                "image_pixel_u": float(pix_u.mean()),
                "cell_maxp": float(cells.max(-1).mean()),
                "acc": float((arg[valid] == labels[valid]).mean()) if valid.any() else None,
                "argmax_changed": float((arg != clear_arg).mean()),
            }
        rows.append(row)
    summary = {}
    for level in levels:
        key = f"{level:.1f}"
        out = {}
        for metric in ("cell_u", "patch_pixel_u", "image_pixel_u", "cell_maxp", "acc", "argmax_changed"):
            values = np.array([r[key][metric] for r in rows if r[key][metric] is not None])
            out[metric] = {"mean": float(values.mean()), "median": float(np.median(values))}
        if level:
            up = [r[key]["cell_u"] > r["0.0"]["cell_u"] for r in rows]
            out["frac_tiles_cell_u_up"] = float(np.mean(up))
        summary[key] = out
    return {"n_tiles": len(rows), "summary": summary, "tiles": rows}


def rover_study(seeds=(1, 2, 3, 4, 5), clear_ticks=6, blur_ticks=4, blur=0.8,
                step_cells=4) -> dict:
    """Clear-then-blur rover drive vs. the all-clear drive on the same poses."""
    from backend.app import perception
    from backend.app.contracts import TILE, Belief, ObserveFlags, Pose, SensorConfig
    from world.builder import build_world

    from ml import infer

    class LiveClear(dict):
        """Clear frames from the same live CPU path and temperature as blur."""
        def __missing__(self, source):
            self[source] = infer.predict_tile(source)
            return self[source]

    live_clear = LiveClear()
    original = perception._predictions
    perception._predictions = lambda: live_clear
    try:
        sensor = SensorConfig()
        per_tile = []
        for seed in seeds:
            world = build_world(seed)
            start = Pose(190, 10 + 20 * (seed % 5), 0)
            poses = [Pose(start.row - step_cells * t, start.col, 0)
                     for t in range(clear_ticks + blur_ticks)]
            clear, mixed = Belief.empty(), Belief.empty()
            blur_cells = np.zeros((200, 200), bool)
            clear_cells = np.zeros((200, 200), bool)
            for t, pose in enumerate(poses):
                assert perception.observe(world, clear, pose, sensor, mode="model").ok
                flags = ObserveFlags(blur=blur) if t >= clear_ticks else ObserveFlags()
                assert perception.observe(world, mixed, pose, sensor, flags, mode="model").ok
                cone = perception.sensor_cone(pose, sensor)
                if t >= clear_ticks:
                    blur_cells |= cone
                else:
                    clear_cells |= cone
            for tr in range(10):
                for tc in range(10):
                    region = np.s_[tr*TILE:(tr+1)*TILE, tc*TILE:(tc+1)*TILE]
                    mask = blur_cells[region]
                    if mask.sum() < 20:  # ignore barely clipped tiles
                        continue
                    # Most blurred cells were already fused from clear frames.
                    seen_clear = bool(clear_cells[region][mask].mean() > 0.5)
                    per_tile.append({
                        "seed": seed, "tile": [tr, tc], "source": world.tile_image[tr][tc],
                        "cells": int(mask.sum()), "had_clear_history": seen_clear,
                        "u_clear": float(clear.unc[region][mask].mean()),
                        "u_blur_phase": float(mixed.unc[region][mask].mean()),
                    })
    finally:
        perception._predictions = original
    u_up = np.array([t["u_blur_phase"] > t["u_clear"] for t in per_tile])
    hist = np.array([t["had_clear_history"] for t in per_tile])
    ratio = np.array([t["u_blur_phase"] / max(t["u_clear"], 1e-12) for t in per_tile])
    return {
        "config": {"seeds": list(seeds), "clear_ticks": clear_ticks, "blur_ticks": blur_ticks,
                   "blur": blur, "step_cells": step_cells},
        "n_tiles": len(per_tile),
        "frac_u_up": float(u_up.mean()),
        "frac_u_up_with_clear_history": float(u_up[hist].mean()) if hist.any() else None,
        "n_with_clear_history": int(hist.sum()),
        "frac_u_up_without_clear_history": float(u_up[~hist].mean()) if (~hist).any() else None,
        "median_u_ratio": float(np.median(ratio)),
        "mean_u_clear": float(np.mean([t["u_clear"] for t in per_tile])),
        "mean_u_blur_phase": float(np.mean([t["u_blur_phase"] for t in per_tile])),
        "tiles": per_tile,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    parser.add_argument("--skip-tiles", action="store_true")
    parser.add_argument("--seeds", type=int, default=20, help="rover study seeds 1..N")
    parser.add_argument("--temperature", type=float, default=None,
                        help="override softmax temperature (default: ml/calibration.json or 1)")
    args = parser.parse_args()
    from ml import infer
    if args.temperature is not None:
        infer.temperature = lambda: args.temperature
    started = time.perf_counter()
    result = {"temperature": infer.temperature()}
    if not args.skip_tiles:
        result["tiles"] = tile_study()
        print(json.dumps(result["tiles"]["summary"], indent=1))
    result["rover"] = rover_study(seeds=tuple(range(1, args.seeds + 1)))
    print(json.dumps({k: v for k, v in result["rover"].items() if k != "tiles"}, indent=1))
    result["seconds"] = time.perf_counter() - started
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
