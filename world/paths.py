"""Single source of truth for file locations (Stream A). Import from here; never hardcode paths.

Two kinds of output:
  * COMMITTED (small, needed on every machine, e.g. the teammate's): under the repo.
  * ARTIFACTS (big or regenerable, gitignored): under RR_ARTIFACTS, which defaults to <repo>/artifacts.
    Git worktrees do NOT share untracked files, so every worktree must point at the same
    folder:  set RR_ARTIFACTS=C:\\Pranoy\\Machacks\\rr-artifacts   (see docs/STREAM_A.md).
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# ---- committed --------------------------------------------------------------
TERRAIN_DIR = REPO / "terrain"
DTM_CROP = TERRAIN_DIR / "jezero_crop.tif"
LIGHT_DELAY = TERRAIN_DIR / "light_delay.json"
TERRAIN_NPZ = REPO / "world" / "terrain.npz"            # A2: elevation/slope/roughness (200,200)
TILES_NPZ = REPO / "world" / "tiles.npz"                # A3: label20 (N,20,20) uint8 + ids
USABLE_TILES = REPO / "world" / "usable_tiles.json"     # A3: list of usable tile ids
DEMO_SEEDS = REPO / "world" / "DEMO_SEEDS.md"           # A4
PREDS_NPZ = REPO / "world" / "preds20.npz"              # A6: probs (N,20,20,4) float16 + ids
FRAMES_DIR = TERRAIN_DIR / "frames"                     # A11: <id>.jpg  NAVCAM frame, 384 px
OVERLAYS_DIR = TERRAIN_DIR / "overlays"                 # A11: <id>.png  class overlay, 384 px
WEIGHTS = REPO / "ml" / "weights" / "unet_r34_fp16.pt"  # A1: deployable weights (<50 MB)
CALIBRATION = REPO / "ml" / "calibration.json"          # A10: {"T": ...}
SEG_METRICS = REPO / "eval" / "seg_metrics.json"        # A5
EVAL_RESULTS = REPO / "eval" / "results.json"           # A9

# ---- artifacts (gitignored) -------------------------------------------------
ARTIFACTS = Path(os.environ.get("RR_ARTIFACTS", REPO / "artifacts"))
DATA_DIR = ARTIFACTS / "data"            # local copy of the AI4Mars train subset
CKPT_DIR = ARTIFACTS / "checkpoints"     # full training checkpoints
LOGITS_DIR = ARTIFACTS / "logits"        # A6: val/test logits for calibration
TILE_IMG_DIR = ARTIFACTS / "tiles_img"   # A3: warped 160 px top-down image patches
RUNS_DIR = ARTIFACTS / "runs"            # logs, plots, contact sheets


def ensure_dirs() -> None:
    for d in (DATA_DIR, CKPT_DIR, LOGITS_DIR, TILE_IMG_DIR, RUNS_DIR, FRAMES_DIR, OVERLAYS_DIR,
              WEIGHTS.parent):
        d.mkdir(parents=True, exist_ok=True)
