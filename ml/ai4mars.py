"""Shared AI4Mars helpers (Stream A). Every task loads data through here.

Facts verified during setup (2026-11-15):
  * columns: image, label_mask, rover_mask, range_mask, has_masks, has_labels; all 1024x1024 uint8
  * labels: 0 soil, 1 bedrock, 2 sand, 3 big rock, 255 no label
  * some TRAIN rows have has_labels=False and label_mask=None -> must be skipped
  * no image-id column -> test_min3 tiles are identified by row index: tile_id_for(i) = "m3_0042"
  * Windows streaming sometimes throws WinError 10038 then retries; prefer cache_train_subset()
"""
from __future__ import annotations

import numpy as np

from world.paths import DATA_DIR

HF_NAME = "hassanjbara/AI4MARS"
IGNORE = 255
TEST_SPLIT = "test_min3"   # expert-agreed labels; NEVER train or tune on this


def tile_id_for(index: int) -> str:
    return f"m3_{index:04d}"


def index_for(tile_id: str) -> int:
    return int(tile_id.split("_")[1])


def clean_label(label: np.ndarray, rover_mask: np.ndarray | None,
                range_mask: np.ndarray | None) -> np.ndarray:
    """Return a label with rover-body pixels and beyond-30 m pixels set to IGNORE.

    Mask convention VERIFIED on test_min3: rover_mask==1 is the rover body (bottom of frame,
    18-44% of pixels); range_mask==1 is beyond 30 m (top of frame). Expert test labels are
    already 255 there; crowdsourced train labels may not be, so always call this.
    Note: 73-85% of each test image is 255 overall; only a band above the rover is labelled.
    """
    out = np.asarray(label, np.uint8).copy()
    if rover_mask is not None:
        out[np.asarray(rover_mask) > 0] = IGNORE
    if range_mask is not None:
        out[np.asarray(range_mask) > 0] = IGNORE
    out[out > 3] = IGNORE
    return out


def load_test():
    """test_min3, 322 rows, small and cached by HF after the first call."""
    from datasets import load_dataset
    return load_dataset(HF_NAME, split=TEST_SPLIT)


def cache_train_subset(n: int = 6500, seed: int = 0):
    """Download n labelled train rows once into DATA_DIR/train_<n> and reuse thereafter."""
    from datasets import load_dataset, load_from_disk
    path = DATA_DIR / f"train_{n}"
    if path.exists():
        return load_from_disk(str(path))
    ds = load_dataset(HF_NAME, split=f"train[:{int(n * 1.3)}]")
    ds = ds.filter(lambda r: bool(r["has_labels"]) and r["label_mask"] is not None)
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(path))
    return ds


def train_val_split(ds, n_val: int = 500, seed: int = 0):
    """Fixed split. VAL is used for checkpoint selection and temperature scaling (A10)."""
    s = ds.train_test_split(test_size=n_val, seed=seed)
    return s["train"], s["test"]
