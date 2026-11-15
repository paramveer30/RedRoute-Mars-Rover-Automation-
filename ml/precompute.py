"""Precompute model tile probabilities and calibration logits.

Run ``python -m ml.precompute`` from the repository root. Calibration is fitted
elsewhere on the fixed validation split; this command never fits a temperature
or changes the model from test results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ml.ai4mars import cache_train_subset, clean_label, load_test, tile_id_for, train_val_split
from ml.model import INPUT_SIZE, load_weights, preprocess
from world.paths import CALIBRATION, LOGITS_DIR, PREDS_NPZ, TILES_NPZ, USABLE_TILES, WEIGHTS
from world.warp import CELLS, warp_probs


LOGIT_SIZE = 64


def resolve_temperature(explicit: float | None = None) -> float:
    """Explicit CLI value wins; otherwise use calibration.json or T=1."""
    if explicit is not None:
        temperature = float(explicit)
    elif CALIBRATION.exists():
        temperature = float(json.loads(CALIBRATION.read_text(encoding="utf-8"))["T"])
    else:
        temperature = 1.0
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and greater than zero")
    return temperature


def atomic_savez(path: Path, **arrays) -> None:
    """Never expose a partially written archive to a running observer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def cleaned_label64(row) -> np.ndarray:
    """Match the cleaned native mask to the 64px raw-logit lattice."""
    label = clean_label(row["label_mask"], row["rover_mask"], row["range_mask"])
    return cv2.resize(label, (LOGIT_SIZE, LOGIT_SIZE), interpolation=cv2.INTER_NEAREST)


def infer_split(model, dataset, device, batch_size: int, *, matrices=None, temperature=1.0):
    """Return uncalibrated 64px logits/labels and optional warped probabilities."""
    count = len(dataset)
    raw_logits = np.empty((count, 4, LOGIT_SIZE, LOGIT_SIZE), np.float16)
    labels = np.empty((count, LOGIT_SIZE, LOGIT_SIZE), np.uint8)
    tile_probs = np.empty((count, CELLS, CELLS, 4), np.float16) if matrices is not None else None
    if matrices is not None and np.asarray(matrices).shape != (count, 3, 3):
        raise ValueError("homographies must match dataset rows exactly")
    with torch.inference_mode():
        for start in range(0, count, batch_size):
            end = min(start + batch_size, count)
            records = [dataset[index] for index in range(start, end)]
            inputs = torch.stack([preprocess(np.asarray(row["image"], np.uint8)) for row in records]).to(device)
            logits = model(inputs).float()
            if tuple(logits.shape) != (end - start, 4, INPUT_SIZE, INPUT_SIZE) or not torch.isfinite(logits).all():
                raise ValueError("model produced invalid logits")
            reduced = F.interpolate(logits, size=(LOGIT_SIZE, LOGIT_SIZE), mode="bilinear", align_corners=False)
            reduced_np = reduced.cpu().numpy().astype(np.float16)
            if not np.isfinite(reduced_np).all():
                raise ValueError("raw logits cannot be represented in float16")
            raw_logits[start:end] = reduced_np
            labels[start:end] = np.stack([cleaned_label64(row) for row in records])
            if matrices is not None:
                probabilities = torch.softmax(logits / temperature, dim=1).permute(0, 2, 3, 1).cpu().numpy()
                for offset, probability in enumerate(probabilities):
                    index = start + offset
                    tile_probs[index] = warp_probs(probability, matrices[index]).astype(np.float16)
            if end == count or start // batch_size % 5 == 0:
                print(f"precomputed {end}/{count}", flush=True)
    return raw_logits, labels, tile_probs


def verify_probabilities(probabilities: np.ndarray, count: int) -> float:
    if probabilities.shape != (count, CELLS, CELLS, 4) or probabilities.dtype != np.float16:
        raise ValueError("prediction cache must be (N,20,20,4) float16")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("prediction cache contains invalid probabilities")
    max_error = float(np.max(np.abs(probabilities.astype(np.float32).sum(axis=-1) - 1)))
    if max_error > 1e-2:
        raise ValueError(f"prediction cache is not normalized: max sum error {max_error}")
    return max_error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("cuda", "cpu"), default=None)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1:
        parser.error("batch-size and threads must be positive")
    temperature = resolve_temperature(args.temperature)
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(1)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    started = time.perf_counter()
    weights_hash = hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()
    metadata = {"input_size": np.int64(INPUT_SIZE), "logit_size": np.int64(LOGIT_SIZE),
                "weights_sha256": np.array(weights_hash)}
    dataset = load_test()
    with np.load(TILES_NPZ, allow_pickle=False) as archive:
        ids = archive["ids"].copy()
        matrices = archive["H"].copy()
        truth_labels = archive["label20"].copy()
    expected_ids = np.array([tile_id_for(index) for index in range(len(dataset))])
    if len(dataset) != 322 or not np.array_equal(ids, expected_ids):
        raise ValueError("test rows must exactly match all 322 archived tile ids")
    if matrices.shape != (len(dataset), 3, 3) or not np.isfinite(matrices).all():
        raise ValueError("tile archive has invalid homographies")
    model = load_weights(device=device)
    print(f"device={device}; temperature={temperature}; input_size={INPUT_SIZE}", flush=True)
    raw, labels, probabilities = infer_split(model, dataset, device, args.batch_size,
                                            matrices=matrices, temperature=temperature)
    max_error = verify_probabilities(probabilities, len(dataset))
    atomic_savez(LOGITS_DIR / "test_logits.npz", logits=raw, labels=labels, ids=ids,
                 split=np.array("test_min3"), **metadata)
    usable_ids = json.loads(USABLE_TILES.read_text(encoding="utf-8"))
    if not usable_ids or not set(usable_ids).issubset(set(ids.tolist())):
        raise ValueError("usable ids must be a nonempty subset of archived tile ids")
    usable = np.isin(ids, usable_ids)
    valid = (truth_labels != 255) & usable[:, None, None]
    if not valid.any():
        raise ValueError("usable tile library has no labelled cells")
    correct = int((probabilities.argmax(axis=-1)[valid] == truth_labels[valid]).sum())
    labelled_cells = int(valid.sum())
    agreement = correct / labelled_cells
    atomic_savez(PREDS_NPZ, ids=ids, probs=probabilities, temperature=np.float64(temperature),
                 tile_agreement=np.float64(agreement), labelled_cells=np.int64(labelled_cells), **metadata)
    print(f"Prediction cache ready: {PREDS_NPZ}", flush=True)
    print(f"Usable tile agreement: {correct}/{labelled_cells} = {agreement:.6%}; max sum error {max_error:.8f}", flush=True)
    _, validation = train_val_split(cache_train_subset(6500), 500)
    val_raw, val_labels, _ = infer_split(model, validation, device, args.batch_size)
    atomic_savez(LOGITS_DIR / "val_logits.npz", logits=val_raw, labels=val_labels,
                 split=np.array("validation"), fingerprint=np.array(validation._fingerprint),
                 validation_seed=np.int64(0), **metadata)
    paths = [PREDS_NPZ, LOGITS_DIR / "test_logits.npz", LOGITS_DIR / "val_logits.npz"]
    print(json.dumps({"temperature": temperature, "tile_agreement": agreement,
                      "usable_tiles": int(usable.sum()), "labelled_cells": labelled_cells,
                      "max_probability_sum_error": max_error,
                      "val_fingerprint": validation._fingerprint,
                      "seconds": time.perf_counter() - started,
                      "file_bytes": {str(path): path.stat().st_size for path in paths}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
