"""Lazy CPU U-Net inference for real-image sensor degradation.

Blur and noise act on the source image before preprocessing. Noise uses a
source-id-seeded generator so the same fault request is reproducible. No
uncertainty is injected: callers receive the model's actual probabilities.
"""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from ml.ai4mars import index_for, load_test, tile_id_for
from ml.model import load_weights, preprocess
from world.paths import CALIBRATION, TILES_NPZ
from world.warp import warp_probs


@lru_cache(maxsize=1)
def get_model():
    """Load an eval-mode fp32 model once on CPU, without downloading weights."""
    # A small pool avoids oversubscription on the laptop during simulation.
    torch.set_num_threads(min(4, torch.get_num_threads()))
    return load_weights(device="cpu")


@lru_cache(maxsize=1)
def _dataset():
    return load_test()


@lru_cache(maxsize=1)
def _read_homographies(path: str, modified: int, size: int):
    with np.load(path, allow_pickle=False) as data:
        ids, matrices = data["ids"], data["H"]
        if matrices.shape != (len(ids), 3, 3) or not np.isfinite(matrices).all():
            raise ValueError("invalid tile homographies")
        if len(set(ids.tolist())) != len(ids):
            raise ValueError("duplicate tile homography ids")
        return {str(source): matrix for source, matrix in zip(ids, matrices)}


def _homographies():
    path = Path(TILES_NPZ)
    stat = path.stat()
    return _read_homographies(str(path), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=1)
def _read_temperature(path: str, modified: int, size: int) -> float:
    temperature = float(json.loads(Path(path).read_text(encoding="utf-8"))["T"])
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("calibration temperature must be finite and positive")
    return temperature


def temperature() -> float:
    """Use validation-fitted calibration if present, otherwise T=1."""
    path = Path(CALIBRATION)
    if not path.exists():
        return 1.0
    stat = path.stat()
    return _read_temperature(str(path), stat.st_mtime_ns, stat.st_size)


def predict_image(tile_id: str, blur: float = 0.0, noise: float = 0.0) -> np.ndarray:
    """Return model-resolution (INPUT_SIZE, INPUT_SIZE, 4) probabilities on CPU.

    blur/noise are severities in [0,1]: source-image Gaussian sigma is up to
    8 pixels; noise sigma up to 40 grayscale levels. First-call model/data
    initialization is lazy; latency comparisons should distinguish cold/warm.
    """
    if not all(np.isfinite(value) and 0 <= value <= 1 for value in (blur, noise)):
        raise ValueError("blur and noise must be finite severities in [0, 1]")
    index = index_for(tile_id)
    if index < 0 or tile_id_for(index) != tile_id:
        raise ValueError(f"invalid source tile id: {tile_id}")
    dataset = _dataset()
    if index >= len(dataset):
        raise ValueError(f"source tile id out of range: {tile_id}")
    image = np.asarray(dataset[index]["image"], dtype=np.uint8)
    if blur:
        image = cv2.GaussianBlur(image, (0, 0), sigmaX=8.0 * blur, sigmaY=8.0 * blur)
    if noise:
        image = np.clip(image.astype(np.float32) + np.random.default_rng(index).normal(
            0.0, 40.0 * noise, image.shape), 0, 255).astype(np.uint8)
    tensor = preprocess(image).unsqueeze(0)
    model = get_model()
    with torch.inference_mode():
        logits = model(tensor)
        probs = torch.softmax(logits.float() / temperature(), dim=1)
    return probs[0].permute(1, 2, 0).numpy()


def predict_tile(tile_id: str, blur: float = 0.0, noise: float = 0.0) -> np.ndarray:
    """Return (20,20,4) float32 probabilities for one source image on CPU."""
    probs = predict_image(tile_id, blur=blur, noise=noise)
    return warp_probs(probs, _homographies()[tile_id])
