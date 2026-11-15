"""Generate the camera panel's source frames and frozen-model class overlays.

Run ``python -m ml.make_overlays``. Model logits are resized before softmax;
only actual model entropy and the rover mask control alpha. No ground-truth
labels are used to construct overlays.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from ml.ai4mars import load_test, tile_id_for
from ml.model import load_weights, preprocess
from world.paths import CALIBRATION, FRAMES_DIR, OVERLAYS_DIR

OUTPUT_SIZE = 384
PALETTE = np.array([[200, 161, 101], [138, 143, 152],
                    [242, 209, 107], [229, 72, 77]], dtype=np.uint8)


def calibration_temperature() -> float:
    """Use a validation-fitted temperature when available, otherwise T=1."""
    value = float(json.loads(CALIBRATION.read_text(encoding="utf-8"))["T"]) if CALIBRATION.exists() else 1.0
    if not np.isfinite(value) or value <= 0:
        raise ValueError("calibration temperature must be finite and positive")
    return value


def overlay_rgba(logits: torch.Tensor, rover_mask: np.ndarray, temperature: float) -> np.ndarray:
    """Render one four-class logit map as a masked 384px RGBA prediction."""
    if logits.ndim != 3 or logits.shape[0] != 4 or not torch.isfinite(logits).all():
        raise ValueError("logits must be finite with shape (4,H,W)")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    resized = F.interpolate(logits[None].float(), size=(OUTPUT_SIZE, OUTPUT_SIZE),
                            mode="bilinear", align_corners=False)[0]
    probabilities = torch.softmax(resized / temperature, dim=0)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=0) / np.log(4)
    classes = probabilities.argmax(dim=0).cpu().numpy()
    uncertainty = entropy.cpu().numpy()
    rover = np.asarray(rover_mask)
    if rover.ndim != 2:
        raise ValueError("rover_mask must be two-dimensional")
    rover = cv2.resize((rover > 0).astype(np.uint8), (OUTPUT_SIZE, OUTPUT_SIZE),
                       interpolation=cv2.INTER_NEAREST).astype(bool)
    rgba = np.empty((OUTPUT_SIZE, OUTPUT_SIZE, 4), np.uint8)
    rgba[..., :3] = PALETTE[classes]
    rgba[..., 3] = np.where((uncertainty > 0.7) | rover, 0, 110).astype(np.uint8)
    return rgba


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1:
        parser.error("batch-size and threads must be positive")
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(1)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    temperature = calibration_temperature()
    rows = load_test()
    if len(rows) != 322:
        raise ValueError(f"expected all 322 test rows, got {len(rows)}")
    model = load_weights(device=device)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    OVERLAYS_DIR.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    transparent, total_pixels = 0, 0
    with torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            end = min(start + args.batch_size, len(rows))
            records = [rows[index] for index in range(start, end)]
            images = [np.asarray(record["image"], np.uint8) for record in records]
            inputs = torch.stack([preprocess(image) for image in images]).to(device)
            logits = model(inputs)
            for offset, (record, image) in enumerate(zip(records, images)):
                source_id = tile_id_for(start + offset)
                frame = cv2.resize(image, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_AREA)
                Image.fromarray(frame).save(FRAMES_DIR / f"{source_id}.jpg", quality=85)
                overlay = overlay_rgba(logits[offset], record["rover_mask"], temperature)
                Image.fromarray(overlay).save(OVERLAYS_DIR / f"{source_id}.png")
                transparent += int((overlay[..., 3] == 0).sum())
                total_pixels += overlay.shape[0] * overlay.shape[1]
            print(f"generated {end}/{len(rows)} frame/overlay pairs", flush=True)
    model.cpu()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print("GPU_RELEASED: frame and overlay generation finished", flush=True)
    ids = [tile_id_for(index) for index in range(len(rows))]
    frames = [FRAMES_DIR / f"{source}.jpg" for source in ids]
    overlays = [OVERLAYS_DIR / f"{source}.png" for source in ids]
    frame_bytes = sum(path.stat().st_size for path in frames)
    overlay_bytes = sum(path.stat().st_size for path in overlays)
    if frame_bytes + overlay_bytes > 15_000_000:
        raise RuntimeError("camera assets exceed the 15 MB target")
    print(json.dumps({"n_images": len(rows), "size": OUTPUT_SIZE, "temperature": temperature,
                      "frame_bytes": frame_bytes, "overlay_bytes": overlay_bytes,
                      "total_bytes": frame_bytes + overlay_bytes,
                      "transparent_pixel_fraction": transparent / total_pixels,
                      "wall_seconds": perf_counter() - started}, indent=2), flush=True)


if __name__ == "__main__":
    main()
