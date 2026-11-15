"""Deterministic flat-ground approximation shared by labels and model output.

No camera intrinsics are available: the square is a nominal 5 m patch, not
metric photogrammetry. Row zero is far/north; the bottom is near the rover.
"""
from __future__ import annotations

import cv2
import numpy as np

from backend.app.contracts import TILE

OUT_PX = 160
CELLS = TILE
SOURCE_PX = 1024
PATCH_HEIGHT = 300.0
ROVER_GAP = 12.0


def source_corners(rover_mask: np.ndarray) -> np.ndarray:
    """Return TL, TR, BR, BL in canonical 1024 px image coordinates.

    A few NAVCAM frames have the arm across the upper image. When that
    causes a majority at the very top, use the lower-half body edge instead.
    With no body majority (e.g. an unobstructed view), use the frame bottom.
    Only this vertical offset varies between images.
    """
    mask = np.asarray(rover_mask)
    if mask.ndim != 2 or min(mask.shape) < 2:
        raise ValueError("rover_mask must be a two-dimensional image")
    h, w = mask.shape
    central = mask[:, round(300 * w / SOURCE_PX):round(724 * w / SOURCE_PX)]
    rows = np.flatnonzero((central > 0).mean(axis=1) > 0.5)
    edge = float(rows[0] * SOURCE_PX / h) if rows.size else float(SOURCE_PX)
    if edge < PATCH_HEIGHT + ROVER_GAP:
        lower = rows[rows >= h // 2]
        edge = float(lower[0] * SOURCE_PX / h) if lower.size else float(SOURCE_PX)
    bottom = edge - ROVER_GAP
    top = bottom - PATCH_HEIGHT
    return np.array([[312, top], [711, top], [923, bottom], [100, bottom]], np.float32)


def homography_for(rover_mask: np.ndarray) -> np.ndarray:
    """Canonical source image pixels -> 160 px top-down patch pixels."""
    dst = np.array([[0, 0], [OUT_PX-1, 0], [OUT_PX-1, OUT_PX-1], [0, OUT_PX-1]], np.float32)
    return cv2.getPerspectiveTransform(source_corners(rover_mask), dst)


def _matrix_for_shape(H: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    matrix = np.asarray(H, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("H must be a finite 3x3 homography")
    h, w = shape[:2]
    if h < 1 or w < 1:
        raise ValueError("input image must not be empty")
    return matrix @ np.diag([SOURCE_PX / w, SOURCE_PX / h, 1.0])


def warp_label(label: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Nearest-neighbour warp and 8x8 majority vote, ignoring unknowns."""
    src = np.asarray(label, dtype=np.uint8)
    if src.ndim != 2:
        raise ValueError("label must be a two-dimensional class map")
    src = np.where(src <= 3, src, 255).astype(np.uint8)
    patch = cv2.warpPerspective(src, _matrix_for_shape(H, src.shape), (OUT_PX, OUT_PX),
                                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    block = OUT_PX // CELLS
    counts = np.stack([(patch == c).reshape(CELLS, block, CELLS, block).sum(axis=(1, 3))
                       for c in range(4)], axis=-1)
    result = counts.argmax(axis=-1).astype(np.uint8)
    result[counts.sum(axis=-1) == 0] = 255
    return result


def warp_image(img: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Bilinear image warp; accepts grayscale or RGB uint8 frames."""
    src = np.asarray(img, dtype=np.uint8)
    if src.ndim not in (2, 3):
        raise ValueError("image must be grayscale or multichannel")
    return cv2.warpPerspective(src, _matrix_for_shape(H, src.shape), (OUT_PX, OUT_PX),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def warp_probs(probs_hw4: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Warp any input resolution, average 8x8 cells, and renormalize.

    Pixels outside the source image carry no evidence; entirely unsupported
    cells become uniform rather than producing NaNs or a false soil label.
    """
    src = np.asarray(probs_hw4, dtype=np.float32)
    if src.ndim != 3 or src.shape[-1] != 4:
        raise ValueError("probabilities must have shape (height, width, 4)")
    if not np.isfinite(src).all() or (src < 0).any():
        raise ValueError("probabilities must be finite and nonnegative")
    patch = cv2.warpPerspective(src, _matrix_for_shape(H, src.shape), (OUT_PX, OUT_PX),
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    block = OUT_PX // CELLS
    cells = patch.reshape(CELLS, block, CELLS, block, 4).mean(axis=(1, 3))
    mass = cells.sum(axis=-1, keepdims=True)
    return np.divide(cells, mass, out=np.full_like(cells, 0.25), where=mass > 0)
