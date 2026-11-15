"""Shared U-Net definition and image preprocessing for training and inference."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch

from world.paths import WEIGHTS

INPUT_SIZE = 256
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def build_model() -> torch.nn.Module:
    """Four-class U-Net with an ImageNet-initialized ResNet34 encoder."""
    return smp.Unet("resnet34", encoder_weights="imagenet", classes=4)


def preprocess(img_u8: np.ndarray, size: int = INPUT_SIZE) -> torch.Tensor:
    """Convert uint8 grayscale/RGB imagery to an ImageNet-normalized CHW tensor."""
    if size <= 0 or size % 32:
        raise ValueError("size must be a positive multiple of 32")
    img = np.asarray(img_u8)
    if img.dtype != np.uint8:
        raise TypeError("image must have dtype uint8")
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)
    elif img.ndim == 3 and img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    elif img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("image must have shape (H,W), (H,W,1), or (H,W,3)")
    interpolation = cv2.INTER_AREA if max(img.shape[:2]) > size else cv2.INTER_LINEAR
    img = cv2.resize(img, (size, size), interpolation=interpolation).astype(np.float32) / 255.0
    img = (img - _MEAN) / _STD
    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))


def load_weights(path: str | Path = WEIGHTS, device: str | torch.device = "cpu") -> torch.nn.Module:
    """Load deployable fp16 state into an eval-mode fp32 model, without downloads."""
    model = smp.Unet("resnet34", encoder_weights=None, classes=4)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model.float().to(device).eval()
