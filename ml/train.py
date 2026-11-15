"""Train the shared AI4Mars U-Net; test_min3 is never loaded here.

Run from the repository root: python ml/train.py --limit 64 --epochs 1
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

# Support the documented direct-script invocation as well as python -m ml.train.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader, Dataset

from ml.ai4mars import cache_train_subset, clean_label, train_val_split
from ml.model import INPUT_SIZE, build_model, load_weights, preprocess
from world.paths import CKPT_DIR, RUNS_DIR, WEIGHTS

CLASS_NAMES = ("soil", "bedrock", "sand", "big_rock")


class PreparedDataset(Dataset):
    """Cache decoded, resized train/val arrays, keeping iteration off the HF decoder."""

    def __init__(self, rows, directory: Path, size: int, augment: bool):
        self.augment = augment
        self.size = size
        directory.mkdir(parents=True, exist_ok=True)
        identity = {"fingerprint": rows._fingerprint, "size": size, "count": len(rows), "version": 1}
        meta_path = directory / "metadata.json"
        complete = meta_path.exists() and json.loads(meta_path.read_text()) == identity
        if not complete:
            images = np.lib.format.open_memmap(directory / "images.npy", mode="w+", dtype=np.uint8,
                                              shape=(len(rows), size, size, 3))
            labels = np.lib.format.open_memmap(directory / "labels.npy", mode="w+", dtype=np.uint8,
                                              shape=(len(rows), size, size))
            for i, row in enumerate(rows):
                image = np.asarray(row["image"], dtype=np.uint8)
                if image.ndim == 2:
                    image = np.repeat(image[..., None], 3, axis=2)
                elif image.ndim == 3 and image.shape[2] == 1:
                    image = np.repeat(image, 3, axis=2)
                if image.ndim != 3 or image.shape[2] != 3:
                    raise ValueError(f"unsupported image shape: {image.shape}")
                label = clean_label(row["label_mask"], row["rover_mask"], row["range_mask"])
                images[i] = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
                labels[i] = cv2.resize(label, (size, size), interpolation=cv2.INTER_NEAREST)
                if (i + 1) % 250 == 0 or i + 1 == len(rows):
                    print(f"prepared {directory.name}: {i + 1}/{len(rows)}", flush=True)
            images.flush()
            labels.flush()
            del images, labels
            meta_path.write_text(json.dumps(identity), encoding="utf-8")
        self.images = np.load(directory / "images.npy", mmap_mode="r")
        self.labels = np.load(directory / "labels.npy", mmap_mode="r")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = np.array(self.images[index], copy=True)
        label = np.array(self.labels[index], copy=True)
        if self.augment:
            if random.random() < 0.5:
                image = image[:, ::-1].copy()
                label = label[:, ::-1].copy()
            contrast = random.uniform(0.9, 1.1)
            brightness = random.uniform(-0.05, 0.05) * 255.0
            image = np.clip((image.astype(np.float32) - 127.5) * contrast + 127.5 + brightness,
                            0, 255).astype(np.uint8)
        return preprocess(image, self.size), torch.from_numpy(label.astype(np.int64))


def class_weights(dataset: PreparedDataset):
    counts = np.zeros(4, dtype=np.int64)
    for label in dataset.labels:
        counts += np.bincount(label[label < 4].ravel(), minlength=4)
    if counts.sum() == 0:
        raise ValueError("training subset contains no valid labels")
    frequency = counts / counts.sum()
    weights = np.clip(1.0 / np.sqrt(np.maximum(frequency, 1e-12)), 1.0, 20.0)
    return torch.tensor(weights, dtype=torch.float32), counts


def iou_from_confusion(confusion: torch.Tensor):
    cm = confusion.detach().cpu().numpy().astype(np.float64)
    intersection = np.diag(cm)
    union = cm.sum(0) + cm.sum(1) - intersection
    iou = np.divide(intersection, union, out=np.full(4, np.nan), where=union > 0)
    return float(np.nanmean(iou)), iou


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None, help="smoke-test row count (full run: 6500)")
    parser.add_argument("--size", type=int, default=INPUT_SIZE)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.size < 32 or args.size % 32:
        parser.error("epochs/batch size must be positive; size must be a positive multiple of 32")
    if args.limit is not None and args.limit < 2:
        parser.error("limit must be at least 2")
    started = time.perf_counter()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    if amp:
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
    print(f"device={device}; size={args.size}; batch={args.batch_size}; seed={args.seed}", flush=True)
    smoke = args.limit is not None
    run_dir = RUNS_DIR / "train"
    checkpoint_dir = CKPT_DIR
    if smoke:
        run_dir = run_dir / "smoke"
        checkpoint_dir = checkpoint_dir / "smoke"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ds = cache_train_subset(args.limit if smoke else 6500)
    n_val = min(500, max(1, len(ds) // 8)) if smoke else 500
    if len(ds) <= n_val:
        raise ValueError("dataset has too few labelled rows for the fixed validation split")
    train_rows, val_rows = train_val_split(ds, n_val)
    print(f"train={len(train_rows)}; val={len(val_rows)}; train fingerprint={train_rows._fingerprint}; "
          f"val fingerprint={val_rows._fingerprint}", flush=True)
    prepared = run_dir / f"prepared_{len(ds)}_{args.size}"
    train_ds = PreparedDataset(train_rows, prepared / "train", args.size, True)
    val_ds = PreparedDataset(val_rows, prepared / "val", args.size, False)
    weights, counts = class_weights(train_ds)
    print(f"valid-pixel class counts (training labels at {args.size}px): {counts.tolist()}", flush=True)
    print(f"inverse-sqrt class weights (clip 1..20): {weights.tolist()}", flush=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=amp)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=amp)
    model = build_model().to(device)
    ce = torch.nn.CrossEntropyLoss(weight=weights.to(device), ignore_index=255)
    dice = smp.losses.DiceLoss("multiclass", ignore_index=255)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best_iou, best_epoch, best_per_class = -math.inf, 0, None
    checkpoint_path = checkpoint_dir / "best.pt"
    fieldnames = ["epoch", "train_loss", "val_miou", *[f"iou_{name}" for name in CLASS_NAMES],
                  "lr", "epoch_seconds", "elapsed_seconds"]
    with (run_dir / "log.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        stream.flush()
        for epoch in range(1, args.epochs + 1):
            epoch_started = time.perf_counter()
            model.train()
            total_loss = 0.0
            examples = 0
            for images, labels in train_loader:
                images = images.to(device, non_blocking=amp)
                labels = labels.to(device, non_blocking=amp)
                if not torch.any(labels != 255):
                    continue
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda") if amp else nullcontext():
                    logits = model(images)
                    loss = ce(logits, labels) + dice(logits, labels)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss at epoch {epoch}")
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item() * images.shape[0]
                examples += images.shape[0]
            if examples == 0:
                raise RuntimeError("no supervised training examples")
            model.eval()
            confusion = torch.zeros((4, 4), dtype=torch.int64, device=device)
            with torch.inference_mode():
                for images, labels in val_loader:
                    images = images.to(device, non_blocking=amp)
                    labels = labels.to(device, non_blocking=amp)
                    with torch.autocast("cuda") if amp else nullcontext():
                        predicted = model(images).argmax(1)
                    valid = labels != 255
                    confusion += torch.bincount(4 * labels[valid] + predicted[valid], minlength=16).reshape(4, 4)
            miou, per_class = iou_from_confusion(confusion)
            record = dict(epoch=epoch, train_loss=total_loss / examples, val_miou=miou,
                          lr=optimizer.param_groups[0]["lr"], epoch_seconds=time.perf_counter() - epoch_started,
                          elapsed_seconds=time.perf_counter() - started)
            record.update({f"iou_{name}": float(value) for name, value in zip(CLASS_NAMES, per_class)})
            writer.writerow(record)
            stream.flush()
            print(json.dumps(record), flush=True)
            if miou > best_iou:
                best_iou, best_epoch, best_per_class = miou, epoch, per_class.copy()
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                            "epoch": epoch, "val_miou": miou, "class_iou": per_class.tolist(),
                            "class_weights": weights.tolist(), "class_counts": counts.tolist(),
                            "input_size": args.size, "seed": args.seed,
                            "train_fingerprint": train_rows._fingerprint,
                            "val_fingerprint": val_rows._fingerprint}, checkpoint_path)
            scheduler.step()
            # Give the full run at least fifteen epochs before declaring a plateau.
            if epoch >= 15 and epoch - best_epoch >= args.patience:
                print(f"validation plateau: best epoch {best_epoch}", flush=True)
                break
    best = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.cpu().load_state_dict(best["model"])
    del best
    export_path = (checkpoint_dir / "unet_r34_fp16.pt") if smoke else WEIGHTS
    export_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.half().state_dict(), export_path)
    if export_path.stat().st_size >= 50_000_000:
        raise RuntimeError(f"deployable weights must be under 50 MB: {export_path.stat().st_size} bytes")
    # Test offline loading, CPU fp32 execution and preprocessing on a held-out train row.
    cpu_model = load_weights(export_path, "cpu")
    with torch.inference_mode():
        output = cpu_model(preprocess(val_ds.images[0], args.size).unsqueeze(0))
    if output.shape != (1, 4, args.size, args.size) or not torch.isfinite(output).all():
        raise RuntimeError("CPU inference verification failed")
    summary = {"best_val_miou": best_iou, "best_epoch": best_epoch,
               "best_class_iou": dict(zip(CLASS_NAMES, best_per_class.tolist())),
               "epochs_completed": epoch, "wall_seconds": time.perf_counter() - started,
               "class_weights": weights.tolist(), "class_counts": counts.tolist(),
               "train_rows": len(train_rows), "val_rows": len(val_rows),
               "input_size": args.size, "seed": args.seed, "smoke": smoke,
               "checkpoint": str(checkpoint_path), "weights": str(export_path),
               "weight_bytes": export_path.stat().st_size, "cpu_inference": "passed"}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
