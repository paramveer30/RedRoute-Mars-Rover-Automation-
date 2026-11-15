"""Evaluate the frozen AI4Mars checkpoint; never train or select on test data.

Run ``python -m ml.evaluate`` from the repository root. Native-resolution
metrics interpolate logits before argmax; 256px validation metrics are also
reported to distinguish this protocol from A01's checkpoint-selection metric.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from backend.app.contracts import CLASS_NAMES
from ml.ai4mars import cache_train_subset, clean_label, load_test, train_val_split
from ml.model import INPUT_SIZE, load_weights, preprocess
from world.paths import RUNS_DIR, SEG_METRICS, WEIGHTS


def confusion_counts(labels: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Four-class counts, truth rows/prediction columns; ignore truth 255."""
    labels, predicted = np.asarray(labels), np.asarray(predicted)
    if labels.shape != predicted.shape:
        raise ValueError("truth and prediction shapes differ")
    valid = labels != 255
    truth = labels[valid].astype(np.int64)
    guesses = predicted[valid].astype(np.int64)
    if np.any((truth < 0) | (truth > 3)) or np.any((guesses < 0) | (guesses > 3)):
        raise ValueError("labelled pixels and their predictions must be classes 0..3")
    return np.bincount(truth * 4 + guesses, minlength=16).reshape(4, 4)


def segmentation_metrics(confusion: np.ndarray) -> dict:
    """Global pixel-weighted confusion metrics, never image-averaged IoU."""
    cm = np.asarray(confusion, dtype=np.int64)
    if cm.shape != (4, 4) or (cm < 0).any() or cm.sum() == 0:
        raise ValueError("confusion must be a nonnegative 4x4 matrix with labelled pixels")
    intersection = np.diag(cm)
    union = cm.sum(axis=0) + cm.sum(axis=1) - intersection
    ious = np.divide(intersection, union, out=np.full(4, np.nan), where=union > 0)
    return {
        "miou": float(np.nanmean(ious)),
        "iou": {name: float(value) if np.isfinite(value) else None
                for name, value in zip(CLASS_NAMES, ious)},
        "pixel_acc": float(intersection.sum() / cm.sum()),
        "class_share": dict(zip(CLASS_NAMES, (cm.sum(axis=1) / cm.sum()).tolist())),
        "class_pixels": dict(zip(CLASS_NAMES, cm.sum(axis=1).tolist())),
        "labelled_pixels": int(cm.sum()),
        "confusion_matrix": cm.tolist(),
    }


@torch.inference_mode()
def evaluate_split(model, rows, device, batch_size=16, collect_samples=0):
    """Decode only this split and evaluate full-size cleaned expert/crowd masks."""
    native_cm = np.zeros((4, 4), np.int64)
    input_cm = np.zeros((4, 4), np.int64)
    samples = []
    for start in range(0, len(rows), batch_size):
        tensors, labels = [], []
        for index in range(start, min(start + batch_size, len(rows))):
            row = rows[index]
            tensor = preprocess(np.asarray(row["image"], dtype=np.uint8))
            label = clean_label(row["label_mask"], row["rover_mask"], row["range_mask"])
            tensors.append(tensor)
            labels.append(label)
            if len(samples) < collect_samples:
                samples.append(tensor.clone())
        logits = model(torch.stack(tensors).to(device))
        low_predictions = logits.argmax(dim=1).cpu().numpy()
        for j, label in enumerate(labels):
            high = F.interpolate(logits[j:j + 1], size=label.shape,
                                 mode="bilinear", align_corners=False)
            predicted = high.argmax(dim=1)[0].cpu().numpy()
            native_cm += confusion_counts(label, predicted)
            low_label = cv2.resize(label, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_NEAREST)
            input_cm += confusion_counts(low_label, low_predictions[j])
        if (start // batch_size + 1) % 5 == 0 or start + batch_size >= len(rows):
            print(f"evaluated {min(start + batch_size, len(rows))}/{len(rows)} images", flush=True)
    return segmentation_metrics(native_cm), segmentation_metrics(input_cm), samples


@torch.inference_mode()
def gpu_throughput(model, samples, device, batches=20):
    """Warm, resident batch-16 FP32 forward throughput with CUDA synchronization."""
    if device.type != "cuda":
        return None
    batch = torch.stack(samples[:16]).to(device)
    if len(batch) != 16:
        raise ValueError("GPU throughput requires 16 prepared images")
    for _ in range(5):
        model(batch)
    torch.cuda.synchronize(device)
    start = perf_counter()
    for _ in range(batches):
        model(batch)
    torch.cuda.synchronize(device)
    return len(batch) * batches / (perf_counter() - start)


@torch.inference_mode()
def cpu_latency(model, samples):
    """Batch-one forward only; decoding/preprocessing/loading are excluded."""
    if len(samples) != 50:
        raise ValueError("CPU timing requires exactly 50 prepared test images")
    model.cpu().float().eval()
    for tensor in samples[:5]:
        model(tensor.unsqueeze(0))
    timings = []
    for tensor in samples:
        start = perf_counter()
        model(tensor.unsqueeze(0))
        timings.append((perf_counter() - start) * 1000)
    p50, p95 = np.percentile(timings, [50, 95])
    return {"p50": float(p50), "p95": float(p95), "n_images": 50,
            "warmup_images": 5, "batch_size": 1, "threads": torch.get_num_threads(),
            "precision": "float32", "includes": "model forward only",
            "excludes": ["dataset decoding", "preprocess", "model loading"]}



def _encoded_image_hashes(rows):
    """Audit exact source-file duplicates without decoding unrelated mask columns."""
    from datasets import Image
    encoded = rows.select_columns(["image"]).cast_column("image", Image(decode=False))
    hashes = []
    for row in encoded:
        item = row["image"]
        content = item["bytes"]
        if content is None:
            content = Path(item["path"]).read_bytes()
        hashes.append(hashlib.sha256(content).hexdigest())
    return hashes


def audit_source_overlap(subset, train_rows, val_rows, test_rows):
    """Encoded-byte overlap is a lower-bound leakage check, not a perceptual audit."""
    full = _encoded_image_hashes(subset)
    train = set(_encoded_image_hashes(train_rows))
    val = set(_encoded_image_hashes(val_rows))
    test = _encoded_image_hashes(test_rows)
    overlap = set(full) & set(test)
    return {
        "method": "SHA256 of encoded source image bytes via shared loaders; Image(decode=False)",
        "subset_rows": len(full), "subset_unique_images": len(set(full)),
        "test_rows": len(test), "test_unique_images": len(set(test)),
        "subset_test_overlap_unique_images": len(overlap),
        "test_rows_overlapping_subset": sum(digest in overlap for digest in test),
        "training6000_test_overlap_unique_images": len(train & set(test)),
        "validation500_test_overlap_unique_images": len(val & set(test)),
        "training_validation_overlap_unique_images": len(train & val),
        "limitation": "Different encodings of identical pixels are not detected by this encoded-byte audit.",
    }


def verify_validation_membership(rows, subset_size):
    """Optionally compare actual validation contents to A01's prepared cache.

    Hugging Face storage fingerprints can change after save/load. Compare
    image/label arrays rather than treating that cache key as a row identity.
    """
    cache = RUNS_DIR / "train" / f"prepared_{subset_size}_{INPUT_SIZE}" / "val"
    if not (cache / "images.npy").exists() or not (cache / "labels.npy").exists():
        return {"status": "A01 prepared cache unavailable; seeded shared split used"}
    images = np.load(cache / "images.npy", mmap_mode="r")
    labels = np.load(cache / "labels.npy", mmap_mode="r")
    if len(images) != len(rows) or len(labels) != len(rows):
        raise ValueError("A01 cached validation row count differs")
    for index, row in enumerate(rows):
        image = np.asarray(row["image"], dtype=np.uint8)
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=2)
        elif image.ndim == 3 and image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        resized = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
        label = clean_label(row["label_mask"], row["rover_mask"], row["range_mask"])
        low = cv2.resize(label, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_NEAREST)
        if not np.array_equal(resized, images[index]) or not np.array_equal(low, labels[index]):
            raise ValueError(f"A01 validation membership/order differs at row {index}")
    return {
        "images_and_labels_matched_a01_cache_in_order": len(rows),
        "method": "Exact RGB area-resized images and cleaned nearest-resized labels at INPUT_SIZE.",
        "note": "HF save/load fingerprints are not stable row identities; actual validation content matched.",
    }


def save_confusion_plot(confusion, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.asarray(confusion, np.float64)
    totals = cm.sum(axis=1, keepdims=True)
    normalized = np.divide(cm, totals, out=np.zeros_like(cm), where=totals > 0)
    fig, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
    heatmap = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    labels = [name.replace("_", " ") for name in CLASS_NAMES]
    axis.set_xticks(range(4), labels, rotation=20)
    axis.set_yticks(range(4), labels)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("Expert truth class")
    axis.set_title("AI4Mars test_min3 | 322 images\nRow-normalized confusion, ignored pixels excluded")
    for row in range(4):
        for col in range(4):
            value = normalized[row, col]
            text = f"{value:.1%}" if totals[row, 0] > 0 else "N/A"
            axis.text(col, row, text, ha="center", va="center",
                      color="white" if value > 0.5 else "black")
    fig.colorbar(heatmap, ax=axis, label="Fraction of labelled truth pixels")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def markdown_table(result):
    lines = ["| Metric | Test, native resolution | Fixed validation, native resolution |",
             "| --- | ---: | ---: |",
             f'| mIoU | {result["miou"]:.6f} | {result["val_miou"]:.6f} |',
             f'| Pixel accuracy | {result["pixel_acc"]:.6f} | {result["val_pixel_acc"]:.6f} |']
    for name in CLASS_NAMES:
        test, val = result["iou"][name], result["val_iou"][name]
        values = [(f"{value:.6f}" if value is not None else "N/A") for value in (test, val)]
        lines.append(f'| {name} IoU | {values[0]} | {values[1]} |')
    lines.extend(["", "| Timing metric | Measured value |", "| --- | ---: |",
                  f'| CPU batch 1 p50 | {result["cpu_latency_ms"]["p50"]:.3f} ms |',
                  f'| CPU batch 1 p95 | {result["cpu_latency_ms"]["p95"]:.3f} ms |',
                  f'| GPU batch 16 throughput | {result["gpu_imgs_per_s"]:.3f} images/s |'
                  if result["gpu_imgs_per_s"] is not None else '| GPU throughput | unavailable |',
                  "", "Timing excludes dataset decoding, preprocessing and model loading."])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1 or args.cpu_threads < 1:
        parser.error("batch size and CPU threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    cv2.setNumThreads(1)
    device = torch.device(args.device)
    started = perf_counter()
    source_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    weights_hash = hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()
    model = load_weights(device=device)
    test_rows = load_test()
    if len(test_rows) != 322:
        raise ValueError(f"expected all 322 expert test images, received {len(test_rows)}")
    subset = cache_train_subset(6500)
    train_rows, val_rows = train_val_split(subset, 500)
    print(f"device={device}; test={len(test_rows)}; val={len(val_rows)}; input={INPUT_SIZE}", flush=True)
    print("Evaluating expert test split at native mask resolution", flush=True)
    test, _, samples = evaluate_split(model, test_rows, device, args.batch_size, 50)
    print("Evaluating fixed 500-row validation split at native mask resolution", flush=True)
    validation, low_validation, _ = evaluate_split(model, val_rows, device, args.batch_size)
    throughput = gpu_throughput(model, samples, device)
    model.cpu()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print("GPU_RELEASED: all evaluation and throughput stages finished", flush=True)
    latency = cpu_latency(model, samples)
    audit = audit_source_overlap(subset, train_rows, val_rows, test_rows)
    membership = verify_validation_membership(val_rows, len(subset))
    result = {"split": "test_min3", "n_images": len(test_rows), "input_size": INPUT_SIZE,
              **test, "val_miou": validation["miou"], "val_iou": validation["iou"],
              "val_pixel_acc": validation["pixel_acc"], "val_n_images": len(val_rows),
              "val_fingerprint": val_rows._fingerprint,
              "source_image_overlap_audit": audit,
              "validation_membership_verification": membership,
              "val_input_resolution_miou": low_validation["miou"],
              "val_input_resolution_iou": low_validation["iou"],
              "native_evaluation": {"logit_resize": "bilinear", "align_corners": False,
                                    "argmax_after_resize": True, "ignored_label": 255},
              "cpu_latency_ms": latency, "gpu_imgs_per_s": throughput,
              "gpu_benchmark": {"batch_size": 16, "warmup_batches": 5, "measured_batches": 20,
                                "precision": "float32", "includes": "resident model forward only",
                                "device": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
              "weights": WEIGHTS.relative_to(WEIGHTS.parents[2]).as_posix(),
              "weights_sha256": weights_hash, "weights_bytes": WEIGHTS.stat().st_size,
              "source_git_sha": source_sha,
              "wall_seconds": perf_counter() - started,
              "notes": ["Frozen A01 checkpoint; no training or model selection on test_min3.",
                        "Native test/val metrics use cleaned full-resolution labels and bilinearly resized logits.",
                        "A01 checkpoint-selection validation used nearest-resized 256px labels and AMP; "
                        "val_input_resolution metrics here use FP32 deployment inference."]}
    if audit["subset_test_overlap_unique_images"]:
        result["notes"].append(
            "DATASET LEAKAGE: source-image overlap exists between the 6500-row subset and test split; "
            "test metrics are not an independent held-out image estimate.")
    else:
        result["notes"].append(
            "Encoded image SHA256 audit found no overlap between the 6500-row subset and 322 test images; "
            "differently encoded duplicates are not ruled out.")
    SEG_METRICS.parent.mkdir(parents=True, exist_ok=True)
    SEG_METRICS.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    plot_path = RUNS_DIR / "eval" / "confusion.png"
    save_confusion_plot(test["confusion_matrix"], plot_path)
    print(markdown_table(result), flush=True)
    print(f"Test labelled-pixel class shares: {test['class_share']}", flush=True)
    print(f"256px deployment validation mIoU: {low_validation['miou']:.6f}", flush=True)
    print(f"Saved {SEG_METRICS}; {plot_path}", flush=True)
    if test["iou"]["big_rock"] is not None and test["iou"]["big_rock"] < 0.15:
        print("HUMAN CHECKPOINT: test big-rock IoU below 0.15. Report honestly; do not tune on test results.", flush=True)


if __name__ == "__main__":
    main()
