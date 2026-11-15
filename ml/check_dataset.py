"""Sanity-check the AI4Mars Hugging Face mirror: loads test_min3 (322 images) and streams a
few training rows. Prints columns, shapes and label values.

    .venv/Scripts/python ml/check_dataset.py
"""
from __future__ import annotations

import numpy as np
from datasets import load_dataset

test = load_dataset("hassanjbara/AI4MARS", split="test_min3")
print(test)
row = test[0]
for k, v in row.items():
    arr = np.asarray(v) if not isinstance(v, (str, int, float)) else v
    print(k, getattr(arr, "shape", arr), getattr(arr, "dtype", ""))
print("label values in first mask:", np.unique(np.asarray(row["label_mask"])))

train = load_dataset("hassanjbara/AI4MARS", split="train", streaming=True).take(3)
for r in train:
    print("train row:", {k: getattr(np.asarray(v), "shape", v) for k, v in r.items()})
