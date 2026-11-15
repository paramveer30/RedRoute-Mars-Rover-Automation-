"""Build the shared AI4Mars tile library and reproducible geometry review sheets.

Run from the repository root: python -m world.make_tiles
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from ml.ai4mars import clean_label, load_test, tile_id_for
from world.paths import RUNS_DIR, TILE_IMG_DIR, TILES_NPZ, USABLE_TILES
from world.warp import homography_for, source_corners, warp_image, warp_label

CLASS_NAMES = ('soil', 'bedrock', 'sand', 'big_rock')
COLORS = np.array([[166, 119, 75], [140, 150, 164], [242, 204, 96], [218, 66, 60]], np.uint8)


def _label_rgb(labels):
    out = np.full((*labels.shape, 3), 30, np.uint8)
    for cls, color in enumerate(COLORS):
        out[labels == cls] = color
    return out


def _contact_sheet(ds, indices, homographies, labels, fractions, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Polygon

    fig, axes = plt.subplots(len(indices), 3, figsize=(12, 3.2 * len(indices)), squeeze=False)
    for row, index in enumerate(indices):
        record = ds[int(index)]
        img = np.asarray(record['image'])
        axes[row, 0].imshow(img, cmap='gray', vmin=0, vmax=255)
        axes[row, 0].add_patch(Polygon(source_corners(np.asarray(record['rover_mask'])),
                                    fill=False, edgecolor='lime', linewidth=1.5))
        axes[row, 0].set_title(f'{tile_id_for(int(index))}: source + trapezoid')
        axes[row, 1].imshow(warp_image(img, homographies[index]), cmap='gray', vmin=0, vmax=255)
        axes[row, 1].set_title('Top-down approximation (far at top)')
        axes[row, 2].imshow(_label_rgb(labels[index]), interpolation='nearest')
        axes[row, 2].set_title(f'Expert labels: {fractions[index]:.1%} valid')
        for ax in axes[row]:
            ax.axis('off')
    legend = [Patch(facecolor=color / 255, label=name) for name, color in zip(CLASS_NAMES, COLORS)]
    legend.append(Patch(facecolor='#1e1e1e', label='unlabelled / masked'))
    fig.legend(handles=legend, loc='upper center', bbox_to_anchor=(.5, .984), ncol=5)
    fig.suptitle('Flat-ground homography; no intrinsics or calibrated metric scale', y=.997)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(path, dpi=120)
    plt.close(fig)


def make_tiles():
    ds = load_test()
    TILE_IMG_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = RUNS_DIR / 'warp'
    out_dir.mkdir(parents=True, exist_ok=True)
    ids, matrices, labels, fractions = [], [], [], []
    suspect = []
    for i, record in enumerate(ds):
        image = np.asarray(record['image'])
        rover = np.asarray(record['rover_mask'])
        mask = clean_label(np.asarray(record['label_mask']), rover, np.asarray(record['range_mask']))
        H = homography_for(rover)
        label20 = warp_label(mask, H)
        image160 = warp_image(image, H)
        image_path = TILE_IMG_DIR / f'{tile_id_for(i)}.png'
        if not cv2.imwrite(str(image_path), cv2.cvtColor(image160, cv2.COLOR_RGB2BGR)
                           if image160.ndim == 3 else image160):
            raise OSError(f'Could not write {image_path}')
        ids.append(tile_id_for(i))
        matrices.append(H)
        labels.append(label20)
        fractions.append(float(np.mean(label20 != 255)))
        body_fraction = (warp_image((rover > 0).astype(np.uint8) * 255, H) > 127).mean()
        if body_fraction > .05:
            suspect.append((tile_id_for(i), float(body_fraction)))
        if (i + 1) % 50 == 0:
            print(f'Warped {i+1}/{len(ds)} images', flush=True)
    ids = np.asarray(ids)
    matrices = np.asarray(matrices, np.float64)
    labels = np.asarray(labels, np.uint8)
    fractions = np.asarray(fractions, np.float32)
    threshold = .6
    if (fractions >= threshold).sum() < 60:
        threshold = .5
        print('Fewer than 60 tiles at 0.6: lowered valid fraction threshold to 0.5.')
    usable = fractions >= threshold
    np.savez_compressed(TILES_NPZ, ids=ids, H=matrices, label20=labels, valid_frac=fractions)
    USABLE_TILES.write_text(json.dumps(sorted(ids[usable].tolist()), indent=2) + '\n', encoding='utf-8')
    counts = {name: int((labels[usable] == cls).sum()) for cls, name in enumerate(CLASS_NAMES)}
    tile_counts = {name: int(np.any(labels[usable] == cls, axis=(1, 2)).sum())
                   for cls, name in enumerate(CLASS_NAMES)}
    print(f'Tiles: {len(ids)}; usable: {int(usable.sum())}; valid_frac threshold: {threshold}')
    print(f'Class cell counts in usable tiles: {counts}')
    print(f'Usable tiles containing each class: {tile_counts}')
    print(f'Potential body intrusion >5% (visual review required): {suspect}')
    rng = np.random.default_rng(3)
    indices = rng.choice(len(ds), size=min(12, len(ds)), replace=False)
    for page in range(2):
        path = out_dir / f'contact_random_{page+1:02d}.png'
        _contact_sheet(ds, indices[page*6:(page+1)*6], matrices, labels, fractions, path)
        print(f'Contact sheet: {path}')
    # Diagnostic frames include a top-arm view, unobstructed terrain, and
    # examples of rare labels; selecting them does not alter warp geometry.
    rare = [int(np.argmax((labels == cls).sum(axis=(1, 2)))) for cls in (2, 3)]
    review = list(dict.fromkeys([0, 20, 60, 100, 160, 240] + rare))
    path = out_dir / 'contact_edge_cases.png'
    _contact_sheet(ds, review, matrices, labels, fractions, path)
    print(f'Edge-case contact sheet: {path}')
    return {'count': len(ids), 'usable': int(usable.sum()), 'threshold': threshold,
            'class_cells': counts, 'class_tiles': tile_counts, 'suspect_body': suspect}


if __name__ == '__main__':
    make_tiles()

