"""Jezero elevation, slope, and native-resolution plane-fit roughness grids."""
from __future__ import annotations

from pathlib import Path
import sys

# Support the documented ``python world/terrain.py`` invocation.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import rasterio

from backend.app.contracts import GRID, RES_M
from world.paths import DTM_CROP, RUNS_DIR, TERRAIN_NPZ

_KEYS = ("elevation", "slope_deg", "roughness")
_PADDING_M = 4.0


def plane_fit_roughness(elevation: np.ndarray) -> np.ndarray:
    """Return the standard deviation of 5x5 plane residuals in metres.

    Fit on the native DTM, before interpolation. Reflect padding is only used
    outside the input raster; the committed crop's 4 m margin keeps it outside
    the final world. Float64 prevents cancellation at the DTM's large offset.
    """
    z = np.asarray(elevation, dtype=np.float64)
    if z.ndim != 2 or not np.isfinite(z).all():
        raise ValueError("elevation must be a finite two-dimensional array")
    yy, xx = np.mgrid[-2:3, -2:3]
    design = np.column_stack((xx.ravel(), yy.ravel(), np.ones(25)))
    windows = np.lib.stride_tricks.sliding_window_view(
        np.pad(z, 2, mode="reflect"), (5, 5)
    ).reshape(*z.shape, 25)
    coefficients = windows @ np.linalg.pinv(design).T
    residuals = windows - coefficients @ design.T
    return residuals.std(axis=-1)


def _validate(grids: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    for key in _KEYS:
        arr = grids[key]
        if arr.shape != (GRID, GRID) or arr.dtype != np.float32:
            raise ValueError(f"{key} must be ({GRID}, {GRID}) float32")
        if not np.isfinite(arr).all():
            raise ValueError(f"{key} contains non-finite values")
    if np.any((grids["roughness"] < 0) | (grids["roughness"] > 1)):
        raise ValueError("roughness must lie in [0, 1]")
    return grids


def build_terrain(path=DTM_CROP) -> dict[str, np.ndarray]:
    """Build north-up 0.25 m grids from the padded 1 m Jezero crop."""
    with rasterio.open(path) as dataset:
        if not np.allclose(dataset.res, (1.0, 1.0)):
            raise ValueError("expected a DTM with 1 m native resolution")
        transform = dataset.transform
        if transform.a <= 0 or transform.e >= 0 or transform.b != 0 or transform.d != 0:
            raise ValueError("expected an axis-aligned, north-up DTM")
        native = dataset.read(1, masked=True)
        if np.ma.getmaskarray(native).any():
            raise ValueError("DTM crop contains nodata")
        native = np.asarray(native, dtype=np.float64)
    expected = int(round(GRID * RES_M + 2 * _PADDING_M))
    if native.shape != (expected, expected) or not np.isfinite(native).all():
        raise ValueError(f"expected a finite padded {expected}x{expected} DTM")

    full_size = int(round(expected / RES_M))
    padded_z = cv2.resize(native, (full_size, full_size), interpolation=cv2.INTER_LINEAR)
    dz_row, dz_col = np.gradient(padded_z, RES_M)
    padded_slope = np.degrees(np.arctan(np.hypot(dz_row, dz_col)))
    native_roughness = plane_fit_roughness(native)
    padded_roughness = cv2.resize(
        native_roughness, (full_size, full_size), interpolation=cv2.INTER_LINEAR
    )
    pad = int(round(_PADDING_M / RES_M))
    patch = np.s_[pad:pad + GRID, pad:pad + GRID]
    roughness = padded_roughness[patch]
    p99 = float(np.percentile(roughness, 99))
    # A perfectly planar input has only floating-point residuals; keep it zero.
    roughness = np.clip(roughness / p99, 0, 1) if p99 > 1e-10 else np.zeros_like(roughness)
    return _validate({
        "elevation": padded_z[patch].astype(np.float32),
        "slope_deg": padded_slope[patch].astype(np.float32),
        "roughness": roughness.astype(np.float32),
    })


def load_terrain() -> dict[str, np.ndarray]:
    """Read the committed terrain grids without raster processing."""
    with np.load(TERRAIN_NPZ, allow_pickle=False) as data:
        return _validate({key: data[key].copy() for key in _KEYS})


def _save_plot(grids: dict[str, np.ndarray], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource

    elevation = grids["elevation"]
    shade = LightSource(azdeg=315, altdeg=45).shade(
        elevation, cmap=plt.get_cmap("terrain"), dx=RES_M, dy=RES_M,
        vert_exag=1, blend_mode="soft"
    )
    extent = (0, GRID * RES_M, GRID * RES_M, 0)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    axes[0].imshow(shade, extent=extent, origin="upper")
    scalar = plt.cm.ScalarMappable(cmap="terrain", norm=plt.Normalize(elevation.min(), elevation.max()))
    fig.colorbar(scalar, ax=axes[0], label="Elevation (m)", shrink=0.8)
    slope_image = axes[1].imshow(grids["slope_deg"], extent=extent, origin="upper", cmap="magma", vmin=0)
    centres = (np.arange(GRID) + 0.5) * RES_M
    if grids["slope_deg"].min() < 20 < grids["slope_deg"].max():
        axes[1].contour(centres, centres, grids["slope_deg"], levels=[20], colors="cyan", linewidths=0.6)
    fig.colorbar(slope_image, ax=axes[1], label="Slope (degrees); cyan = 20 degrees", shrink=0.8)
    rough_image = axes[2].imshow(grids["roughness"], extent=extent, origin="upper", cmap="viridis", vmin=0, vmax=1)
    fig.colorbar(rough_image, ax=axes[2], label="Normalized 5x5 plane residual", shrink=0.8)
    for axis, title in zip(axes, ("Jezero elevation + hillshade", "Slope", "Native 1 m roughness")):
        axis.set_title(title)
        axis.set_xlabel("East (m)")
        axis.set_ylabel("South from north edge (m)")
    fig.suptitle("Jezero 50 m patch | north up | 0.25 m cells")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    grids = build_terrain()
    TERRAIN_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(TERRAIN_NPZ, **grids)
    plot_path = RUNS_DIR / "terrain" / "terrain.png"
    _save_plot(grids, plot_path)
    z, slope, roughness = (grids[key] for key in _KEYS)
    print(f"Elevation range: {z.min():.6f} to {z.max():.6f} m")
    print(f"Slope median: {np.median(slope):.6f} deg; p95: {np.percentile(slope, 95):.6f} deg")
    print(f"Slope >20 deg: {np.mean(slope > 20):.6%}")
    print(f"Roughness std: {roughness.std():.6f}; range: {roughness.min():.6f} to {roughness.max():.6f}")
    print(f"Saved {TERRAIN_NPZ} ({TERRAIN_NPZ.stat().st_size} bytes)")
    print(f"Saved {plot_path}")


if __name__ == "__main__":
    main()
