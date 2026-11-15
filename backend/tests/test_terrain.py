"""Checks for the real Jezero grids and the native-resolution roughness fit."""
import numpy as np
import pytest

from backend.app.contracts import GRID
from world.terrain import build_terrain, load_terrain, plane_fit_roughness


@pytest.fixture(scope="module")
def grids():
    return build_terrain()


def test_grid_contract(grids):
    assert set(grids) == {"elevation", "slope_deg", "roughness"}
    for grid in grids.values():
        assert grid.shape == (GRID, GRID)
        assert grid.dtype == np.float32
        assert np.isfinite(grid).all()


def test_real_roughness_is_bounded_and_non_degenerate(grids):
    roughness = grids["roughness"]
    assert roughness.min() >= 0
    assert roughness.max() <= 1
    assert roughness.std() > 0.05


def test_real_scarp_and_drivable_plateau(grids):
    slope = grids["slope_deg"]
    assert 0.02 <= np.mean(slope > 20) <= 0.25
    assert np.median(slope) < 10
    # The selected crop's delta-front scarp is at the east edge, not west.
    assert np.mean(slope[:, -40:] > 20) > np.mean(slope[:, :40] > 20)


def test_tilted_plane_has_zero_native_roughness():
    yy, xx = np.mgrid[:58, :58]
    plane = -2490 + 0.25 * xx - 0.17 * yy
    residual = plane_fit_roughness(plane)
    # Ignore the helper's reflected outer rim, just as build_terrain trims it.
    np.testing.assert_allclose(residual[2:-2, 2:-2], 0, atol=1e-10)


def test_native_roughness_detects_nonplanar_terrain():
    yy, xx = np.mgrid[:58, :58]
    surface = -2490 + 0.25 * xx - 0.17 * yy + 0.2 * xx**2
    assert plane_fit_roughness(surface)[2:-2, 2:-2].min() > 0.1


def test_committed_grids_match_source(grids):
    committed = load_terrain()
    for key in grids:
        np.testing.assert_allclose(committed[key], grids[key], rtol=1e-6, atol=1e-5)
