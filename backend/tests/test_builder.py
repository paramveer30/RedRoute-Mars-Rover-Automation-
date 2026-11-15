"""World assembly and sampling contracts for real terrain and unrotated tiles."""
import numpy as np
import pytest

from backend.app.contracts import GRID, N_TILES, TILE, World
from world.builder import build_world
from world.paths import TILES_NPZ
from world.terrain import load_terrain


def test_world_shapes_and_dtypes():
    world = build_world(0)
    assert isinstance(world, World)
    assert world.seed == 0
    for name in ("elevation", "slope_deg", "roughness"):
        array = getattr(world, name)
        assert array.shape == (GRID, GRID)
        assert array.dtype == np.float32
        assert np.isfinite(array).all()
    assert world.truth.shape == world.geo_obstacle.shape == (GRID, GRID)
    assert world.truth.dtype == np.uint8
    assert world.geo_obstacle.dtype == bool
    assert not world.geo_obstacle.any()
    assert set(np.unique(world.truth)) <= {0, 1, 2, 3, 255}


def test_seed_is_deterministic_but_changes_tile_layout():
    first, same, different = build_world(12), build_world(12), build_world(13)
    for name in ("elevation", "slope_deg", "roughness", "truth", "geo_obstacle"):
        np.testing.assert_array_equal(getattr(first, name), getattr(same, name))
    assert first.tile_image == same.tile_image
    assert first.tile_image != different.tile_image
    assert not np.array_equal(first.truth, different.truth)
    for name in ("elevation", "slope_deg", "roughness"):
        np.testing.assert_array_equal(getattr(first, name), getattr(different, name))
        np.testing.assert_array_equal(getattr(first, name), load_terrain()[name])


def test_source_ids_resolve_and_truth_is_exact_unrotated_tile():
    world = build_world(0)
    assert len(world.tile_image) == N_TILES
    with np.load(TILES_NPZ, allow_pickle=False) as archive:
        source = dict(zip(archive["ids"].tolist(), archive["label20"]))
        all_refs = []
        for tr, row in enumerate(world.tile_image):
            assert len(row) == N_TILES
            for tc, ref in enumerate(row):
                all_refs.append(ref)
                assert ref in source
                np.testing.assert_array_equal(
                    world.truth[tr*TILE:(tr+1)*TILE, tc*TILE:(tc+1)*TILE], source[ref])
    assert len(set(all_refs)) == N_TILES * N_TILES  # 149 usable tiles: no replacement needed.


def test_world_arrays_and_layouts_do_not_share_mutable_state():
    first, second = build_world(4), build_world(4)
    for name in ("elevation", "slope_deg", "roughness", "truth", "geo_obstacle"):
        assert not np.shares_memory(getattr(first, name), getattr(second, name))
    first.elevation.fill(0)
    first.truth.fill(3)
    first.geo_obstacle.fill(True)
    first.tile_image[0][0] = "changed"
    fresh = build_world(4)
    np.testing.assert_array_equal(second.elevation, fresh.elevation)
    np.testing.assert_array_equal(second.truth, fresh.truth)
    assert not fresh.geo_obstacle.any()
    assert second.tile_image == fresh.tile_image


def test_positive_bias_is_deterministic_and_prefers_hazard_tiles():
    uniform = [build_world(seed) for seed in range(6)]
    biased = [build_world(seed, sand_bias=10) for seed in range(6)]
    np.testing.assert_array_equal(build_world(0, 10).truth, biased[0].truth)
    hazard_fraction = lambda w: np.isin(w.truth, [2, 3]).mean()
    assert np.mean([hazard_fraction(w) for w in biased]) > np.mean([hazard_fraction(w) for w in uniform])


@pytest.mark.parametrize("bias", [-1, float("nan"), float("inf")])
def test_invalid_bias_is_rejected(bias):
    with pytest.raises(ValueError):
        build_world(0, bias)
