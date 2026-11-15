import cv2
import numpy as np
import pytest

from world.warp import CELLS, OUT_PX, homography_for, source_corners, warp_image, warp_label, warp_probs


def body(top=700):
    mask = np.zeros((1024, 1024), np.uint8)
    mask[top:] = 1
    return mask


def test_homography_maps_corners_and_orientation():
    mask = body()
    H = homography_for(mask)
    corners = source_corners(mask)
    assert H.dtype == np.float64
    assert corners[2, 1] == 688
    assert corners[0, 1] < corners[2, 1]
    result = cv2.perspectiveTransform(corners[None], H)[0]
    np.testing.assert_allclose(result, [[0, 0], [159, 0], [159, 159], [0, 159]], atol=1e-5)


def test_mask_edge_excludes_upper_arm_and_handles_no_rover():
    mask = body()
    mask[:100] = 1
    assert source_corners(mask)[2, 1] == 688
    assert source_corners(np.zeros_like(mask))[2, 1] == 1012


@pytest.mark.parametrize('cls', [0, 1, 2, 3, 255])
def test_constant_labels_and_dtype(cls):
    result = warp_label(np.full((1024, 1024), cls, np.uint8), homography_for(body()))
    assert result.shape == (CELLS, CELLS)
    assert result.dtype == np.uint8
    assert np.all(result == cls)


def test_majority_vote_ignores_unknown_and_invalid_labels():
    label = np.full((160, 160), 255, np.uint8)
    label[::8, ::8] = 2
    label[1::8, ::8] = 17
    H = np.diag([160 / 1024, 160 / 1024, 1])
    np.testing.assert_array_equal(warp_label(label, H), np.full((20, 20), 2, np.uint8))


@pytest.mark.parametrize('size', [256, 1024, 73])
def test_probability_resolution_and_normalization(size):
    probs = np.empty((size, size, 4), np.float32)
    probs[:] = [0.1, 0.2, 0.3, 0.4]
    result = warp_probs(probs, homography_for(body()))
    assert result.shape == (20, 20, 4)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result.sum(-1), 1, atol=1e-6)
    np.testing.assert_allclose(result, np.broadcast_to(probs[0, 0], result.shape), atol=1e-6)


def test_probability_spatial_alignment_across_resolutions():
    H = homography_for(body())
    outputs = []
    for size in (256, 1024):
        x = np.arange(size, dtype=np.float32) / size
        p = np.zeros((size, size, 4), np.float32)
        p[..., 0] = x
        p[..., 1] = 1 - x
        outputs.append(warp_probs(p, H))
    np.testing.assert_allclose(outputs[0], outputs[1], atol=2e-4)


def test_unsupported_probabilities_are_uniform():
    result = warp_probs(np.zeros((256, 256, 4), np.float32), np.eye(3))
    np.testing.assert_array_equal(result, np.full((20, 20, 4), .25, np.float32))


def test_image_shape_dtype_and_forward_direction():
    src = np.broadcast_to(np.arange(1024, dtype=np.uint16)[:, None], (1024, 1024))
    result = warp_image((src // 4).astype(np.uint8), homography_for(body()))
    assert result.shape == (OUT_PX, OUT_PX)
    assert result.dtype == np.uint8
    assert result[0].mean() < result[-1].mean()


def test_random_labels_are_valid():
    label = np.random.default_rng(0).choice([0, 1, 2, 3, 255], size=(1024, 1024)).astype(np.uint8)
    assert set(np.unique(warp_label(label, homography_for(body())))) <= {0, 1, 2, 3, 255}
