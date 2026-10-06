"""Brightness-plane subtraction and rotation search (lbpface/robust.py)."""
import numpy as np
from scipy import ndimage

from lbpface.distances import pairwise_distances
from lbpface.features import LBPDescriptor
from lbpface.lbp import lbp_u2
from lbpface.robust import ROTATION_GRID, RobustLBP, rotate, subtract_brightness_plane


def _texture(seed=0, shape=(112, 92)):
    rng = np.random.default_rng(seed)
    return ndimage.gaussian_filter(rng.uniform(0, 255, shape), 1.5) * 2 - 60  # ~40..200, no ties


def test_plane_subtraction_makes_lbp_invariant_to_a_brightness_ramp():
    img = _texture()
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    ramped = img + 0.6 * xx - 0.3 * yy + 10
    assert (lbp_u2(img, 16, 2) != lbp_u2(ramped, 16, 2)).mean() > 0.05  # the ramp changes codes
    fit_all = dict(lo=-np.inf, hi=np.inf)  # unclipped float images: use every pixel
    np.testing.assert_array_equal(lbp_u2(subtract_brightness_plane(img, **fit_all), 16, 2),
                                  lbp_u2(subtract_brightness_plane(ramped, **fit_all), 16, 2))


def test_clipped_pixels_are_left_out_of_the_fit():
    h, w = 40, 60
    yy, xx = np.mgrid[0:h, 0:w]
    plane = 8.0 * xx - 2.0 * yy - 100  # clipped to 0 on the left, to 255 on the right
    out = subtract_brightness_plane(np.clip(plane, 0, 255))
    inside = (plane > 0) & (plane < 255)
    np.testing.assert_allclose(out[inside], 0, atol=1e-9)


def test_without_additions_it_is_the_paper_recogniser():
    imgs = np.stack([np.clip(_texture(s), 0, 255).astype(np.uint8) for s in range(4)])
    desc = LBPDescriptor.orl()
    model = RobustLBP(plane=False, angles=(0.0,))
    expected = pairwise_distances(desc.transform(imgs[:2]), desc.transform(imgs[2:]))
    got = model.distances(model.features(imgs[:2]), model.gallery_features(imgs[2:]))
    np.testing.assert_array_equal(got, expected)


def test_rotation_search_finds_the_rotated_copy():
    g = np.clip(_texture(1), 0, 255)
    probe = rotate(g, 12.0)
    model = RobustLBP(plane=False, angles=(-12.0, 0.0, 12.0))
    gf = model.gallery_features([g])
    assert gf.shape == (3, 1, model.descriptor.feature_length(g.shape))
    assert model.distances(model.features([probe]), gf)[0, 0] == 0.0
    plain = RobustLBP(plane=False, angles=(0.0,))
    assert plain.distances(plain.features([probe]), plain.gallery_features([g]))[0, 0] > 0.0


def test_default_grid_avoids_the_test_angles():
    assert 0.0 in ROTATION_GRID and not {5.0, 10.0, 20.0} & {abs(a) for a in ROTATION_GRID}
