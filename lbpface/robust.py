"""Robustness to uneven lighting and in-plane rotation (workshop extension, not in the paper).

The paper's recogniser (LBP^{u2}_{16,2}, 30x37 windows, chi2, 1-NN) is unchanged; two steps
are added around it.

* Brightness-plane subtraction (Sung & Poggio, 1998): fit ``a*x + b*y + c`` to the image by
  least squares, leaving clipped (0 / 255) pixels out, and subtract it.  LBP is already
  invariant to a monotonic change of the whole face, but a brightness ramp across the face
  flips bits wherever the face is smooth; subtracting the plane removes a linear ramp exactly.
* Rotation search: the windows are fixed in the image, so a rotation moves eyes, nose and
  mouth into other windows.  Every gallery image is also stored rotated by ``ROTATION_GRID``
  (edge pixels repeated, no black corners) and a probe takes the smallest chi2 over the
  copies.  The grid does not contain the test angles (5, 10, 20 deg).

``RobustLBP(plane=False, angles=(0.0,))`` is exactly the paper's recogniser.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy import ndimage

from .distances import pairwise_distances
from .features import LBPDescriptor

ROTATION_GRID = (-24.0, -18.0, -12.0, -6.0, 0.0, 6.0, 12.0, 18.0, 24.0)


def subtract_brightness_plane(image: np.ndarray, lo: float = 0, hi: float = 255) -> np.ndarray:
    """Subtract the least-squares plane fitted to the pixels strictly inside ``(lo, hi)``."""
    img = np.asarray(image, dtype=np.float64)
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    A = np.stack([xx.ravel() / w, yy.ravel() / h, np.ones(h * w)], axis=1)
    ok = ((img > lo) & (img < hi)).ravel()
    coef = np.linalg.lstsq(A[ok], img.ravel()[ok], rcond=None)[0]
    return img - (A @ coef).reshape(h, w)


def rotate(image: np.ndarray, angle: float, fill: str = "nearest") -> np.ndarray:
    """Bilinear rotation by ``angle`` degrees (counter-clockwise) about the centre, same size."""
    return ndimage.rotate(np.asarray(image, dtype=np.float64), angle, reshape=False, order=1, mode=fill)


@dataclass(frozen=True)
class RobustLBP:
    """The paper's recogniser plus brightness-plane subtraction and rotation search."""

    plane: bool = True
    angles: Tuple[float, ...] = ROTATION_GRID
    descriptor: LBPDescriptor = LBPDescriptor.orl()

    def features(self, images, angle: float = 0.0) -> np.ndarray:
        """``(n_images, feature_length)`` after rotating by ``angle`` and preprocessing."""
        ims = [rotate(im, angle) if angle else np.asarray(im, dtype=np.float64) for im in images]
        return self.descriptor.transform([subtract_brightness_plane(im) if self.plane else im for im in ims])

    def gallery_features(self, images) -> np.ndarray:
        """``(n_angles, n_images, feature_length)``"""
        return np.stack([self.features(images, a) for a in self.angles])

    def distances(self, probe_feats: np.ndarray, gallery_feats: np.ndarray) -> np.ndarray:
        """``(n_probes, n_gallery)``: smallest chi2 over the rotated copies."""
        return np.min([pairwise_distances(probe_feats, g) for g in gallery_feats], axis=0)
