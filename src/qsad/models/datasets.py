"""Synthetic classical datasets for Experiment A.

A suite of one-class point clouds in ``[0, 1]^2`` spanning the fairness range
for linear PCA:

* ``gaussian_blob`` -- tilted elongated Gaussian; the linear-correlation case
  PCA is designed for (PCA should do well).
* ``moon_cloud``    -- a curved crescent; PCA's straight slab cannot follow it.
* ``ring``          -- an annulus; linear PCA fails outright (the mean sits in
  the empty centre, so no principal direction is meaningful).

These three (blob / moons / circles) are the standard trio in the scikit-learn
outlier-detection and kernel-method comparisons.
"""

import numpy as np
from sklearn.datasets import make_moons


def _unit_square(X, margin=0.1):
    """Affine-rescale a 2D cloud into ``[margin, 1 - margin]^2``."""
    lo, hi = X.min(axis=0), X.max(axis=0)
    return margin + (1.0 - 2.0 * margin) * (X - lo) / (hi - lo)


def gaussian_blob(n=400, seed=0, elongation=4.0, angle=np.pi / 6):
    """Tilted elongated Gaussian -- the linear case PCA is designed for."""
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((n, 2)) * np.array([elongation, 1.0])
    c, s = np.cos(angle), np.sin(angle)
    pts = pts @ np.array([[c, -s], [s, c]]).T          # rotate the elongation axis
    return _unit_square(pts)


def moon_cloud(n=400, noise=0.07, seed=0):
    """One crescent from ``make_moons`` -- curved, so PCA cannot follow the arc."""
    X, y = make_moons(n_samples=2 * n, noise=noise, random_state=seed)
    return _unit_square(X[y == 0])                     # keep a single moon


def ring(n=400, seed=0, width=0.1):
    """Annulus -- linear PCA fails (the mean is in the empty centre)."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, 2.0 * np.pi, n)
    r = 1.0 + width * rng.standard_normal(n)
    pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    return _unit_square(pts)


def grid_points(res=300, lo=0.0, hi=1.0):
    """Dense evaluation grid over the square; returns ``(points, xs, ys)``.

    ``points`` has shape ``(res*res, 2)`` (row-major), suitable for scoring;
    reshape a per-point score to ``(res, res)`` for ``imshow``.
    """
    xs = np.linspace(lo, hi, res)
    ys = np.linspace(lo, hi, res)
    gx, gy = np.meshgrid(xs, ys)
    points = np.column_stack([gx.ravel(), gy.ravel()])
    return points, xs, ys


def uniform_anomalies(n=400, lo=0.0, hi=1.0, seed=1):
    """Uniform background points, used as anomalies for scoring."""
    rng = np.random.default_rng(seed)
    return rng.uniform(lo, hi, size=(n, 2))


DATASETS = {"blob": gaussian_blob, "moon": moon_cloud, "ring": ring}
