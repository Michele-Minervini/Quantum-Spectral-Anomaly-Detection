"""Classical kernel-PCA novelty detection -- the matched nonlinear baseline.

The anomaly score is the reconstruction error in centered feature space: the
squared distance from a test point to the top-K kernel-principal subspace
(Hoffmann, *Pattern Recognition* 2007).  With an RBF kernel this is the
apples-to-apples classical counterpart of QSAD on the ``rff_gaussian`` feature
map (which approximates the same RBF kernel), so it controls for the
feature-space dimensionality that a quantum embedding would otherwise be
credited with.
"""

import numpy as np
from scipy.spatial.distance import pdist
from sklearn.metrics.pairwise import rbf_kernel


def median_bandwidth(X):
    """RBF ``gamma`` from the median pairwise-distance heuristic.

    A standard, label-free default for kernel methods, so the bandwidth is not
    hand-tuned to favour any one method.
    """
    sq = pdist(np.asarray(X, dtype=float), metric="sqeuclidean")
    return float(1.0 / (2.0 * np.median(sq)))


class KernelPCANovelty:
    """RBF kernel-PCA reconstruction-error detector fitted on nominal ``X``.

    The retained rank is chosen by an explained-variance target ``alpha``, the
    same criterion QSAD uses for retained mass, so the two are matched.
    """

    def __init__(self, X, gamma=4.0, alpha=0.9, center=True):
        self.X = np.asarray(X, dtype=float)
        self.gamma = gamma
        self.center = center
        N = len(self.X)

        K = rbf_kernel(self.X, gamma=gamma)               # (N, N) nominal kernel
        if center:                                        # textbook kernel PCA
            self.col_mean = K.mean(axis=0)
            self.all_mean = K.mean()
            Kc = K - self.col_mean[None, :] - self.col_mean[:, None] + self.all_mean
        else:                                             # matches QSAD's uncentered C
            self.col_mean = np.zeros(N)
            self.all_mean = 0.0
            Kc = K

        w, V = np.linalg.eigh(Kc)                         # kernel-PCA spectrum
        order = np.argsort(w)[::-1]
        w = np.clip(w[order], 1e-12, None)
        V = V[:, order]
        self.n_comp = int(np.searchsorted(np.cumsum(w) / w.sum(), alpha) + 1)
        self.lam = w[:self.n_comp]
        self.alphas = V[:, :self.n_comp]                  # component directions

    def _project(self, Z):
        """Return projections onto the retained PCs and the feature-space norm^2."""
        Z = np.atleast_2d(Z)
        Kz = rbf_kernel(Z, self.X, gamma=self.gamma)      # (m, N)
        if self.center:
            Kz_c = (Kz - Kz.mean(axis=1, keepdims=True)
                    - self.col_mean[None, :] + self.all_mean)
            norm2 = 1.0 - 2.0 * Kz.mean(axis=1) + self.all_mean
        else:
            Kz_c = Kz
            norm2 = np.ones(Kz.shape[0])                  # ||phi(z)||^2 = k(z,z) = 1
        proj = (Kz_c @ self.alphas) / np.sqrt(self.lam)
        return proj, norm2

    def scores(self, Z):
        """Reconstruction-error novelty score (the kernel analogue of Q)."""
        proj, norm2 = self._project(Z)
        return np.clip(norm2 - np.sum(proj ** 2, axis=1), 0.0, None)

    def t2_scores(self, Z, gamma=0.0):
        """Kernel Hotelling T^2: inverse-variance-weighted retained projections."""
        proj, _ = self._project(Z)
        var = self.lam / len(self.X)                      # variance along each PC
        return np.sum(proj ** 2 / (var + gamma), axis=1)
