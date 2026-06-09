"""Classical PCA monitoring -- the rigid-boundary comparator for QSAD.

Standard centered PCA with the two process-monitoring statistics:

* ``Q_K(z) = ||(I - P_K) z||^2``         residual energy outside the subspace
* ``T2_K(z) = sum_{j<=K} <u_j,z>^2 / (lambda_j + gamma)``   in-subspace leverage

The retained rank ``K`` is chosen by an explained-variance target ``alpha``.
"""

import numpy as np


class ClassicalPCA:
    """Centered PCA fitted on nominal vectors ``X`` (shape ``(N, d)``)."""

    def __init__(self, X):
        X = np.asarray(X, dtype=float)
        self.mean = X.mean(axis=0)
        Y = X - self.mean
        C = (Y.T @ Y) / len(X)                     # sample covariance
        w, V = np.linalg.eigh(C)
        order = np.argsort(w)[::-1]
        self.eigvals = np.clip(w[order], 0.0, None)
        self.eigvecs = V[:, order]                 # columns u_j, descending

    def choose_K(self, alpha):
        """Smallest ``K`` whose cumulative explained variance reaches ``alpha``."""
        frac = np.cumsum(self.eigvals) / self.eigvals.sum()
        return int(np.searchsorted(frac, alpha) + 1)

    def _coords(self, X):
        """Principal coordinates ``<u_j, x - mean>`` of test vectors."""
        return (np.atleast_2d(X) - self.mean) @ self.eigvecs

    def q_scores(self, X, K):
        """Residual ``Q`` statistic for each row of ``X``."""
        return (self._coords(X)[:, K:] ** 2).sum(axis=1)

    def t2_scores(self, X, K, gamma=0.0):
        """Hotelling ``T^2`` statistic for each row of ``X``."""
        coords = self._coords(X)[:, :K]
        return (coords ** 2 / (self.eigvals[:K] + gamma)).sum(axis=1)
