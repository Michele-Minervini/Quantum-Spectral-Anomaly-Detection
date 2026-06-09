"""The Detector ``M^f_{mu,T} = f((C - mu I) / T)``.

Given a nominal operator ``C`` (a density operator for QSAD; trace one), the
detector is diagonal in the eigenbasis of ``C`` with occupations
``f((lambda_j - mu) / T)``.  The threshold ``mu`` is calibrated so the retained
nominal mass ``R(mu) = Tr(C M) = sum_j lambda_j f((lambda_j - mu)/T)`` equals a
target ``alpha`` -- the quantum analogue of an explained-variance level.
"""

import numpy as np
from scipy.optimize import brentq

from .responses import GAUSSIAN


class SpectralDetector:
    """Spectral detector built from the nominal operator ``C``.

    Diagonalizes ``C`` once; all detector quantities reuse the eigenpairs.
    Eigenvalues/vectors are stored in descending eigenvalue order.
    """

    def __init__(self, C, response=GAUSSIAN):
        w, V = np.linalg.eigh(C)              # C is Hermitian PSD
        order = np.argsort(w)[::-1]
        self.eigvals = np.clip(w[order].real, 0.0, None)  # lambda_j, descending
        self.eigvecs = V[:, order]                        # columns u_j
        self.response = response
        self.dim = C.shape[0]

    @classmethod
    def from_states(cls, states, response=GAUSSIAN, weights=None):
        """Build the detector from the nominal mixture ``C = sum_i w_i |psi_i><psi_i|``.

        ``states`` is an ``(N, d)`` array of state vectors; ``weights`` default
        to uniform ``1/N``.
        """
        states = np.atleast_2d(states)
        if weights is None:
            C = (states.T @ states.conj()) / len(states)
        else:
            w = np.asarray(weights, dtype=float)
            w = w / w.sum()
            C = (states.T * w) @ states.conj()
        return cls(C, response=response)

    def occupations(self, mu, T):
        """Occupations ``f((lambda_j - mu) / T)`` of the detector."""
        return self.response.f((self.eigvals - mu) / T)

    def retained_mass(self, mu, T):
        """Retained nominal mass ``R(mu) = Tr(C M^f_{mu,T})``."""
        return float(np.sum(self.eigvals * self.occupations(mu, T)))

    def soft_rank(self, mu, T):
        """Soft rank ``Tr(M^f_{mu,T})`` (the I/d-probe calibration quantity)."""
        return float(np.sum(self.occupations(mu, T)))

    def calibrate_mu(self, alpha, T):
        """Solve ``R(mu) = alpha`` for the retained-mass threshold ``mu_alpha``.

        ``R`` decreases monotonically from Tr(C) to 0, so a bracketed root
        search is robust.  A pad of 40*T saturates the response at the bracket
        ends regardless of the resolution ``T``.
        """
        lo = self.eigvals.min() - 40.0 * T
        hi = self.eigvals.max() + 40.0 * T
        return brentq(lambda mu: self.retained_mass(mu, T) - alpha, lo, hi)

    def calibrate_rank(self, kappa, T):
        """Solve ``Tr(M^f_{mu,T}) = kappa`` for a soft-rank threshold."""
        lo = self.eigvals.min() - 40.0 * T
        hi = self.eigvals.max() + 40.0 * T
        return brentq(lambda mu: self.soft_rank(mu, T) - kappa, lo, hi)

    def effect(self, mu, T):
        """Materialize the detector effect matrix ``M^f_{mu,T}`` (d x d)."""
        occ = self.occupations(mu, T)
        return (self.eigvecs * occ) @ self.eigvecs.conj().T

    def hard_projector(self, K):
        """Top-K spectral projector ``P_K`` (the sharp-limit comparator)."""
        U = self.eigvecs[:, :K]
        return U @ U.conj().T
