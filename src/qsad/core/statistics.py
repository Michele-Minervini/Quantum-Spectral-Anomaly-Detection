"""QSAD anomaly statistics: the residual ``Q`` score and shell-resolved ``T^2``.

For a test state ``sigma`` the retained-support score is ``S = Tr(M sigma)`` and
the residual is ``Q = 1 - S``.  ``T^2`` partitions the retained support into
rank-calibrated spectral shells ``D_l`` and sums inverse-variance-weighted
shell masses ``p_sigma(l) / (lambda_bar_l + gamma)``.

Test states are passed as a real/complex array ``states`` of shape ``(m, d)``
whose rows are normalized state vectors (the pure-state case used throughout
the experiments).
"""

import numpy as np


def _populations(detector, states):
    """Populations ``|<u_j|psi_i>|^2`` of each test state in the C eigenbasis."""
    states = np.atleast_2d(states)
    amps = states @ detector.eigvecs.conj()      # <u_j | psi_i>
    return np.abs(amps) ** 2                       # (m, d)


def q_scores(detector, states, mu, T):
    """Residual ``Q^f_alpha(sigma) = 1 - Tr(M^f_{mu,T} sigma)`` per test state."""
    occ = detector.occupations(mu, T)            # (d,)
    retained = _populations(detector, states) @ occ
    return 1.0 - retained


def raw_scores(detector, states):
    """Density-weighted overlap score ``Q_raw = 1 - <psi|C|psi>``."""
    return 1.0 - _populations(detector, states) @ detector.eigvals


def hard_scores(detector, states, K):
    """Hard support score ``Q_hard = 1 - <psi|P_K|psi>`` (top-K projector)."""
    return 1.0 - _populations(detector, states)[:, :K].sum(axis=1)


def shell_denominators(detector, T, kappas):
    """Rank-calibrated shells: return ``(D, r, v, lambda_bar)``.

    ``kappas`` are increasing target soft ranks (e.g. ``[1, 2, 3]`` for
    unit-soft-rank shells).  ``D[l]`` holds the eigenvalue weights of the shell
    effect ``D_l = M_l - M_{l-1}`` with ``M_0 = 0``.
    """
    mus = np.array([detector.calibrate_rank(k, T) for k in kappas])
    occ = np.stack([detector.occupations(mu, T) for mu in mus])   # (L, d)
    M = np.vstack([np.zeros(detector.dim), occ])                   # prepend M_0 = 0
    D = np.diff(M, axis=0)                                         # (L, d) shell weights
    r = D.sum(axis=1)                                             # soft ranks
    v = D @ detector.eigvals                                      # nominal mass per shell
    lambda_bar = np.divide(v, r, out=np.zeros_like(v), where=r > 0)
    return D, r, v, lambda_bar


def t2_scores(detector, states, T, kappas, gamma=0.0, retained=None):
    """Shell-resolved Hotelling ``T^2`` per test state.

    ``retained`` selects which shells contribute (default: all of them).
    """
    D, r, v, lambda_bar = shell_denominators(detector, T, kappas)
    p_shell = _populations(detector, states) @ D.T               # (m, L) shell masses
    sel = np.arange(len(kappas)) if retained is None else np.asarray(retained)
    return p_shell[:, sel] @ (1.0 / (lambda_bar[sel] + gamma))
