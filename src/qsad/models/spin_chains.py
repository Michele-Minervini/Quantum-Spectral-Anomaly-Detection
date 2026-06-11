"""1D spin-chain Hamiltonians and state preparation for Experiment B.

Transverse-field Ising model (TFIM)

    H = -J sum_i Z_i Z_{i+1} - h sum_i X_i,

built as sparse operators and diagonalized for its lowest-energy states.  The
quantum critical point is at ``h/J = 1`` (ordered for ``h < J``, paramagnetic
for ``h > J``).  A small Heisenberg/XXZ builder is included for reuse.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

_I = sp.identity(2, format="csr")
_X = sp.csr_matrix([[0.0, 1.0], [1.0, 0.0]])
_Y = sp.csr_matrix([[0.0, -1.0j], [1.0j, 0.0]])
_Z = sp.csr_matrix([[1.0, 0.0], [0.0, -1.0]])


def _embed(op, site, n):
    """Place single-site operator ``op`` at ``site`` of an ``n``-spin chain."""
    factors = [op if k == site else _I for k in range(n)]
    out = factors[0]
    for f in factors[1:]:
        out = sp.kron(out, f, format="csr")
    return out


def tfim_hamiltonian(n, J=1.0, h=1.0, periodic=False):
    """Sparse TFIM Hamiltonian on ``n`` spins."""
    dim = 2 ** n
    H = sp.csr_matrix((dim, dim))
    bonds = n if periodic else n - 1
    for i in range(bonds):
        H = H - J * (_embed(_Z, i, n) @ _embed(_Z, (i + 1) % n, n))
    for i in range(n):
        H = H - h * _embed(_X, i, n)
    return H


def heisenberg_hamiltonian(n, J=1.0, delta=1.0, periodic=False):
    """Sparse XXZ Heisenberg Hamiltonian (anisotropy ``delta``)."""
    dim = 2 ** n
    H = sp.csr_matrix((dim, dim))
    bonds = n if periodic else n - 1
    for i in range(bonds):
        j = (i + 1) % n
        H = H + J * (_embed(_X, i, n) @ _embed(_X, j, n)
                     + _embed(_Y, i, n) @ _embed(_Y, j, n)
                     + delta * _embed(_Z, i, n) @ _embed(_Z, j, n))
    return H


def low_energy_states(H, k):
    """Return the ``k`` lowest eigenstates of ``H`` as rows of a real array,
    sorted by ascending energy."""
    vals, vecs = eigsh(H, k=k, which="SA")
    order = np.argsort(vals)
    return np.ascontiguousarray(vecs[:, order].T.real)        # (k, dim)


def perturbed_samples(modes, weights, n_samples, noise=0.02, seed=0):
    """Sample states from a weighted mixture of ``modes`` with small noise.

    Each sample picks a mode by ``weights``, adds Gaussian noise in amplitude
    space, and renormalizes -- a finite cloud around each nominal mode.
    """
    rng = np.random.default_rng(seed)
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    dim = modes.shape[1]
    idx = rng.choice(len(modes), size=n_samples, p=weights)
    # Scale by 1/sqrt(dim) so the perturbation norm is ~noise regardless of
    # Hilbert-space dimension (otherwise it would grow as sqrt(dim)).
    kick = (noise / np.sqrt(dim)) * rng.standard_normal((n_samples, dim))
    samples = modes[idx] + kick
    samples /= np.linalg.norm(samples, axis=1, keepdims=True)
    return samples
