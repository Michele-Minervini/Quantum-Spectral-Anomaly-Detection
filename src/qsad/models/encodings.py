"""Classical-to-quantum feature maps ``x in [0,1]^2 -> normalized amplitudes``.

Each map sends a batch ``X`` of shape ``(m, 2)`` to feature states of shape
``(m, D)`` with unit-norm rows.  Swapping maps is how Experiment A tests which
embedding gives the cleanest curved QSAD boundary.  All maps here target two
qubits (``D = 4``); ``iqp`` produces complex amplitudes.
"""

import numpy as np

# Two-qubit gates / sign tables reused by the IQP map.
_H2 = np.array([[1, 1, 1, 1],
                [1, -1, 1, -1],
                [1, 1, -1, -1],
                [1, -1, -1, 1]], dtype=float) / 2.0
_S0 = np.array([1.0, 1.0, -1.0, -1.0])     # Z on qubit 0, per basis state
_S1 = np.array([1.0, -1.0, 1.0, -1.0])     # Z on qubit 1
_S01 = _S0 * _S1                            # Z0 Z1


def _normalize(F):
    return F / np.linalg.norm(F, axis=1, keepdims=True)


def poly_amplitude(X, a=1.25, b=0.75):
    """Homogeneous polynomial amplitude feature (real, D=4)."""
    xt = 2.0 * X[:, 0] - 1.0                # rescale to [-1, 1]
    yt = 2.0 * X[:, 1] - 1.0
    F = np.stack([np.ones_like(xt), a * xt, a * yt, b * xt * yt], axis=1)
    return _normalize(F)


def angle_encoding(X):
    """Product of single-qubit rotations: ``|q0> tensor |q1>`` (real, D=4)."""
    theta = (np.pi / 2.0) * X              # [0, 1] -> [0, pi/2]
    c, s = np.cos(theta), np.sin(theta)
    F = np.stack([c[:, 0] * c[:, 1], c[:, 0] * s[:, 1],
                  s[:, 0] * c[:, 1], s[:, 0] * s[:, 1]], axis=1)
    return _normalize(F)


def iqp_encoding(X, depth=2, scale=np.pi):
    """Second-order Pauli-Z (IQP / ZZ) feature map (complex, D=4).

    Applies ``(U_Phi H^{\\otimes 2})^{depth}`` to ``|00>`` with single-qubit
    phases ``x_i`` and an entangling phase ``(pi - x_0)(pi - x_1)``.
    """
    x = scale * X
    phi0, phi1 = x[:, 0], x[:, 1]
    phi01 = (np.pi - x[:, 0]) * (np.pi - x[:, 1])
    phases = (phi0[:, None] * _S0 + phi1[:, None] * _S1
              + phi01[:, None] * _S01)             # (m, 4)
    diag = np.exp(1j * phases)

    state = np.zeros((len(X), 4), dtype=complex)
    state[:, 0] = 1.0                                # |00>
    for _ in range(depth):
        state = (state @ _H2) * diag                # H^{otimes 2} then U_Phi
    return _normalize(state)


def poly2(X):
    """Full degree-2 polynomial features {1, x, y, x^2, xy, y^2} (real, D=6).

    The degree-2 polynomial-kernel feature map -- a richer, more curved
    relative of ``poly_amplitude``.  (D need not be a power of two for the
    simulation; on hardware one would pad to 3 qubits.)
    """
    xt, yt = 2.0 * X[:, 0] - 1.0, 2.0 * X[:, 1] - 1.0
    F = np.stack([np.ones_like(xt), xt, yt, xt ** 2, xt * yt, yt ** 2], axis=1)
    return _normalize(F)


def fourier(X, harmonics=2):
    """Tensor-product Fourier features, harmonics 1..H per axis (real, D=4H^2).

    With the default ``H=2`` this is a 16-dim (four-qubit) periodic feature
    map that captures oscillatory / closed structure.
    """
    def axis_feats(t):
        ang = np.pi * np.outer(t, np.arange(1, harmonics + 1))
        return np.concatenate([np.cos(ang), np.sin(ang)], axis=1)        # (m, 2H)

    fx, fy = axis_feats(X[:, 0]), axis_feats(X[:, 1])
    F = np.einsum("mi,mj->mij", fx, fy).reshape(len(X), -1)              # tensor product
    return _normalize(F)


def rff_gaussian(X, n_features=16, gamma=4.0, seed=0):
    """Random Fourier features approximating an RBF kernel (real, D=16).

    Approximates ``exp(-gamma||x-y||^2)`` -- the standard nonlinear kernel for
    anomaly detection.  The random projection is regenerated from a fixed seed
    on every call, so train, grid, and test states share one feature map.
    """
    rng = np.random.default_rng(seed)
    W = rng.normal(0.0, np.sqrt(2.0 * gamma), size=(2, n_features))
    b = rng.uniform(0.0, 2.0 * np.pi, size=n_features)
    return _normalize(np.cos(X @ W + b))


ENCODINGS = {
    "poly_amplitude": poly_amplitude,
    "angle": angle_encoding,
    "iqp": iqp_encoding,
    "poly2": poly2,
    "fourier": fourier,
    "rff_gaussian": rff_gaussian,
}
