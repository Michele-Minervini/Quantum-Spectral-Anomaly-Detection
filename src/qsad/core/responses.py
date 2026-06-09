"""Optimization-admissible response functions ``f: R -> (0, 1)``.

These are the smooth thresholds used by the QSAD detector
``M = f((C - mu I) / T)``.  Two responses are provided:

* ``GAUSSIAN`` -- the probit response ``f(x) = Phi(x) = erfc(-x/sqrt2)/2``.
  This is the default, since its qumode control state is a plain Gaussian
  wavepacket (hardware efficient).
* ``LOGISTIC`` -- the Fermi-Dirac response ``f(x) = 1/(1 + e^-x)``.

Each response also exposes its inverse (quantile) ``finv``, used for shell
threshold reasoning and sharp-limit checks.
"""

import numpy as np
from scipy.special import erfc, erfcinv, expit, logit


class Response:
    """A monotone smooth threshold ``f`` together with its inverse ``finv``."""

    def __init__(self, name, f, finv):
        self.name = name
        self._f = f
        self._finv = finv

    def f(self, x):
        """Evaluate the response on a scalar or array argument."""
        return self._f(np.asarray(x, dtype=float))

    def finv(self, m):
        """Evaluate the quantile (inverse response) on ``m`` in (0, 1)."""
        return self._finv(np.asarray(m, dtype=float))

    def __call__(self, x):
        return self.f(x)

    def __repr__(self):
        return f"Response({self.name!r})"


# Gaussian / probit response.  Phi(x) = erfc(-x / sqrt(2)) / 2 is preferred for
# numerical stability over forming 1 + erf directly.
def _gaussian_f(x):
    return 0.5 * erfc(-x / np.sqrt(2.0))


def _gaussian_finv(m):
    return -np.sqrt(2.0) * erfcinv(2.0 * m)


GAUSSIAN = Response("gaussian", _gaussian_f, _gaussian_finv)

# Logistic / Fermi-Dirac response, using SciPy's stable expit/logit.
LOGISTIC = Response("logistic", expit, logit)

# Registry for easy lookup by name from scripts.
RESPONSES = {r.name: r for r in (GAUSSIAN, LOGISTIC)}
