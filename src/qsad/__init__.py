"""Quantum Spectral Anomaly Detection (QSAD).

A measurement-based one-class anomaly-detection framework that learns soft
spectral tests from a nominal operator ``C`` and produces quantum analogues of
the classical PCA monitoring statistics ``Q`` (residual support) and Hotelling's
``T^2`` (within-support leverage).

Subpackages
-----------
``qsad.core``    response functions, the regularized spectral detector, the Q/T^2
                 statistics, the classical-PCA comparator, percentile calibration.
``qsad.models``  classical datasets, quantum feature maps, spin-chain models.
``qsad.viz``     shared style and figure builders.
"""

from . import core, models, viz

__all__ = ["core", "models", "viz"]
__version__ = "0.1.0"
