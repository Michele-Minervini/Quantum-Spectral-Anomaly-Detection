"""Turn raw anomaly scores into normal-calibrated percentiles and metrics.

A detector's raw ``Q`` / ``T^2`` values are not comparable across detectors.
We calibrate them against the empirical distribution of *nominal* scores: a
test score is mapped to the fraction of nominal scores below it, so 0 means
"deep nominal" and 1 means "more extreme than any nominal sample".
"""

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


class PercentileCalibrator:
    """Empirical-CDF map from raw scores to normal-calibrated percentiles."""

    def __init__(self, nominal_scores):
        self.reference = np.sort(np.asarray(nominal_scores, dtype=float))

    def transform(self, scores):
        """Percentile in [0, 1] of each score within the nominal distribution."""
        ranks = np.searchsorted(self.reference, scores, side="right")
        return ranks / len(self.reference)


def roc_auc(labels, scores):
    """ROC-AUC with label 1 = anomaly, 0 = nominal."""
    return float(roc_auc_score(labels, scores))


def roc_points(labels, scores):
    """Return ``(fpr, tpr)`` arrays for plotting an ROC curve."""
    fpr, tpr, _ = roc_curve(labels, scores)
    return fpr, tpr
