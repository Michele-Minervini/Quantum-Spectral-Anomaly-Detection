"""Classical kernel-PCA baselines vs QSAD feature maps on one dataset.

Builds the appendix comparison figure: three classical kernel-PCA novelty
detectors (polynomial, RBF, Laplacian) in the top row and QSAD with three
feature maps (degree-two polynomial, Gaussian random features, Fourier) in
the bottom row, all on the moon cloud under the same calibration, with the
one-class AUC in each panel title.  Columns pair related kernels: the
polynomial pair, the RBF pair (the random-feature map approximates the RBF
kernel), and the remaining strong kernel on each side.
"""

from functools import partial
from pathlib import Path

import numpy as np

from qsad.core import (
    GAUSSIAN,
    KernelPCANovelty,
    PercentileCalibrator,
    SpectralDetector,
    median_bandwidth,
    roc_auc,
)
from qsad.core import statistics as stats
from qsad.models import grid_points, moon_cloud, uniform_anomalies
from qsad.models.encodings import ENCODINGS, rff_gaussian
from qsad.viz import panel_grid

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "figures"

ALPHA, T, RES = 0.88, 0.035, 280


def main():
    train = moon_cloud(n=450, noise=0.06, seed=0)
    nominal = moon_cloud(n=300, noise=0.06, seed=7)      # held-out nominal
    anomaly = uniform_anomalies(n=300, seed=11)
    test = np.vstack([nominal, anomaly])
    grid, xs, ys = grid_points(RES)
    labels = np.r_[np.zeros(len(nominal)), np.ones(len(anomaly))]

    g_rbf = median_bandwidth(train)
    g_lap = median_bandwidth(train, metric="cityblock")
    kernels = [
        ("kernel-PCA (polynomial)", "poly", 1.0),
        ("kernel-PCA (RBF)", "rbf", g_rbf),
        ("kernel-PCA (Laplacian)", "laplacian", g_lap),
    ]
    encodings = [
        ("QSAD (degree-two polynomial)", ENCODINGS["poly2"]),
        ("QSAD (random features)", partial(rff_gaussian, gamma=g_rbf)),
        ("QSAD (Fourier)", ENCODINGS["fourier"]),
    ]

    panels = []
    for label, kernel, gamma in kernels:
        det = KernelPCANovelty(train, gamma=gamma, alpha=ALPHA, kernel=kernel)
        auc = roc_auc(labels, det.scores(test))
        perc = PercentileCalibrator(det.scores(train))
        panels.append((label, perc.transform(det.scores(grid)).reshape(RES, RES), auc))

    for label, encode in encodings:
        det = SpectralDetector.from_states(encode(train), response=GAUSSIAN)
        mu = det.calibrate_mu(ALPHA, T)
        auc = roc_auc(labels, stats.q_scores(det, encode(test), mu, T))
        perc = PercentileCalibrator(stats.q_scores(det, encode(train), mu, T))
        grid_perc = perc.transform(stats.q_scores(det, encode(grid), mu, T))
        panels.append((label, grid_perc.reshape(RES, RES), auc))

    panel_grid(FIGS / "kernel_encoding_comparison.png", panels, xs, ys, data=train)

    print("Kernel vs encoding comparison (residual AUC, moon vs uniform background):")
    for label, _, auc in panels:
        print(f"  {label:<30} AUC = {auc:.3f}")
    print(f"  saved -> {FIGS / 'kernel_encoding_comparison.png'}")


if __name__ == "__main__":
    main()
