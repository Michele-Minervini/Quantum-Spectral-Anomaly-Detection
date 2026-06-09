"""Compare quantum feature maps for Experiment A.

For each encoding it builds the QSAD detector on the moon cloud, scores a held-out
nominal set against uniform-background anomalies (ROC-AUC), and renders the QSAD
Q-percentile map.  Use it to justify the encoding chosen for the headline grid.
"""

from pathlib import Path

import numpy as np

from qsad.core import GAUSSIAN, PercentileCalibrator, SpectralDetector, roc_auc
from qsad.core import statistics as stats
from qsad.models import grid_points, moon_cloud, uniform_anomalies
from qsad.models.encodings import ENCODINGS
from qsad.viz import encoding_comparison

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "figures"

ALPHA, T, RES = 0.88, 0.035, 280


def main():
    train = moon_cloud(n=450, noise=0.06, seed=0)
    nominal = moon_cloud(n=300, noise=0.06, seed=7)      # held-out nominal
    anomaly = uniform_anomalies(n=300, seed=11)
    grid, xs, ys = grid_points(RES)
    labels = np.r_[np.zeros(len(nominal)), np.ones(len(anomaly))]

    panels, results = [], {}
    for name, encode in ENCODINGS.items():
        det = SpectralDetector.from_states(encode(train), response=GAUSSIAN)
        mu = det.calibrate_mu(ALPHA, T)
        q_test = stats.q_scores(det, encode(np.vstack([nominal, anomaly])), mu, T)
        auc = roc_auc(labels, q_test)
        q_grid = stats.q_scores(det, encode(grid), mu, T)
        perc = PercentileCalibrator(stats.q_scores(det, encode(train), mu, T))
        panels.append((name, perc.transform(q_grid).reshape(RES, RES), auc))
        results[name] = auc

    encoding_comparison(FIGS / "encoding_comparison.png", panels, xs, ys, data=train)

    best = max(results, key=results.get)
    print("Encoding comparison (QSAD-Q AUC, moon vs uniform background):")
    for name, auc in sorted(results.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<16} AUC = {auc:.3f}")
    print(f"  best -> {best}")
    print(f"  saved -> {FIGS/'encoding_comparison.png'}")


if __name__ == "__main__":
    main()
