"""Input-centering check for the classical-data experiment.

Re-runs the kernel-PCA baselines and the QSAD encodings with the raw inputs
replaced by inputs centered with the training-set mean, and reports the AUC
shift.  Translation-invariant kernels (RBF, Laplacian, and those induced by
the Fourier and random-feature maps) are unaffected by a common shift, so the
polynomial kernel and map are the cases the check actually probes.
"""

import csv
from functools import partial
from pathlib import Path

import numpy as np

from qsad.core import KernelPCANovelty, SpectralDetector, median_bandwidth, roc_auc
from qsad.core import statistics as stats
from qsad.models import DATASETS, uniform_anomalies
from qsad.models.encodings import ENCODINGS, rff_gaussian

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

ALPHA, T = 0.90, 0.05      # as in run_dataset_comparison.py


def split(dataset_fn):
    """Deterministic train / (nominal+anomaly) test split for one dataset."""
    train = dataset_fn(n=500, seed=0)
    nominal = dataset_fn(n=300, seed=7)
    anomaly = uniform_anomalies(n=300, seed=11)
    test = np.vstack([nominal, anomaly])
    labels = np.r_[np.zeros(len(nominal)), np.ones(len(anomaly))]
    return train, test, labels


def kpca_auc(train, test, labels, kernel, gamma):
    det = KernelPCANovelty(train, gamma=gamma, alpha=ALPHA, kernel=kernel)
    return roc_auc(labels, det.scores(test))


def qsad_auc(train, test, labels, encode):
    det = SpectralDetector.from_states(encode(train))
    mu = det.calibrate_mu(ALPHA, T)
    return roc_auc(labels, stats.q_scores(det, encode(test), mu, T))


def main():
    rows = []
    print(f"{'dataset':>7} | {'method':>17} | {'raw':>6} | {'centered':>8} | {'delta':>7}")
    print("-" * 60)
    for dname, fn in DATASETS.items():
        train, test, labels = split(fn)
        mean = train.mean(axis=0)
        g2 = median_bandwidth(train)                       # shift-invariant
        g1 = median_bandwidth(train, metric="cityblock")
        methods = {
            "KPCA rbf": lambda tr, te: kpca_auc(tr, te, labels, "rbf", g2),
            "KPCA laplacian": lambda tr, te: kpca_auc(tr, te, labels, "laplacian", g1),
            "KPCA poly": lambda tr, te: kpca_auc(tr, te, labels, "poly", 1.0),
            "QSAD fourier": lambda tr, te: qsad_auc(tr, te, labels, ENCODINGS["fourier"]),
            "QSAD rff_gaussian": lambda tr, te: qsad_auc(
                tr, te, labels, partial(rff_gaussian, gamma=g2)),
            "QSAD poly2": lambda tr, te: qsad_auc(tr, te, labels, ENCODINGS["poly2"]),
        }
        for mname, auc_fn in methods.items():
            raw = auc_fn(train, test)
            cen = auc_fn(train - mean, test - mean)
            rows.append([dname, mname, f"{raw:.4f}", f"{cen:.4f}", f"{cen - raw:+.4f}"])
            print(f"{dname:>7} | {mname:>17} | {raw:>6.4f} | {cen:>8.4f} | {cen - raw:>+7.4f}")

    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "centering_check.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["dataset", "method", "auc_raw", "auc_centered", "delta"])
        writer.writerows(rows)
    print(f"\nsaved -> {RESULTS / 'centering_check.csv'}")


if __name__ == "__main__":
    main()
