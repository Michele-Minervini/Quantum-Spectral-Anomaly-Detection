"""Fair PCA vs kernel-PCA vs QSAD comparison across dataset geometries.

QSAD on classical data is a kernel method, so the honest baseline is not just
linear PCA but a classical kernel-PCA with the *same* kernel.  This script:

1. reports a dataset x method ROC-AUC table including a classical RBF kernel-PCA
   novelty detector (a strong, matched, nonlinear classical baseline);
2. shows that QSAD with the ``rff_gaussian`` map *converges* to that classical
   RBF kernel-PCA as the encoding dimension (qubits) grows -- i.e. QSAD's edge
   over linear PCA is the kernel lift, not an unfair dimensionality advantage,
   and on classical data it claims no accuracy advantage over a good classical
   kernel method.
"""

import csv
from functools import partial
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from qsad.core import (
    ClassicalPCA,
    KernelPCANovelty,
    PercentileCalibrator,
    SpectralDetector,
    median_bandwidth,
    roc_auc,
)
from qsad.core import statistics as stats
from qsad.models import DATASETS, grid_points, uniform_anomalies
from qsad.models.encodings import ENCODINGS, rff_gaussian
from qsad.viz import comparison_grid

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "figures"
RESULTS = ROOT / "results"

ALPHA, T, K_PCA, RES = 0.90, 0.05, 1, 200
CONV_DIMS = [16, 64, 256]   # rff feature dimensions (4, 6, 8 qubits)


def split(dataset_fn):
    """Deterministic train / (nominal+anomaly) test split for one dataset."""
    train = dataset_fn(n=500, seed=0)
    nominal = dataset_fn(n=300, seed=7)
    anomaly = uniform_anomalies(n=300, seed=11)
    test = np.vstack([nominal, anomaly])
    labels = np.r_[np.zeros(len(nominal)), np.ones(len(anomaly))]
    return train, test, labels


def evaluate(train, test, labels, grid):
    """AUC and percentile grids for linear PCA, RBF kernel-PCA, and QSAD."""
    def pgrid(ref, vals):
        return PercentileCalibrator(ref).transform(vals).reshape(RES, RES)

    gamma = median_bandwidth(train)              # principled per-dataset bandwidth
    aucs, grids = {}, {}
    pca = ClassicalPCA(train)
    aucs["Classical PCA"] = roc_auc(labels, pca.q_scores(test, K_PCA))
    grids["Classical PCA"] = pgrid(pca.q_scores(train, K_PCA), pca.q_scores(grid, K_PCA))

    kpca = KernelPCANovelty(train, gamma=gamma, alpha=ALPHA)
    aucs["Classical KPCA"] = roc_auc(labels, kpca.scores(test))
    grids["Classical KPCA"] = pgrid(kpca.scores(train), kpca.scores(grid))

    for name, encode in ENCODINGS.items():
        enc = partial(rff_gaussian, gamma=gamma) if name == "rff_gaussian" else encode
        det = SpectralDetector.from_states(enc(train))
        mu = det.calibrate_mu(ALPHA, T)
        aucs[f"QSAD/{name}"] = roc_auc(labels, stats.q_scores(det, enc(test), mu, T))
        grids[f"QSAD/{name}"] = pgrid(stats.q_scores(det, enc(train), mu, T),
                                      stats.q_scores(det, enc(grid), mu, T))
    return aucs, grids


def convergence_row(train, test):
    """Spearman corr of QSAD(rff, D) vs exact RBF-KPCA on the test set, per D.

    The target uses the uncentered convention, matching QSAD's mixture ``C``.
    """
    gamma = median_bandwidth(train)
    target = KernelPCANovelty(train, gamma=gamma, alpha=ALPHA,
                              center=False).scores(test)
    corrs = []
    for D in CONV_DIMS:
        det = SpectralDetector.from_states(rff_gaussian(train, n_features=D, gamma=gamma))
        mu = det.calibrate_mu(ALPHA, T)
        q = stats.q_scores(det, rff_gaussian(test, n_features=D, gamma=gamma), mu, T)
        corrs.append(spearmanr(q, target)[0])
    return corrs


def main():
    grid, xs, ys = grid_points(RES)
    results, grids, data = {}, {}, {}
    for name, fn in DATASETS.items():
        train, test, labels = split(fn)
        results[name], grids[name] = evaluate(train, test, labels, grid)
        data[name] = (train, test)

    methods = list(next(iter(results.values())))
    print("Dataset x method ROC-AUC (nominal vs uniform anomalies):\n")
    header = f"{'dataset':>7} | " + " | ".join(f"{m:>15}" for m in methods)
    print(header)
    print("-" * len(header))
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "dataset_comparison_auc.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["dataset"] + methods)
        for d in DATASETS:
            row = [results[d][m] for m in methods]
            writer.writerow([d] + [f"{a:.4f}" for a in row])
            print(f"{d:>7} | " + " | ".join(f"{a:>15.3f}" for a in row))

    print("\nQSAD(rff) -> classical RBF-KPCA as the encoding grows "
          "(test-set Spearman corr):")
    print(f"{'dataset':>7} | " + " | ".join(f"D={d} ({int(np.log2(d))}q)"
                                            for d in CONV_DIMS))
    for d in DATASETS:
        corrs = convergence_row(*data[d])
        print(f"{d:>7} | " + " | ".join(f"{c:>9.3f}" for c in corrs))
    print("=> QSAD on classical data IS kernel PCA; the gap at few qubits is "
          "finite-encoding resolution, not a quantum advantage.")

    encs = [m for m in methods if m.startswith("QSAD/")]
    best = max(encs, key=lambda m: np.mean([results[d][m] for d in DATASETS]))
    print(f"\nBest QSAD encoding by mean AUC: {best.split('/')[1]}")

    # Montage: linear PCA vs strong classical kernel PCA vs best QSAD kernel.
    cols = list(DATASETS)
    rows = ["Classical PCA", "Classical KPCA", best]
    labels = ["Classical PCA", "Classical KPCA (RBF)", f"QSAD ({best.split('/')[1]})"]
    montage = [[grids[d][r] for d in cols] for r in rows]
    col_data = [data[d][0] for d in cols]
    comparison_grid(FIGS / "dataset_comparison.png", montage, xs, ys,
                    row_labels=labels, col_labels=cols, col_data=col_data)

    print(f"\nsaved -> {FIGS/'dataset_comparison.png'}")
    print(f"saved -> {RESULTS/'dataset_comparison_auc.csv'}")


if __name__ == "__main__":
    main()
