"""Experiment A: a kernel-PCA consistency check for the QSAD statistics.

3x3 headline grid on a one-class moon cloud.  Rows: linear Classical PCA
(rigid), the standard centered RBF kernel-PCA (strong classical nonlinear
baseline), and QSAD on a quantum feature map.  Columns: Q residual, T^2 leverage
(inside the Q-accepted region), and the combined max-percentile.

The goal is a CONSISTENCY check, not an accuracy claim: QSAD -- accessed by
sampling + threshold measurements -- reproduces the nonlinear spectral geometry
that classical kernel PCA obtains from the N x N Gram matrix.  RBF-KPCA and QSAD
use different (representative, strong) kernels, so their boundaries are similar
but not identical; run_dataset_comparison.py shows QSAD converges to RBF-KPCA
exactly when the kernel is matched.
"""

from pathlib import Path

import numpy as np

from qsad.core import (
    GAUSSIAN,
    ClassicalPCA,
    KernelPCANovelty,
    PercentileCalibrator,
    SpectralDetector,
    median_bandwidth,
)
from qsad.core import statistics as stats
from qsad.models import grid_points, moon_cloud
from qsad.models.encodings import ENCODINGS
from qsad.viz import comparison_grid

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "figures"

# --- experiment parameters ---
ENCODING = "fourier"     # fixed representative QSAD feature map
ALPHA = 0.88             # retained-mass / explained-variance target
T = 0.1                  # QSAD soft spectral resolution
GAMMA = 1e-3             # T^2 ridge
SHELLS = [1, 2, 3]       # unit-soft-rank shells for QSAD T^2
ACCEPT = 0.90            # Q-accept percentile (decision boundary)
RES = 320                # grid resolution per axis


def masked_t2(t2_perc, q_perc, accept=ACCEPT):
    """Show T^2 percentile only where Q accepts; elsewhere NaN (rendered white)."""
    out = t2_perc.copy()
    out[q_perc > accept] = np.nan
    return out


def method_panels(q_grid, q_ref, t2_grid, t2_ref, shape):
    """Image panels (Q, masked T^2, combined) and the contours to draw on each.

    The T^2 panel is imaged with its masked (white-outside) map but contoured
    with the combined map, so the cyan line is the final Q-and-T^2 accept region.
    """
    q = PercentileCalibrator(q_ref).transform(q_grid).reshape(shape)
    t2 = PercentileCalibrator(t2_ref).transform(t2_grid).reshape(shape)
    combined = np.maximum(q, t2)
    images = [q, masked_t2(t2, q), combined]
    contours = [q, combined, combined]
    return images, contours


def main():
    data = moon_cloud(n=450, noise=0.06, seed=0)
    grid, xs, ys = grid_points(RES)
    shape = (RES, RES)
    gamma = median_bandwidth(data)             # principled, label-free bandwidth

    # Row 1 -- linear PCA (centered, K=1): the residual is the off-axis distance.
    pca = ClassicalPCA(data)
    K = 1
    pca_img, pca_con = method_panels(pca.q_scores(grid, K), pca.q_scores(data, K),
                            pca.t2_scores(grid, K, GAMMA),
                            pca.t2_scores(data, K, GAMMA), shape)

    # Row 2 -- standard (centered) RBF kernel-PCA: the classical nonlinear baseline.
    kpca = KernelPCANovelty(data, gamma=gamma, alpha=ALPHA, center=True)
    kpca_img, kpca_con = method_panels(kpca.scores(grid), kpca.scores(data),
                             kpca.t2_scores(grid, GAMMA),
                             kpca.t2_scores(data, GAMMA), shape)

    # Row 3 -- QSAD on a quantum feature map (accessed by sampling, no Gram matrix).
    encode = ENCODINGS[ENCODING]
    F = encode(data)
    det = SpectralDetector.from_states(F, response=GAUSSIAN)
    mu = det.calibrate_mu(ALPHA, T)
    qsad_img, qsad_con = method_panels(stats.q_scores(det, encode(grid), mu, T),
                             stats.q_scores(det, F, mu, T),
                             stats.t2_scores(det, encode(grid), T, SHELLS, GAMMA),
                             stats.t2_scores(det, F, T, SHELLS, GAMMA), shape)

    out = FIGS / "experiment_A_grid.png"
    comparison_grid(
        out, [pca_img, kpca_img, qsad_img], xs, ys,
        row_labels=["Classical PCA", "Classical RBF kernel-PCA", f"QSAD ({ENCODING})"],
        col_labels=["$Q$ residual", "$T^2$ in $Q$-accepted", "combined max-percentile"],
        data=data, contour_grids=[pca_con, kpca_con, qsad_con],
        contour_level=ACCEPT, cbar_label="normal-calibrated percentile",
    )

    print("Experiment A  [linear PCA vs centered RBF kernel-PCA vs QSAD]")
    print(f"  classical PCA:  K={K} (linear), "
          f"explained variance={pca.eigvals[0] / pca.eigvals.sum():.3f}")
    print(f"  RBF kernel-PCA: gamma={gamma:.2f} (median heuristic), "
          f"retained components={kpca.n_comp}")
    print(f"  QSAD ({ENCODING}): alpha={ALPHA}, T={T}, mu={mu:.6f}, "
          f"occupations={np.round(det.occupations(mu, T)[:6], 3)}")
    print(f"  saved -> {out}")


if __name__ == "__main__":
    main()
