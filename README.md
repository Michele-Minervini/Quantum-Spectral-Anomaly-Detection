# Quantum Spectral Anomaly Detection (QSAD)

Numerical experiments for **QSAD**, a measurement-based one-class anomaly
detector that learns soft spectral tests from a nominal operator `C` and
produces quantum analogues of the classical PCA monitoring statistics:

- **`Q`** — residual support score, `Q = 1 − Tr(M σ)` (does the input leave the
  nominal subspace?)
- **`T²`** — shell-resolved Hotelling leverage (is the input atypical *within*
  the retained subspace?)

The detector is the **Regularized Spectral Detector**

```
M^f_{μ,T} = f( (C − μ I) / T ) = Σ_j f( (λ_j − μ) / T ) |u_j⟩⟨u_j|
```

where `f` is a smooth monotone response (Gaussian/probit `Φ` by default, or
logistic), `T` is the spectral resolution, and the threshold `μ` is calibrated
so the retained nominal mass `R(μ) = Tr(C M)` hits a target `α` (the quantum
analogue of an explained-variance level).

## Repository layout

```
src/qsad/
  core/      responses · detector · statistics (Q, T²) · classical_pca · kernel_pca · calibration
  models/    datasets (moons) · encodings (quantum feature maps) · spin_chains (TFIM)
  viz/       shared style · figure builders
scripts/     run_experiment_A.py · run_experiment_B.py · run_encoding_comparison.py
figures/     generated figures
results/     generated tables (CSV)
```

The core math engine (`qsad.core`) is independent of data generation
(`qsad.models`) and plotting (`qsad.viz`); the scripts only orchestrate.

## Installation

```bash
pip install -e .          # then `import qsad` works anywhere
# or, without installing:
pip install -r requirements.txt
PYTHONPATH=src python scripts/run_experiment_A.py
```

Requires Python ≥ 3.10 with numpy, scipy, scikit-learn, matplotlib. On a
headless machine, set `MPLBACKEND=Agg`.

## Experiments

### A — Curved boundaries (classical data via quantum feature states)

```bash
python scripts/run_experiment_A.py            # the 3x3 headline grid
python scripts/run_encoding_comparison.py     # 6 feature maps on one dataset
python scripts/run_dataset_comparison.py      # PCA vs kernel-PCA vs QSAD
```

A one-class moon cloud in `[0,1]²` is embedded into a quantum feature state. The
3×3 headline grid (rows: linear **Classical PCA**, **uncentered RBF kernel-PCA**,
**QSAD** on the matching `rff_gaussian` map; columns: `Q` residual, `T²`
leverage, combined) shows linear PCA is rigid while the two nonlinear rows are
**near-identical** (Spearman ≈ 0.99) — QSAD reproduces *uncentered* kernel PCA
with the same RBF kernel. Match the **kernel** (bandwidth `γ`), not the ambient
dimension: the encoding size `D` (qubits) is only the approximation resolution.
QSAD uses the uncentered mixture `C` (the paper's convention); the AUC table
below uses textbook **centered** kernel PCA as the strongest classical baseline.

![Experiment A](figures/experiment_A_grid.png)

**Encodings** (`qsad.models.encodings`, all swappable): `poly_amplitude`,
`angle`, `iqp`, `poly2`, `fourier`, `rff_gaussian`. The `fourier` map gives the
cleanest boundary and the highest AUC; `iqp` is expressive but fragmented.

**Datasets** (`qsad.models.datasets`): a fairness suite spanning linear PCA's
range — `gaussian_blob` (PCA-friendly), `moon` (curved), `ring` (PCA fails: its
slab runs through the empty centre).

**Fair baseline — read this before quoting Experiment A.** On classical data
QSAD *is* a kernel method, so the honest comparison is not linear PCA alone but
a classical kernel-PCA with a matched kernel. `run_dataset_comparison.py` adds
an RBF kernel-PCA novelty baseline:

![Dataset comparison](figures/dataset_comparison.png)

| dataset | Classical PCA (linear) | Classical KPCA (RBF) | QSAD (fourier) |
|---|---|---|---|
| blob | 0.89 | 0.93 | 0.93 |
| moon | 0.66 | 0.87 | 0.88 |
| ring | 0.57 | 0.82 | 0.88 |

Linear PCA is insufficient on curved/closed geometries; a strong *classical*
kernel method handles them, and QSAD only reaches **parity** with it. We make
**no classical-data accuracy claim** — Experiment A is a kernel-PCA sanity
check. In fact QSAD with `rff_gaussian` *converges* to classical RBF kernel-PCA
as the encoding grows (ring Spearman corr 0.75 → 0.87 → 0.93 at 4 → 6 → 8
qubits, with the median-heuristic bandwidth): the lift is the kernel, available
classically too. QSAD's actual
contribution is the quantum measurement **access model** and **quantum-native
data** (Experiment B).

### B — Quantum-native data (TFIM phases)

```bash
python scripts/run_experiment_B.py
```

The nominal source is a 70/20/10 mixture of the low-energy eigenstates of a
transverse-field Ising model deep in the **ordered phase** (parameter set A);
anomalies are low-energy states from the **paramagnetic phase** (set B). The
nominal mixture eigenvalues recover the mode weights (`≈ 0.70, 0.20, 0.10`).

| figure | content |
|---|---|
| `experiment_B_sweep.png`   | `Q` vs transverse field `h` — rises across the critical point `h_c` |
| `experiment_B_sectors.png` | per-sector scores: `Q_raw` flags the rare valid sector, `Q_QSAD` does not |
| `experiment_B_roc.png`     | ROC curves (n = 8) |
| `experiment_B_auc.csv`     | ROC-AUC per detector × system size |

![Experiment B sweep](figures/experiment_B_sweep.png)

**Key result.** `Q_QSAD` and `Q_hard` reach AUC ≈ 1.0; `Q_raw` is worse because
it is density-weighted and over-penalizes the rare-but-valid nominal sector
(mean score `0.90`, indistinguishable from a true anomaly at `0.94`), whereas the
calibrated `Q_QSAD` keeps it nominal (`0.08`). `T²` is a within-support leverage
diagnostic and intentionally does *not* separate out-of-support anomalies.

## Reference

See `Qsad1.tex` for the full theoretical framework (the detector derivation,
calibration protocols, and sharp-limit correspondence with classical `Q`/`T²`).
