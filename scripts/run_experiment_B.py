"""Experiment B: quantum-native anomaly detection on the TFIM.

Nominal (parameter set A) = a 70/20/10 mixture of the low-energy eigenstates of
a TFIM deep in the ordered phase.  Anomalies (parameter set B) = low-energy
states of a TFIM in the paramagnetic phase.  Reports ROC-AUC for the QSAD
support detectors and a score-vs-field diagnostic across the critical point.
"""

import csv
from pathlib import Path

import numpy as np

from qsad.core import GAUSSIAN, LOGISTIC, SpectralDetector, roc_auc, roc_points
from qsad.core import statistics as stats
from qsad.models import low_energy_states, perturbed_samples, tfim_hamiltonian
from qsad.viz import (auc_curve, roc_panel, score_distributions,
                      sector_distributions, sector_temperature, temperature_sweep)

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "figures"
RESULTS = ROOT / "results"

# --- parameters ---
J, H_A, H_B, H_C = 1.0, 0.4, 1.2, 1.0     # ordered nominal / (subtle) paramagnetic anomaly
WEIGHTS = [0.7, 0.2, 0.1]                  # nominal mode proportions
N_MODES = 3
# alpha < the eigenvalue mass of the 3 physical modes (~0.997): retain the real
# modes without diving into the diffuse noise floor, which would accept anomalies.
ALPHA, T, GAMMA = 0.99, 0.1, 1e-4
SHELLS = [1, 2, 3]
N_TRAIN, N_TEST, NOISE = 600, 300, 0.05
SIZES = [8, 10]
T_SWEEP = list(np.geomspace(0.3, 0.004, 7))   # log-spaced resolutions, soft -> sharp


def build_size(n):
    """Train the detector and score a nominal-vs-anomaly test set for size n."""
    modes_A = low_energy_states(tfim_hamiltonian(n, J, H_A), N_MODES)
    modes_B = low_energy_states(tfim_hamiltonian(n, J, H_B), N_MODES)

    train = perturbed_samples(modes_A, WEIGHTS, N_TRAIN, NOISE, seed=1)
    nominal = perturbed_samples(modes_A, WEIGHTS, N_TEST, NOISE, seed=2)
    anomaly = perturbed_samples(modes_B, WEIGHTS, N_TEST, NOISE, seed=3)
    test = np.vstack([nominal, anomaly])
    labels = np.r_[np.zeros(N_TEST), np.ones(N_TEST)]

    detectors = {r.name: SpectralDetector.from_states(train, response=r)
                 for r in (GAUSSIAN, LOGISTIC)}
    mus = {name: d.calibrate_mu(ALPHA, T) for name, d in detectors.items()}
    det = detectors["gaussian"]

    mu_g = mus["gaussian"]
    scores = {
        "Q_raw": stats.raw_scores(det, test),
        "Q_hard": stats.hard_scores(det, test, N_MODES),
        "Q_QSAD (Gaussian)": stats.q_scores(det, test, mu_g, T),
        "Q_QSAD (logistic)": stats.q_scores(detectors["logistic"], test,
                                            mus["logistic"], T),
        "T^2": stats.t2_scores(det, test, T, SHELLS, GAMMA),
    }
    aucs = {name: roc_auc(labels, s) for name, s in scores.items()}

    # Per-sector groups: dominant nominal mode (70%), rare nominal mode (10%),
    # and the anomaly.  Q_raw over-penalizes the rare-but-valid sector.
    dominant = perturbed_samples(modes_A[[0]], [1.0], N_TEST, NOISE, seed=4)
    rare = perturbed_samples(modes_A[[2]], [1.0], N_TEST, NOISE, seed=5)
    sectors = {
        "$Q_{raw}$": [stats.raw_scores(det, g) for g in (dominant, rare, anomaly)],
        "$Q_{QSAD}$": [stats.q_scores(det, g, mu_g, T)
                       for g in (dominant, rare, anomaly)],
    }
    return {"n": n, "det": det, "mu": mu_g, "scores": scores, "labels": labels,
            "test": test, "aucs": aucs, "eigvals": det.eigvals, "sectors": sectors}


def field_sweep(info, hs):
    """Score the TFIM ground state vs transverse field with a fixed detector."""
    det, mu, n = info["det"], info["mu"], info["n"]
    raw, hard, qsad = [], [], []
    for h in hs:
        gs = low_energy_states(tfim_hamiltonian(n, J, h), 1)
        raw.append(stats.raw_scores(det, gs)[0])
        hard.append(stats.hard_scores(det, gs, N_MODES)[0])
        qsad.append(stats.q_scores(det, gs, mu, T)[0])
    # Order so the dashed Q_hard is drawn last (it coincides with Q_QSAD).
    return {"$Q_{raw}$": np.array(raw), "$Q_{QSAD}$": np.array(qsad),
            "$Q_{hard}$": np.array(hard)}


def field_sweep_temperatures(info, hs, t_values):
    """Score the TFIM ground state across fields at several resolutions T.

    Ground states are T-independent, so they are computed once; only mu and the
    occupations change with T.  Returns ({T: Q array}, Q_hard, Q_raw); as T -> 0
    the QSAD curves converge to Q_hard.
    """
    det, n = info["det"], info["n"]
    gs = np.array([low_energy_states(tfim_hamiltonian(n, J, h), 1)[0] for h in hs])
    q_hard = stats.hard_scores(det, gs, N_MODES)
    q_raw = stats.raw_scores(det, gs)
    q_by_T = {t: stats.q_scores(det, gs, det.calibrate_mu(ALPHA, t), t)
              for t in t_values}
    return q_by_T, q_hard, q_raw


def auc_vs_temperature(info, t_values, shot_counts, n_draws=25, seed=0):
    """Detection AUC vs T at infinite and finite measurement-shot budgets.

    QSAD estimates the acceptance ``<psi|M|psi>`` by sampling, so finite shots
    add binomial noise to the score -- a sharper (smaller T) detector is more
    robust to it.  Returns ``{label: (mean, std_or_None)}``.
    """
    det, test, labels = info["det"], info["test"], info["labels"]
    pops = np.abs(test @ det.eigvecs.conj()) ** 2
    acc = {t: pops @ det.occupations(det.calibrate_mu(ALPHA, t), t) for t in t_values}
    out = {r"$\infty$ shots": (np.array([roc_auc(labels, 1.0 - acc[t])
                                         for t in t_values]), None)}
    rng = np.random.default_rng(seed)
    for m in shot_counts:
        draws = np.array([[roc_auc(labels, 1.0 - rng.binomial(m, np.clip(acc[t], 0, 1)) / m)
                           for _ in range(n_draws)] for t in t_values])
        out[f"{m} shots"] = (draws.mean(axis=1), draws.std(axis=1))
    return out


def sector_temperatures(info, t_values):
    """Mean Q_raw and Q_QSAD(T) for the three sectors (dominant/rare/anomaly)."""
    det, n = info["det"], info["n"]
    modes_A = low_energy_states(tfim_hamiltonian(n, J, H_A), N_MODES)
    modes_B = low_energy_states(tfim_hamiltonian(n, J, H_B), N_MODES)
    groups = [perturbed_samples(modes_A[[0]], [1.0], N_TEST, NOISE, seed=4),
              perturbed_samples(modes_A[[2]], [1.0], N_TEST, NOISE, seed=5),
              perturbed_samples(modes_B, WEIGHTS, N_TEST, NOISE, seed=3)]
    q_raw = np.array([stats.raw_scores(det, g).mean() for g in groups])
    q_qsad = {t: np.array([stats.q_scores(det, g, det.calibrate_mu(ALPHA, t), t).mean()
                           for g in groups]) for t in t_values}
    return q_raw, q_qsad


def main():
    runs = {n: build_size(n) for n in SIZES}

    # --- ROC-AUC table ---
    detector_names = list(runs[SIZES[0]]["aucs"])
    print("Experiment B: ROC-AUC (label 1 = paramagnetic-phase anomaly)\n")
    header = f"{'n':>3} | " + " | ".join(f"{d:>17}" for d in detector_names)
    print(header)
    print("-" * len(header))
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "experiment_B_auc.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["n"] + detector_names)
        for n in SIZES:
            row = [runs[n]["aucs"][d] for d in detector_names]
            writer.writerow([n] + [f"{a:.4f}" for a in row])
            print(f"{n:>3} | " + " | ".join(f"{a:>17.4f}" for a in row))

    big = runs[SIZES[-1]]
    det, mu = big["det"], big["mu"]
    print("\nNominal mixture C leading eigenvalues (n=%d): %s"
          % (big["n"], np.round(big["eigvals"][:6], 4)))
    print("QSAD: alpha=%g, T=%g, mu=%.6f, occupations=%s"
          % (ALPHA, T, mu, np.round(det.occupations(mu, T)[:6], 3)))

    # --- per-sector mean scores: the rare-but-valid sector ---
    sector_labels = ["dominant nominal (70%)", "rare nominal (10%)", "anomaly"]
    print("\nMean anomaly score by sector (n=%d):" % big["n"])
    print(f"{'detector':>12} | " + " | ".join(f"{s:>22}" for s in sector_labels))
    for name, groups in big["sectors"].items():
        means = " | ".join(f"{np.mean(g):>22.3f}" for g in groups)
        print(f"{name.strip('$'):>12} | {means}")
    print("Q_raw flags the rare valid sector as strongly as the anomaly;"
          " Q_QSAD keeps it nominal.")
    print("T^2 is a within-support leverage diagnostic, so it does not separate"
          " out-of-support anomalies.")

    # --- figures: phase sweep, per-sector violins, ROC ---
    hs = np.linspace(0.05, 2.0, 60)
    score_distributions(FIGS / "experiment_B_sweep.png", hs, field_sweep(big, hs),
                        H_C, train_band=(H_A - 0.06, H_A + 0.06),
                        styles={"$Q_{hard}$": "--"})

    # Temperature sweep: QSAD across the transition at several T, converging to
    # the hard projector as T -> 0 (the sharp-limit M -> P_K).
    q_by_T, q_hard_sw, q_raw_sw = field_sweep_temperatures(big, hs, T_SWEEP)
    temperature_sweep(FIGS / "experiment_B_temperature_sweep.png", hs, q_by_T,
                      T_SWEEP, q_hard_sw, q_raw_sw, H_C,
                      train_band=(H_A - 0.06, H_A + 0.06))

    # (a) detection AUC vs resolution T, at infinite and finite measurement shots
    t_auc = list(np.geomspace(0.005, 3.0, 12))
    auc_curve(FIGS / "experiment_B_auc_vs_T.png", t_auc,
              auc_vs_temperature(big, t_auc, [200, 50, 20]), big["aucs"]["Q_raw"])

    # combined sectors: Q_raw bar + Q_QSAD bars at several temperatures
    t_sect = list(np.geomspace(0.3, 0.004, 5))
    q_raw_s, q_qsad_s = sector_temperatures(big, t_sect)
    sector_temperature(FIGS / "experiment_B_sectors_temperature.png",
                       ["dominant\nnominal\n(70%)", "rare\nnominal\n(10%)",
                        "anomaly\n(phase B)"], q_raw_s, q_qsad_s, t_sect)

    sector_distributions(FIGS / "experiment_B_sectors.png", big["sectors"],
                         ["dominant\nnominal\n(70%)", "rare\nnominal\n(10%)",
                          "anomaly\n(phase B)"],
                         group_colors=["#4c72b0", "#4c72b0", "#c44e52"])

    # ROC at the smaller size, where Q_raw is visibly imperfect.
    small = runs[SIZES[0]]
    roc_curves = {}
    for name in ["Q_raw", "Q_hard", "Q_QSAD (Gaussian)"]:
        fpr, tpr = roc_points(small["labels"], small["scores"][name])
        roc_curves[name] = (fpr, tpr, small["aucs"][name])
    roc_panel(FIGS / "experiment_B_roc.png", roc_curves)

    print(f"\nsaved -> {FIGS/'experiment_B_sweep.png'}")
    print(f"saved -> {FIGS/'experiment_B_temperature_sweep.png'}")
    print(f"saved -> {FIGS/'experiment_B_auc_vs_T.png'}")
    print(f"saved -> {FIGS/'experiment_B_sectors.png'}")
    print(f"saved -> {FIGS/'experiment_B_sectors_temperature.png'}")
    print(f"saved -> {FIGS/'experiment_B_roc.png'} (n={small['n']})")
    print(f"saved -> {RESULTS/'experiment_B_auc.csv'}")


if __name__ == "__main__":
    main()
