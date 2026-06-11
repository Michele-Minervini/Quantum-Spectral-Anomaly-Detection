"""Experiment B: quantum-native anomaly detection on the TFIM.

Nominal = the M lowest eigenstates of an ordered-phase TFIM populated with
geometrically decreasing weights (a thermal-like occupation), so the nominal
spectrum is graded and crosses the 1/N sampling resolution inside the ladder:
no rank K is privileged.  Anomalies = low-energy states of the paramagnetic
phase just across the critical point.  The experiment compares the calibrated
soft detector (one retained-mass target alpha) against hard top-K projectors
across K, against the naive density-weighted overlap, and across the quantum
phase transition.
"""

import csv
from pathlib import Path

import numpy as np

from qsad.core import GAUSSIAN, LOGISTIC, SpectralDetector, roc_auc
from qsad.core import statistics as stats
from qsad.models import low_energy_states, perturbed_samples, tfim_hamiltonian
from qsad.viz import (auc_curve, mode_profile, sector_temperature,
                      spectrum_occupations, temperature_sweep)

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "figures"
RESULTS = ROOT / "results"

# --- parameters ---
N_SPINS = 10
J, H_A, H_B, H_C = 1.0, 0.4, 1.2, 1.0    # ordered nominal / near-critical anomaly
M_LADDER = 12                # nominal ladder depth
LADDER_RATIO = 0.5           # thermal-like mode weights w_j ~ ratio^j
ANOM_WEIGHTS = [0.7, 0.2, 0.1]
ALPHA, T_MAIN, GAMMA = 0.99, 3e-3, 1e-4
SHELLS = [1, 2, 3]
N_TRAIN, N_TEST, NOISE = 600, 300, 0.05
T_FAMILY = [3e-2, 1e-2, 3e-3, 1e-3]      # soft -> sharp
K_SWEEP = [2, 4, 6, 8, 10, 12]
K_SHOW = [4, 6, 8]           # hard ranks displayed in the mode profile
RARE_MODE = 5                # 0-based index of the rare-but-valid demo sector


def k_alpha(det, alpha):
    """Explained-variance rank: smallest K whose cumulative mass reaches alpha."""
    frac = np.cumsum(det.eigvals) / det.eigvals.sum()
    return int(np.searchsorted(frac, alpha) + 1)


def build():
    """Train the detector on the graded ladder and assemble all test sets."""
    modes = low_energy_states(tfim_hamiltonian(N_SPINS, J, H_A), M_LADDER)
    weights = LADDER_RATIO ** np.arange(M_LADDER, dtype=float)
    weights /= weights.sum()
    amodes = low_energy_states(tfim_hamiltonian(N_SPINS, J, H_B), len(ANOM_WEIGHTS))

    train = perturbed_samples(modes, weights, N_TRAIN, NOISE, seed=1)
    nominal = perturbed_samples(modes, weights, 2 * N_TEST, NOISE, seed=2)
    anomaly = perturbed_samples(amodes, ANOM_WEIGHTS, N_TEST, NOISE, seed=3)
    per_mode = [perturbed_samples(modes[[j]], [1.0], N_TEST, NOISE, seed=10 + j)
                for j in range(M_LADDER)]

    detectors = {r.name: SpectralDetector.from_states(train, response=r)
                 for r in (GAUSSIAN, LOGISTIC)}
    return {"modes": modes, "weights": weights, "train": train,
            "nominal": nominal, "anomaly": anomaly, "per_mode": per_mode,
            "det": detectors["gaussian"], "detectors": detectors}


def auc_table(info):
    """ROC-AUC of every detector on the nominal-mixture vs anomaly pool."""
    det, nominal, anomaly = info["det"], info["nominal"], info["anomaly"]
    pool = np.vstack([nominal, anomaly])
    labels = np.r_[np.zeros(len(nominal)), np.ones(len(anomaly))]
    out = {"Q_raw": roc_auc(labels, stats.raw_scores(det, pool))}
    for K in K_SWEEP:
        out[f"Q_hard K={K}"] = roc_auc(labels, stats.hard_scores(det, pool, K))
    for name, d in info["detectors"].items():
        mu = d.calibrate_mu(ALPHA, T_MAIN)
        out[f"Q_QSAD ({name})"] = roc_auc(labels, stats.q_scores(d, pool, mu, T_MAIN))
    out["T^2"] = roc_auc(labels, stats.t2_scores(det, pool, T_MAIN, SHELLS, GAMMA))
    return out


def mode_profiles(info):
    """Per-mode mean scores for the hard ranks and the soft T family."""
    det, per_mode, anomaly = info["det"], info["per_mode"], info["anomaly"]
    hard = {K: np.array([stats.hard_scores(det, g, K).mean() for g in per_mode])
            for K in K_SHOW}
    anom_hard = {K: stats.hard_scores(det, anomaly, K).mean() for K in K_SHOW}
    soft, anom_soft = {}, {}
    for t in T_FAMILY:
        mu = det.calibrate_mu(ALPHA, t)
        soft[t] = np.array([stats.q_scores(det, g, mu, t).mean() for g in per_mode])
        anom_soft[t] = stats.q_scores(det, anomaly, mu, t).mean()
    return hard, soft, anom_hard, anom_soft


def sector_scores(info, k_values):
    """Mean scores on the dominant / rare / anomaly sectors."""
    det = info["det"]
    groups = [info["per_mode"][0], info["per_mode"][RARE_MODE], info["anomaly"]]
    q_raw = np.array([stats.raw_scores(det, g).mean() for g in groups])
    q_hard = {K: np.array([stats.hard_scores(det, g, K).mean() for g in groups])
              for K in k_values}
    q_soft = {t: np.array([stats.q_scores(det, g, det.calibrate_mu(ALPHA, t), t).mean()
                           for g in groups]) for t in T_FAMILY}
    return q_raw, q_hard, q_soft


def field_sweep_temperatures(info, hs, k_hard):
    """Score the TFIM ground state across fields at several resolutions T.

    Ground states are T-independent, so they are computed once; only mu and
    the occupations change with T.  Returns ({T: Q array}, Q_hard, Q_raw).
    """
    det = info["det"]
    gs = np.array([low_energy_states(tfim_hamiltonian(N_SPINS, J, h), 1)[0]
                   for h in hs])
    q_hard = stats.hard_scores(det, gs, k_hard)
    q_raw = stats.raw_scores(det, gs)
    q_by_T = {t: stats.q_scores(det, gs, det.calibrate_mu(ALPHA, t), t)
              for t in T_FAMILY}
    return q_by_T, q_hard, q_raw


def auc_vs_temperature(info, t_values, shot_counts, n_draws=25, seed=0):
    """Detection AUC vs T at infinite and finite measurement-shot budgets.

    QSAD estimates the acceptance ``<psi|M|psi>`` by sampling, so finite shots
    add binomial noise to the score -- a sharper (smaller T) detector is more
    robust to it.  Returns ``{label: (mean, std_or_None)}``.
    """
    det = info["det"]
    pool = np.vstack([info["nominal"], info["anomaly"]])
    labels = np.r_[np.zeros(len(info["nominal"])), np.ones(len(info["anomaly"]))]
    pops = np.abs(pool @ det.eigvecs.conj()) ** 2
    acc = {t: pops @ det.occupations(det.calibrate_mu(ALPHA, t), t)
           for t in t_values}
    out = {r"$\infty$ shots": (np.array([roc_auc(labels, 1.0 - acc[t])
                                         for t in t_values]), None)}
    rng = np.random.default_rng(seed)
    for m in shot_counts:
        draws = np.array([[roc_auc(labels, 1.0 - rng.binomial(m, np.clip(acc[t], 0, 1)) / m)
                           for _ in range(n_draws)] for t in t_values])
        out[f"{m} shots"] = (draws.mean(axis=1), draws.std(axis=1))
    return out


def main():
    info = build()
    det, weights = info["det"], info["weights"]
    k99 = k_alpha(det, ALPHA)

    print("Experiment B: graded nominal ladder (n=%d spins)" % N_SPINS)
    print("mode weights x N_train:",
          " ".join(f"{w * N_TRAIN:.0f}" for w in weights))
    print("C eigenvalues:", " ".join(f"{v:.1e}" for v in det.eigvals[:M_LADDER + 2]))
    print(f"sampling resolution 1/N = {1 / N_TRAIN:.1e}")
    for a in (0.95, 0.98, 0.99, 0.995):
        print(f"  explained-variance rank K({a}) = {k_alpha(det, a)}")

    # --- ROC-AUC table ---
    aucs = auc_table(info)
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "experiment_B_auc.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["detector", "auc"])
        print("\nROC-AUC (nominal mixture vs near-critical anomaly):")
        for name, auc in aucs.items():
            writer.writerow([name, f"{auc:.4f}"])
            print(f"  {name:<18} {auc:.4f}")

    # --- per-mode profile: hard top-K vs the soft family ---
    hard, soft, anom_hard, anom_soft = mode_profiles(info)
    counts = [f"{w * N_TRAIN:.0f}" for w in weights]
    mode_profile(FIGS / "experiment_B_mode_profile.png", hard, soft, T_FAMILY,
                 anom_hard, anom_soft, counts)

    print(f"\nRare-but-valid mode m{RARE_MODE + 1} "
          f"(weight {weights[RARE_MODE]:.3f}, "
          f"~{weights[RARE_MODE] * N_TRAIN:.0f} training samples):")
    for K in K_SHOW:
        print(f"  Q_hard K={K}: {hard[K][RARE_MODE]:.2f}"
              f"   (anomaly: {anom_hard[K]:.2f})")
    for t in T_FAMILY:
        print(f"  Q_QSAD T={t:g}: {soft[t][RARE_MODE]:.2f}"
              f"   (anomaly: {anom_soft[t]:.2f})")

    # --- spectrum with soft occupations and the hard cutoff ---
    occ = {t: det.occupations(det.calibrate_mu(ALPHA, t), t) for t in T_FAMILY}
    spectrum_occupations(FIGS / "experiment_B_spectrum.png", det.eigvals, occ,
                         T_FAMILY, k99, N_TRAIN)

    # --- sectors: dominant / rare / anomaly, raw vs hard vs soft ---
    q_raw_s, q_hard_s, q_soft_s = sector_scores(info, k_values=[4, 6])
    sector_temperature(FIGS / "experiment_B_sectors_temperature.png",
                       [f"dominant\nnominal\n({weights[0]:.0%})",
                        f"rare\nnominal\n({weights[RARE_MODE]:.1%})",
                        "anomaly\n(paramagnetic)"],
                       q_raw_s, q_soft_s, T_FAMILY, q_hard_by_K=q_hard_s)

    # --- phase sweep at several resolutions ---
    hs = np.linspace(0.05, 2.0, 60)
    q_by_T, q_hard_sw, q_raw_sw = field_sweep_temperatures(info, hs, k99)
    temperature_sweep(FIGS / "experiment_B_temperature_sweep.png", hs, q_by_T,
                      T_FAMILY, q_hard_sw, q_raw_sw, H_C,
                      train_band=(H_A - 0.02, H_A + 0.02))

    # --- detection AUC vs resolution T at finite measurement shots ---
    t_auc = list(np.geomspace(3e-4, 1e-1, 12))
    auc_curve(FIGS / "experiment_B_auc_vs_T.png", t_auc,
              auc_vs_temperature(info, t_auc, [200, 50, 20]), aucs["Q_raw"])

    for name in ("experiment_B_mode_profile", "experiment_B_spectrum",
                 "experiment_B_sectors_temperature",
                 "experiment_B_temperature_sweep", "experiment_B_auc_vs_T"):
        print(f"saved -> {FIGS / (name + '.png')}")
    print(f"saved -> {RESULTS / 'experiment_B_auc.csv'}")


if __name__ == "__main__":
    main()
