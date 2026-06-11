"""Figure builders for the QSAD experiments."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

from .style import ACCENT, NOMINAL_PT, PERCENTILE_CMAP, use_style


def panel_grid(fig_path, panels, xs, ys, data=None, contour_level=0.9,
               ncols=3, cbar_label="$Q$ percentile"):
    """Grid of residual-percentile maps, one per method, with its AUC in the title."""
    use_style()
    extent = [xs[0], xs[-1], ys[0], ys[-1]]
    n = len(panels)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.7 * nrows),
                             constrained_layout=True, squeeze=False)
    im = None
    for k in range(nrows * ncols):
        ax = axes[k // ncols][k % ncols]
        if k >= n:
            ax.axis("off")
            continue
        name, grid, auc = panels[k]
        im = ax.imshow(grid, origin="lower", extent=extent, vmin=0.0, vmax=1.0,
                       cmap=PERCENTILE_CMAP, aspect="auto", interpolation="bilinear")
        ax.contour(grid, levels=[contour_level], extent=extent,
                   colors=ACCENT, linewidths=1.3)
        if data is not None:
            ax.scatter(data[:, 0], data[:, 1], s=3, c=NOMINAL_PT, alpha=0.55,
                       linewidths=0)
        ax.set_title(f"{name}\nAUC = {auc:.3f}")
        if k // ncols == nrows - 1:
            ax.set_xlabel("$x_1$")
        if k % ncols == 0:
            ax.set_ylabel("$x_2$")
    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.015)
    cbar.set_label(cbar_label)
    fig.savefig(fig_path)
    plt.close(fig)


def comparison_grid(fig_path, grids, xs, ys, row_labels, col_labels,
                    col_data=None, data=None, contour_grids=None,
                    contour_level=0.9, cbar_label="$Q$ percentile"):
    """R x C montage of percentile maps (rows = methods, columns = statistics or
    datasets).  Overlay ``col_data[c]`` on column c, or the same ``data`` on all.
    NaN cells render white (masked-out regions); ``contour_grids`` lets the cyan
    boundary be drawn from a different grid than the image.
    """
    use_style()
    cmap = plt.get_cmap(PERCENTILE_CMAP).copy()
    cmap.set_bad("white")                       # NaN (masked-out) -> white
    extent = [xs[0], xs[-1], ys[0], ys[-1]]
    nr, nc = len(row_labels), len(col_labels)
    fig, axes = plt.subplots(nr, nc, figsize=(3.4 * nc, 3.2 * nr),
                             constrained_layout=True, squeeze=False)
    im = None
    for r in range(nr):
        for c in range(nc):
            ax = axes[r][c]
            im = ax.imshow(grids[r][c], origin="lower", extent=extent, vmin=0.0,
                           vmax=1.0, cmap=cmap, aspect="auto",
                           interpolation="bilinear")
            cg = contour_grids[r][c] if contour_grids is not None else grids[r][c]
            ax.contour(cg, levels=[contour_level], extent=extent,
                       colors=ACCENT, linewidths=1.3)
            overlay = col_data[c] if col_data is not None else data
            if overlay is not None:
                ax.scatter(overlay[:, 0], overlay[:, 1], s=3, c=NOMINAL_PT,
                           alpha=0.55, linewidths=0)
            if r == 0:
                ax.set_title(col_labels[c])
            if c == 0:
                ax.set_ylabel(row_labels[r])
    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.015)
    cbar.set_label(cbar_label)
    fig.savefig(fig_path)
    plt.close(fig)


def spectrum_occupations(fig_path, eigvals, occ_by_T, t_values, k_hard, n_train,
                         n_show=16, cmap="plasma"):
    """Nominal spectrum (log scale) with the soft occupations laid on top.

    Shows the graded eigenvalue ladder of ``C`` crossing the ``1/N`` sampling
    resolution, the hard rank cutoff ``K(alpha)``, and the soft occupation
    profiles ``f((lambda_j - mu_alpha)/T)`` for several resolutions ``T``.
    """
    use_style()
    norm = LogNorm(vmin=min(t_values), vmax=max(t_values))
    cmap_obj = plt.get_cmap(cmap)
    idx = np.arange(1, n_show + 1)
    fig, ax = plt.subplots(figsize=(6.6, 4.2), constrained_layout=True)
    ax.semilogy(idx, eigvals[:n_show], "o-", color="0.15", ms=4.5, lw=1.3,
                zorder=5, label="eigenvalues of $C$")
    ax.axhline(1.0 / n_train, color="0.5", ls=":", lw=1.3)
    ax.text(n_show - 0.4, 1.25 / n_train, "sampling resolution $1/N$",
            ha="right", va="bottom", color="0.35")
    ax.axvline(k_hard + 0.5, color="k", ls="--", lw=1.0)
    ax.text(k_hard + 0.7, eigvals[0] * 0.5, rf"hard cutoff $K(\alpha)={k_hard}$",
            va="top")
    ax.set_xlabel("spectral index $j$")
    ax.set_ylabel("eigenvalue $\\lambda_j$")
    ax.set_xticks(idx)
    ax2 = ax.twinx()
    for t in t_values:
        ax2.plot(idx, occ_by_T[t][:n_show], color=cmap_obj(norm(t)), lw=1.6)
    ax2.set_ylim(-0.04, 1.04)
    ax2.set_ylabel(r"soft occupation $f((\lambda_j-\mu_\alpha)/T)$")
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap_obj), ax=ax2,
                        pad=0.12)
    cbar.set_label("QSAD resolution $T$")
    ax.legend(loc="lower left")
    fig.savefig(fig_path)
    plt.close(fig)


def mode_profile(fig_path, hard_curves, soft_curves, t_values, anom_hard,
                 anom_soft, counts, cmap="plasma"):
    """Mean anomaly score per nominal mode: hard top-K versus soft QSAD.

    ``hard_curves``/``soft_curves`` map K / T to per-mode score arrays; the
    matching ``anom_*`` dicts give each detector's mean score on the anomaly
    class, drawn in a separate column on the right.  ``counts`` are the
    expected training samples per mode, shown under the mode index.
    """
    use_style()
    norm = LogNorm(vmin=min(t_values), vmax=max(t_values))
    cmap_obj = plt.get_cmap(cmap)
    m = len(next(iter(hard_curves.values())))
    x = np.arange(1, m + 1)
    x_anom = m + 1.3
    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    greys = ["0.75", "0.5", "0.0"]
    for (K, curve), c in zip(hard_curves.items(), greys):
        ax.plot(x, curve, color=c, ls="--", marker="s", ms=4, lw=1.3,
                label=rf"$Q_{{hard}}$, $K={K}$")
        ax.plot([x_anom], [anom_hard[K]], color=c, marker="s", ms=5)
    for t in t_values:
        ax.plot(x, soft_curves[t], color=cmap_obj(norm(t)), marker="o", ms=4,
                lw=1.6)
        ax.plot([x_anom], [anom_soft[t]], color=cmap_obj(norm(t)), marker="o",
                ms=5)
    ax.axvline(m + 0.65, color="0.8", lw=0.8)
    ax.set_xticks(list(x) + [x_anom])
    ax.set_xticklabels([f"{j}\n{c}" for j, c in zip(x, counts)] + ["anom.\n"])
    ax.set_xlabel("nominal mode $j$ (expected training samples)")
    ax.set_ylabel("mean anomaly score")
    ax.set_ylim(-0.04, 1.09)
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap_obj), ax=ax,
                        pad=0.015)
    cbar.set_label(r"$Q_{QSAD}$ resolution $T$")
    ax.legend(loc="upper left")
    fig.savefig(fig_path)
    plt.close(fig)


def temperature_sweep(fig_path, h_values, q_by_T, t_values, q_hard, q_raw,
                      h_c, train_band, cmap="plasma"):
    """QSAD score vs transverse field at several resolutions T, coloured by T
    (log scale).  As T -> 0 the curves converge to the hard-projector Q_hard.
    """
    use_style()
    norm = LogNorm(vmin=min(t_values), vmax=max(t_values))
    cmap_obj = plt.get_cmap(cmap)
    fig, ax = plt.subplots(figsize=(6.8, 4.3), constrained_layout=True)
    ax.axvspan(*train_band, color="0.85")
    ax.axvline(h_c, color="0.4", ls="--", lw=1.0)
    for t in t_values:
        ax.plot(h_values, q_by_T[t], color=cmap_obj(norm(t)), lw=1.6)
    ax.plot(h_values, q_hard, color="k", ls="none", marker="o", ms=2.5,
            markevery=1, zorder=5)
    ax.plot(h_values, q_raw, color="0.55", ls=":", lw=1.5)
    ax.set_xlabel("transverse field $h / J$")
    ax.set_ylabel("anomaly score $Q$")
    ax.set_xlim(h_values[0], h_values[-1])
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap_obj), ax=ax, pad=0.015)
    cbar.set_label("QSAD resolution $T$")
    handles = [
        Line2D([0], [0], color=cmap_obj(0.6), lw=1.6, label=r"$Q_{QSAD}$ (varying $T$)"),
        Line2D([0], [0], color="k", ls="none", marker="o", ms=5, label=r"$Q_{hard}$ ($T\to0$ limit)"),
        Line2D([0], [0], color="0.55", ls=":", lw=1.5, label=r"$Q_{raw}$ (naive)"),
        Line2D([0], [0], color="0.4", ls="--", lw=1.0, label=r"critical point $h_c$"),
    ]
    ax.legend(handles=handles, loc="upper left")
    fig.savefig(fig_path)
    plt.close(fig)


def auc_curve(fig_path, t_values, auc_by_label, auc_raw):
    """Detection AUC vs resolution T for several measurement-shot budgets.

    ``auc_by_label`` maps a label to ``(mean, std_or_None)`` arrays aligned with
    ``t_values``.  ``auc_raw`` is the T-independent naive baseline.
    """
    use_style()
    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    for label, (mean, std) in auc_by_label.items():
        line, = ax.plot(t_values, mean, marker="o", ms=3.5, label=label)
        if std is not None:
            ax.fill_between(t_values, mean - std, mean + std,
                            color=line.get_color(), alpha=0.2)
    ax.axhline(auc_raw, color="0.5", ls=":", lw=1.4, label=r"$Q_{raw}$ (naive)")
    ax.set_xscale("log")
    ax.set_xlabel("QSAD resolution $T$")
    ax.set_ylabel("detection AUC")
    ax.set_ylim(0.45, 1.02)
    ax.legend(loc="lower left", ncol=2)
    fig.savefig(fig_path)
    plt.close(fig)


def sector_temperature(fig_path, group_labels, q_raw, q_qsad_by_T, t_values,
                       q_hard_by_K=None, cmap="plasma"):
    """Grouped bars per sector: Q_raw, hard top-K projectors, and Q_QSAD at
    several resolutions T.

    ``q_raw`` is a per-group array; ``q_qsad_by_T`` maps T -> per-group array;
    ``q_hard_by_K`` (optional) maps K -> per-group array, drawn as hatched
    grey bars between the raw and soft groups.
    """
    use_style()
    norm = LogNorm(vmin=min(t_values), vmax=max(t_values))
    cmap_obj = plt.get_cmap(cmap)
    ts = sorted(t_values, reverse=True)            # high T (light) first
    hards = list(q_hard_by_K or {})
    n_bars = 1 + len(hards) + len(ts)
    width = 0.84 / n_bars
    x = np.arange(len(group_labels))
    offset = -(n_bars - 1) / 2

    fig, ax = plt.subplots(figsize=(7.8, 4.4), constrained_layout=True)
    ax.bar(x + offset * width, q_raw, width, color="0.45", label=r"$Q_{raw}$")
    hard_greys = ["0.85", "0.62"]
    for i, K in enumerate(hards):
        ax.bar(x + (offset + 1 + i) * width, q_hard_by_K[K], width,
               color=hard_greys[i % len(hard_greys)], hatch="//",
               edgecolor="0.25", linewidth=0.4,
               label=rf"$Q_{{hard}}$ ($K={K}$)")
    for i, t in enumerate(ts):
        ax.bar(x + (offset + 1 + len(hards) + i) * width, q_qsad_by_T[t],
               width, color=cmap_obj(norm(t)))
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)
    ax.set_ylabel("mean anomaly score")
    ax.set_ylim(0, 1.05)
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap_obj), ax=ax, pad=0.015)
    cbar.set_label(r"QSAD resolution $T$ (for $Q_{QSAD}$)")
    ax.legend(loc="upper left")
    fig.savefig(fig_path)
    plt.close(fig)
