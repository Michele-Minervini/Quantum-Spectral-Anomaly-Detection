"""Figure builders for the QSAD experiments."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

from .style import ACCENT, NOMINAL_PT, PERCENTILE_CMAP, use_style


def encoding_comparison(fig_path, panels, xs, ys, data=None, contour_level=0.9,
                        ncols=3):
    """Grid of QSAD-Q percentile maps, one per encoding, with its AUC in the title."""
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
    cbar.set_label("QSAD $Q$ percentile")
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


def score_distributions(fig_path, h_values, curves, h_c, train_band, styles=None):
    """Anomaly scores vs transverse field ``h`` (Experiment B diagnostic).

    ``curves`` maps a label to a score array aligned with ``h_values``.  The
    critical point and the nominal training band are shaded for context.
    """
    use_style()
    styles = styles or {}
    fig, ax = plt.subplots(figsize=(6.2, 4.0), constrained_layout=True)
    ax.axvspan(*train_band, color="0.85", label="nominal train band")
    ax.axvline(h_c, color="0.4", ls="--", lw=1.0, label="critical point $h_c$")
    for label, y in curves.items():
        ax.plot(h_values, y, ls=styles.get(label, "-"), label=label)
    ax.set_xlabel("transverse field $h / J$")
    ax.set_ylabel("anomaly score")
    ax.set_xlim(h_values[0], h_values[-1])
    ax.legend(loc="center right")
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
    ax.plot(h_values, q_hard, color="k", ls="--", lw=2.0)
    ax.plot(h_values, q_raw, color="0.55", ls=":", lw=1.5)
    ax.set_xlabel("transverse field $h / J$")
    ax.set_ylabel("anomaly score $Q$")
    ax.set_xlim(h_values[0], h_values[-1])
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap_obj), ax=ax, pad=0.015)
    cbar.set_label("QSAD resolution $T$")
    handles = [
        Line2D([0], [0], color=cmap_obj(0.6), lw=1.6, label=r"$Q_{QSAD}$ (varying $T$)"),
        Line2D([0], [0], color="k", ls="--", lw=2.0, label=r"$Q_{hard}$ ($T\to0$ limit)"),
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
    # ax.axhline(0.5, color="0.75", ls="--", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("QSAD resolution $T$")
    ax.set_ylabel("detection AUC")
    ax.set_ylim(0.45, 1.02)
    ax.legend(loc="lower left", ncol=2)
    fig.savefig(fig_path)
    plt.close(fig)


def sector_temperature(fig_path, group_labels, q_raw, q_qsad_by_T, t_values,
                       cmap="plasma"):
    """Grouped bars per sector: Q_raw plus Q_QSAD at several resolutions T.

    ``q_raw`` is a per-group array; ``q_qsad_by_T`` maps T -> per-group array.
    """
    use_style()
    norm = LogNorm(vmin=min(t_values), vmax=max(t_values))
    cmap_obj = plt.get_cmap(cmap)
    ts = sorted(t_values, reverse=True)            # high T (light) first
    n_bars = len(ts) + 1
    width = 0.82 / n_bars
    x = np.arange(len(group_labels))
    fig, ax = plt.subplots(figsize=(7.8, 4.4), constrained_layout=True)
    ax.bar(x - (n_bars - 1) / 2 * width, q_raw, width, color="0.45",
           label=r"$Q_{raw}$")
    for i, t in enumerate(ts):
        ax.bar(x + (i + 1 - (n_bars - 1) / 2) * width, q_qsad_by_T[t], width,
               color=cmap_obj(norm(t)))
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)
    ax.set_ylabel("mean anomaly score")
    ax.set_ylim(0, 1.05)
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap_obj), ax=ax, pad=0.015)
    cbar.set_label(r"QSAD resolution $T$ (for $Q_{QSAD}$)")
    ax.legend(loc="upper left")
    fig.savefig(fig_path)
    plt.close(fig)


def sector_distributions(fig_path, data, group_labels, group_colors=None):
    """Per-sector score distributions, one violin panel per detector.

    ``data`` maps a detector name to a list of score arrays (one per group).
    Reveals that a density-weighted detector flags the rare-but-valid nominal
    sector, whereas a calibrated QSAD detector does not.
    """
    use_style()
    names = list(data)
    pos = np.arange(len(group_labels))
    fig, axes = plt.subplots(1, len(names), figsize=(4.4 * len(names), 4.2),
                             sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, names):
        parts = ax.violinplot(data[name], positions=pos, widths=0.8,
                              showmeans=True, showextrema=False)
        for i, body in enumerate(parts["bodies"]):
            body.set_facecolor(group_colors[i] if group_colors else "#4c72b0")
            body.set_alpha(0.75)
        parts["cmeans"].set_color("0.2")
        ax.set_xticks(pos)
        ax.set_xticklabels(group_labels)
        ax.set_ylim(-0.05, 1.08)
        ax.set_title(name)
    axes[0].set_ylabel("anomaly score")
    fig.savefig(fig_path)
    plt.close(fig)


def roc_panel(fig_path, curves):
    """Overlay ROC curves.  ``curves`` maps label -> ``(fpr, tpr, auc)``."""
    use_style()
    fig, ax = plt.subplots(figsize=(4.6, 4.4), constrained_layout=True)
    ax.plot([0, 1], [0, 1], color="0.7", ls=":", lw=1.0)
    for label, (fpr, tpr, auc) in curves.items():
        ax.plot(fpr, tpr, label=f"{label} (AUC {auc:.3f})")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(loc="lower right")
    fig.savefig(fig_path)
    plt.close(fig)
