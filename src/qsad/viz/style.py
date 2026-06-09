"""Shared matplotlib styling for QSAD figures."""

import matplotlib as mpl

# Colormap conventions: dark = nominal (low percentile), light = anomalous.
PERCENTILE_CMAP = "magma"
ACCENT = "#1fd2c8"        # cyan decision-boundary contour
NOMINAL_PT = "#7fdbff"    # scatter colour for nominal data

_RC = {
    "figure.dpi": 120,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "axes.linewidth": 0.6,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "lines.linewidth": 1.8,
    "mathtext.default": "regular",
}


def use_style():
    """Apply the QSAD rcParams to the current matplotlib session."""
    mpl.rcParams.update(_RC)
