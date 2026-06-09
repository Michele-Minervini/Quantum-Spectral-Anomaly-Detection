"""Plotting: shared style and figure builders."""

from .style import use_style
from .panels import (
    encoding_comparison,
    comparison_grid,
    score_distributions,
    temperature_sweep,
    auc_curve,
    sector_distributions,
    sector_temperature,
    roc_panel,
)

__all__ = [
    "use_style",
    "encoding_comparison",
    "comparison_grid",
    "score_distributions",
    "temperature_sweep",
    "auc_curve",
    "sector_distributions",
    "sector_temperature",
    "roc_panel",
]
