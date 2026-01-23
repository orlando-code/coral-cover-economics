"""
Coral Cover Economics

A Python package for hierarchical Bayesian beta regression modeling of coral cover
and economic analysis based on environmental variables.
"""

__version__ = "0.1.0"

from . import plot_utils
from .plot_config import (
    EXTENT_PRESETS,
    PAPER_CONFIG,
    PAPER_SPATIAL_CONFIG,
    PlotConfig,
    SpatialPlotConfig,
)

__all__ = [
    "PlotConfig",
    "SpatialPlotConfig",
    "PAPER_CONFIG",
    "PAPER_SPATIAL_CONFIG",
    "EXTENT_PRESETS",
    "plot_utils",
]
