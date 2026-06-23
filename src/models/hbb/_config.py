from __future__ import annotations

import warnings
from typing import Literal

ModelSpec = Literal["reparam", "legacy_r", "centered"]

from src import config

try:
    import arviz as az  # noqa: F401
    import pymc as pm  # noqa: F401

    HAS_PYMC = True
except ImportError:
    HAS_PYMC = False
    warnings.warn("PyMC not installed. Bayesian modeling unavailable.")

SULLY_DATA_DIR = config.data_dir / "sully_og"
OUTPUT_DIR = config.figures_dir / "hbb"

FEATURE_VARS = [
    "lat", "depth", "human_pop", "cyclone", "sst_mean", "ssta_mean", "ssta_min",
    "ssta_freqstdev", "ssta_dhwmax", "tsa_max", "tsa_freqstdev", "turbidity_mean",
    "historical_sst_max",
]
CV_PREDICTORS = [f"{v}_stzd" for v in FEATURE_VARS]
VARS_TO_STANDARDIZE = [
    "lon", "lat", "depth", "human_pop", "cyclone", "sst_mean", "sst_max", "sst_stdev",
    "ssta_min", "ssta_max", "ssta_mean", "ssta_stdev", "ssta_freqmax", "ssta_freqstdev",
    "ssta_dhwmean", "ssta_dhwmax", "tsa_min", "tsa_max", "tsa_mean", "tsa_freqstdev",
    "tsa_dhwmean", "tsa_dhwmax", "tsa_dhwstdev", "turbidity_mean", "turbidity_max",
    "historical_sst_max", "historical_sst_mean", "historical_sst_sd",
]

LOAD_NA_COLS = [
    "average_coral_cover", "sst_mean", "ssta_stdev", "ssta_freqmax", "ssta_freqmean",
    "turbidity_mean", "ecoregion", "cyclone", "depth", "historical_sst_max",
    "sst_mean_rcp85_2100",
]
CLEAN_REQUIRED = [
    "sst_mean", "sst_stdev", "sst_freqmax", "sst_freqmean", "turbidity_mean",
    "cyclone", "depth", "historical_sst_max", "sst_mean_rcp85_2100",
]
