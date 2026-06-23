"""Load and prepare coral-cover data (shared cache; aligned with R beta-GLMM).

Baseline CV uses :mod:`src.models.baseline_features` for per-fold beta-aligned
design matrices; :func:`build_design_matrix` remains for legacy utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Same covariates as src/native_r/beta_model_reparam_utils.R (FEATURE_VARS / design matrix).
FEATURE_VARS = [
    "lat",
    "Depth",
    "Human_pop",
    "Cyclone",
    "SST_mean",
    "SSTA_Mean",
    "SSTA_min",
    "SSTA_freqstdev",
    "SSTA_dhwmax",
    "TSA_max",
    "TSA_freqstdev",
    "Turbidity_mean",
    "Historical_SST_max",
]

COEF_LABELS = [
    "Latitude",
    "Depth",
    "Human_pop",
    "Cyclone",
    "SST_mean",
    "SSTA_Mean",
    "SSTA_min",
    "SSTA_freqstdev",
    "SSTA_dhwmax",
    "TSA_max",
    "TSA_freqstdev",
    "Turbidity_mean",
    "Historical_SST_max",
]


@dataclass(frozen=True)
class StandardizationStats:
    mean: dict[str, float]
    sd: dict[str, float]


def _ensure_lat(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "lat" not in out.columns:
        lat_col = (
            "Latitude.Degrees"
            if "Latitude.Degrees" in out.columns
            else "latitude.degrees"
        )
        out["lat"] = np.abs(out[lat_col].astype(float))
    return out


def _to_proportion(cover: pd.Series) -> pd.Series:
    y = cover.astype(float)
    if y.max(skipna=True) > 1.5:
        y = y / 100.0
    return y


def filter_model_ready_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Match dplyr::filter() in beta_model_reparam_utils.R::filter_model_ready_rows."""
    from src.dataloading.build_sully_model_ready_data import (
        filter_model_ready_rows as _filter,
    )

    return _filter(df)


def load_model_ready_data(
    data_path: Optional[Path] = None,
    *,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """
    Load the shared cached model-ready dataset.

    Built from ``data.csv`` + ``data_for_maps.csv`` (diversity/site lookup) via
    :mod:`src.dataloading.build_sully_model_ready_data`.
    """
    from src.dataloading.build_sully_model_ready_data import (
        load_model_ready_data as _load,
    )

    return _load(cache_path=data_path, force_rebuild=force_rebuild)


def standardize_features(
    df: pd.DataFrame,
    stats: Optional[StandardizationStats] = None,
    *,
    vars_: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, StandardizationStats]:
    """Z-score FEATURE_VARS (train stats when ``stats`` is provided)."""
    vars_ = vars_ or FEATURE_VARS
    out = df.copy()
    means: dict[str, float] = {}
    sds: dict[str, float] = {}

    for v in vars_:
        if stats is not None:
            m, s = stats.mean[v], stats.sd[v]
        else:
            x = out[v].astype(float)
            m = float(x.mean())
            s = float(x.std(ddof=0))
            if not np.isfinite(s) or s == 0:
                s = 1.0
        means[v] = m
        sds[v] = s
        out[v] = (out[v].astype(float) - m) / s

    return out, StandardizationStats(mean=means, sd=sds)


def build_design_matrix(
    df: pd.DataFrame,
    *,
    add_intercept: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """Design matrix matching R ``build_design_matrix()`` (no intercept by default)."""
    X = df[FEATURE_VARS].astype(float).to_numpy()
    if not np.isfinite(X).all():
        raise ValueError("Non-finite values in design matrix.")
    names = list(FEATURE_VARS)
    if add_intercept:
        X = np.column_stack([np.ones(len(df)), X])
        names = ["intercept", *names]
    return X, names


def coral_cover_target(df: pd.DataFrame) -> np.ndarray:
    """Observed coral cover on proportion scale (0–1)."""
    return _to_proportion(df["Average_coral_cover"]).to_numpy(dtype=float)
