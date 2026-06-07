from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.models.hbb._config import CV_PREDICTORS


def transform_to_beta(y: np.ndarray, n: int) -> np.ndarray:
    return (y * (n - 1) + 0.5) / n


def inverse_transform_beta(y_beta: np.ndarray, n: int) -> np.ndarray:
    return (y_beta * n - 0.5) / (n - 1)


def build_design_matrix(
    df: pd.DataFrame,
    predictors: Optional[list[str]] = None,
    add_intercept: bool = True,
) -> tuple[np.ndarray, list[str]]:
    predictors = predictors or list(CV_PREDICTORS)
    cols = [p for p in predictors if p in df.columns]
    X = df[cols].values.astype(float)
    names = list(cols)
    if add_intercept:
        X = np.column_stack([np.ones(len(df)), X])
        names = ["Intercept"] + names
    return X, names


def compute_correlation_matrix(
    df: pd.DataFrame, columns: Optional[list[str]] = None
) -> pd.DataFrame:
    columns = columns or [
        "lat", "depth", "human_pop", "cyclone", "sst_mean", "sst_min", "sst_max",
        "sst_stdev", "ssta_min", "ssta_max", "ssta_mean", "ssta_stdev", "ssta_freqmax",
        "ssta_freqstdev", "ssta_dhwmean", "ssta_dhwmax", "tsa_min", "tsa_max", "tsa_mean",
        "tsa_freqstdev", "tsa_dhwmean", "tsa_dhwmax", "tsa_dhwstdev", "turbidity_mean",
        "turbidity_max", "historical_sst_max", "historical_sst_mean", "historical_sst_sd",
    ]
    use = [c for c in columns if c in df.columns]
    return df[use].corr()
