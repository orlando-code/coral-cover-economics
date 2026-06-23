"""Persistence baseline: prior survey mean, with ecoregion fallback for novel sites."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.cv_methods import pick_first_existing, year_series


def _observation_time(df: pd.DataFrame) -> np.ndarray:
    """Scalar time axis for ordering surveys (days since epoch or calendar year)."""
    days_col = pick_first_existing(df, ["days_since_19811231"])
    if days_col is not None:
        return df[days_col].astype(float).to_numpy()
    return year_series(df).astype(float).to_numpy()


def predict_survey_mean_baseline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_train: np.ndarray,
    *,
    site_col: str = "site",
    region_col: str = "region",
) -> np.ndarray:
    """Predict coral cover from prior surveys, falling back to ecoregion then global mean.

    For each test row:
    1. Mean observed cover at the same site among **training** rows strictly before
       the test observation time.
    2. If the site is novel (no prior training surveys), the training-set mean cover
       for the test row's ecoregion.
    3. If the ecoregion is also unseen in training, the global training mean.
    """
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    y_train = np.asarray(y_train, dtype=float)

    if len(y_train) != len(train_df):
        raise ValueError(
            f"y_train length ({len(y_train)}) must match train_df ({len(train_df)})."
        )

    train_time = _observation_time(train_df)
    test_time = _observation_time(test_df)
    train_site = train_df[site_col].astype(str).to_numpy()
    train_region = train_df[region_col].astype(str).to_numpy()
    test_site = test_df[site_col].astype(str).to_numpy()
    test_region = test_df[region_col].astype(str).to_numpy()

    global_mean = float(np.nanmean(y_train)) if len(y_train) else 0.0
    region_means = (
        pd.DataFrame({"region": train_region, "y": y_train})
        .groupby("region", observed=True)["y"]
        .mean()
    )

    preds = np.full(len(test_df), global_mean, dtype=float)
    for i in range(len(test_df)):
        prior = (train_site == test_site[i]) & (train_time < test_time[i])
        if prior.any():
            preds[i] = float(np.nanmean(y_train[prior]))
            continue
        region = test_region[i]
        if region in region_means.index:
            preds[i] = float(region_means[region])

    return np.clip(preds, 0.0, 1.0)
