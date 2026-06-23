"""Fold-safe dynamic features for longitudinal coral-cover projection.

Adds lagged cover and time-since-last-survey using only information available
at each observation time (strictly prior training surveys for test rows).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.models.baseline_features import prepare_baseline_fold_frames
from src.models.baseline_persistence import _observation_time, predict_survey_mean_baseline

_LOGIT_EPS = 1e-4
_DAYS_PER_YEAR = 365.25
DYNAMIC_INPUT_COLS = (
    "lag_cover_stzd",
    "logit_lag_cover_stzd",
    "time_since_last_survey_years_stzd",
    "has_prior_survey",
)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), _LOGIT_EPS, 1.0 - _LOGIT_EPS)
    return np.log(p / (1.0 - p))


def _prior_cover_lookup(
    history_df: pd.DataFrame,
    history_y: np.ndarray,
    query_df: pd.DataFrame,
    *,
    site_col: str = "site",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (lag_cover, time_since_years, has_prior) for each query row."""
    history_df = history_df.reset_index(drop=True)
    query_df = query_df.reset_index(drop=True)
    history_y = np.asarray(history_y, dtype=float)
    hist_time = _observation_time(history_df)
    query_time = _observation_time(query_df)
    hist_site = history_df[site_col].astype(str).to_numpy()
    query_site = query_df[site_col].astype(str).to_numpy()

    n_q = len(query_df)
    lag = np.full(n_q, np.nan, dtype=float)
    time_since = np.full(n_q, np.nan, dtype=float)
    has_prior = np.zeros(n_q, dtype=bool)

    for i in range(n_q):
        prior_mask = (hist_site == query_site[i]) & (hist_time < query_time[i])
        if not prior_mask.any():
            continue
        prior_times = hist_time[prior_mask]
        prior_y = history_y[prior_mask]
        j = int(np.argmax(prior_times))
        lag[i] = float(prior_y[j])
        time_since[i] = float((query_time[i] - prior_times[j]) / _DAYS_PER_YEAR)
        has_prior[i] = True

    return lag, time_since, has_prior


def _standardize_column(
    train_vals: np.ndarray,
    test_vals: np.ndarray,
    *,
    train_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Train-only z-score with NaN preserved."""
    ref = train_vals if train_mask is None else train_vals[train_mask]
    ref = ref[np.isfinite(ref)]
    if len(ref) < 2:
        mean, std = 0.0, 1.0
    else:
        mean = float(np.mean(ref))
        std = float(np.std(ref))
        if std < 1e-8:
            std = 1.0

    def _z(x: np.ndarray) -> np.ndarray:
        out = (np.asarray(x, dtype=float) - mean) / std
        out[~np.isfinite(x)] = np.nan
        return out

    meta = {"mean": mean, "std": std}
    return _z(train_vals), _z(test_vals), meta


def prepare_dynamic_fold_frames(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_train: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, Any]]:
    """
    Per-fold frames for dynamic projection models.

    Returns train/test feature matrices, feature names, and metadata including
    persistence baselines and raw lag columns for metrics.
    """
    train_prep, test_prep, env_cols = prepare_baseline_fold_frames(train_df, test_df)
    y_train = np.asarray(y_train, dtype=float)

    train_lag, train_dt, train_has = _prior_cover_lookup(
        train_df, y_train, train_df
    )
    test_lag, test_dt, test_has = _prior_cover_lookup(
        train_df, y_train, test_df
    )

    y_persist_train = predict_survey_mean_baseline(train_df, train_df, y_train)
    y_persist_test = predict_survey_mean_baseline(train_df, test_df, y_train)

    train_logit_lag = _logit(train_lag)
    test_logit_lag = _logit(test_lag)

    lag_stzd_train, lag_stzd_test, lag_meta = _standardize_column(
        train_lag, test_lag, train_mask=train_has
    )
    logit_lag_stzd_train, logit_lag_stzd_test, logit_lag_meta = _standardize_column(
        train_logit_lag, test_logit_lag, train_mask=train_has
    )
    dt_stzd_train, dt_stzd_test, dt_meta = _standardize_column(
        train_dt, test_dt, train_mask=train_has
    )

    train_out = train_prep.copy()
    test_out = test_prep.copy()
    dyn_cols = [
        "lag_cover_stzd",
        "logit_lag_cover_stzd",
        "time_since_last_survey_years_stzd",
        "has_prior_survey",
    ]
    train_out["lag_cover_stzd"] = lag_stzd_train
    test_out["lag_cover_stzd"] = lag_stzd_test
    train_out["logit_lag_cover_stzd"] = logit_lag_stzd_train
    test_out["logit_lag_cover_stzd"] = logit_lag_stzd_test
    train_out["time_since_last_survey_years_stzd"] = dt_stzd_train
    test_out["time_since_last_survey_years_stzd"] = dt_stzd_test
    train_out["has_prior_survey"] = train_has.astype(float)
    test_out["has_prior_survey"] = test_has.astype(float)

    feature_cols = list(train_prep.columns) + dyn_cols
    meta: dict[str, Any] = {
        "env_cols": env_cols,
        "dynamic_cols": dyn_cols,
        "lag_standardization": lag_meta,
        "logit_lag_standardization": logit_lag_meta,
        "time_since_standardization": dt_meta,
        "train_has_prior_frac": float(train_has.mean()) if len(train_has) else 0.0,
        "test_has_prior_frac": float(test_has.mean()) if len(test_has) else 0.0,
        "y_persist_train": y_persist_train,
        "y_persist_test": y_persist_test,
        "y_lag_train": train_lag,
        "y_lag_test": test_lag,
        "has_prior_train": train_has,
        "has_prior_test": test_has,
    }
    return train_out[feature_cols], test_out[feature_cols], feature_cols, meta


def dynamic_feature_spec() -> dict[str, Any]:
    return {
        "base_features": "beta_aligned_env_plus_hierarchical_site_logit",
        "dynamic_features": [
            "lag_cover_stzd",
            "logit_lag_cover_stzd",
            "time_since_last_survey_years_stzd",
            "has_prior_survey",
        ],
        "persistence_baseline": "prior_survey_mean_with_region_fallback",
        "notes": (
            "Lagged cover uses strictly prior surveys within the training fold "
            "for test rows. Rows without a prior survey rely on persistence + "
            "environment only."
        ),
    }
