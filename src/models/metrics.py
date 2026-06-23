"""Regression metrics for baseline coral-cover models."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def projection_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    y_persist: np.ndarray | None = None,
    y_lag: np.ndarray | None = None,
    has_prior: np.ndarray | None = None,
) -> dict[str, float]:
    """Level, change, and persistence-relative metrics for projection models."""
    out = {f"level_{k}": v for k, v in regression_metrics(y_true, y_pred).items()}

    if y_persist is not None:
        y_persist = np.asarray(y_persist, dtype=float)
        out.update(
            {f"persist_{k}": v for k, v in regression_metrics(y_true, y_persist).items()}
        )
        delta_true = y_true - y_persist
        delta_pred = y_pred - y_persist
        out.update(
            {
                f"delta_{k}": v
                for k, v in regression_metrics(delta_true, delta_pred).items()
            }
        )

    if y_lag is not None and has_prior is not None:
        mask = np.asarray(has_prior, dtype=bool) & np.isfinite(y_lag)
        if int(mask.sum()) >= 2:
            y_true_m = y_true[mask]
            y_pred_m = y_pred[mask]
            out.update(
                {
                    f"prior_only_{k}": v
                    for k, v in regression_metrics(y_true_m, y_pred_m).items()
                }
            )
            y_lag_m = np.asarray(y_lag, dtype=float)[mask]
            out["r2_lag_baseline"] = float(r2_score(y_true_m, y_lag_m))
            out["n_with_prior"] = float(mask.sum())

    return out
