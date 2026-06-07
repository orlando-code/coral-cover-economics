from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.special import expit as inv_logit
from scipy.stats import beta as beta_dist
from sklearn.metrics import r2_score

from src.models.hbb._config import CV_PREDICTORS
from src.models.hbb.data import standardize_train_test
from src.models.hbb.design import (
    build_design_matrix,
    inverse_transform_beta,
    transform_to_beta,
)
from src.models.hbb.indices import (
    build_region_diversity,
    make_dense_site_region,
)

if TYPE_CHECKING:
    from src.models.hbb.model import HierarchicalBetaModel


def _coral_cover_proportion(cover: np.ndarray) -> np.ndarray:
    y = np.asarray(cover, dtype=float)
    if np.nanmax(y) > 1.5:
        y = y / 100.0
    return np.clip(y, 0.0, 1.0)


def _beta_to_proportion(y_beta: np.ndarray, n_train: int) -> np.ndarray:
    return _coral_cover_proportion(inverse_transform_beta(y_beta, n_train))


def prepare_cv_fold_arrays(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_eps: float = 1e-6,
) -> dict[str, Any]:
    """Build arrays for one CV fold (train fit + test prediction)."""
    std = standardize_train_test(train_df, test_df)
    tr_raw, te_raw = std["train"], std["test"]
    dense = make_dense_site_region(tr_raw)
    tr = dense["data"]
    n_train = len(tr)

    predictors = [p for p in CV_PREDICTORS if p in tr.columns]
    X_train, col_names = build_design_matrix(tr, predictors, add_intercept=False)
    X_test, _ = build_design_matrix(te_raw, predictors, add_intercept=False)

    y_train = transform_to_beta(tr["average_coral_cover"].to_numpy(), n_train)
    y_train = np.clip(y_train, y_eps, 1.0 - y_eps)

    site_idx = tr["site_dense"].to_numpy(dtype=int)
    region_idx = tr["region_dense"].to_numpy(dtype=int)
    site_to_region = dense["region_for_each_site"]
    diversity = build_region_diversity(tr, dense["n_regions"])

    reef_col = "reef_id" if "reef_id" in tr.columns else "reef"
    reef_to_site_map = dict(zip(tr[reef_col], tr["site_dense"]))

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "site_idx": site_idx,
        "region_idx": region_idx,
        "site_to_region": site_to_region,
        "diversity": diversity,
        "col_names": col_names,
        "dense_info": dense,
        "test_df": te_raw,
        "n_train": n_train,
        "reef_to_site_map": reef_to_site_map,
    }


def predict_from_posterior_cv(
    model: HierarchicalBetaModel,
    X_test: np.ndarray,
    test_df: pd.DataFrame,
    dense_info: dict[str, Any],
    n_train: int,
    y_eps: float = 1e-6,
) -> dict[str, Any]:
    """Posterior predictions with metrics on coral-cover proportion scale (0–1)."""
    if model.trace is None:
        raise ValueError("Model must be fit before prediction.")

    post = model.trace.posterior
    beta_draws = post["beta"].stack(sample=("chain", "draw")).values.T
    theta_draws = post["theta"].stack(sample=("chain", "draw")).values
    mu_global_draws = post["mu_global"].stack(sample=("chain", "draw")).values
    beta_div_draws = post["beta_diversity"].stack(sample=("chain", "draw")).values
    site_draws = post["site_effect"].stack(sample=("chain", "draw")).values.T
    eco_draws = post["ecoregion"].stack(sample=("chain", "draw")).values.T

    n_draws = beta_draws.shape[0]
    n_test = X_test.shape[0]
    fixed_part = beta_draws @ X_test.T

    site_dense_map = dense_info["site_dense_map"]
    region_dense_map = dense_info["region_dense_map"]
    div_col = (
        "diversity.standardized"
        if "diversity.standardized" in test_df.columns
        else "diversity"
    )

    hier_part = np.full((n_draws, n_test), np.nan, dtype=float)
    for i in range(n_test):
        s_i = site_dense_map.get(str(test_df["site"].iloc[i]))
        r_i = region_dense_map.get(str(test_df["region"].iloc[i]))
        d_i = test_df[div_col].iloc[i]

        if s_i is not None:
            hier_part[:, i] = site_draws[:, int(s_i)]
        elif r_i is not None:
            hier_part[:, i] = eco_draws[:, int(r_i)]
        elif np.isfinite(d_i):
            hier_part[:, i] = mu_global_draws + beta_div_draws * d_i
        else:
            hier_part[:, i] = mu_global_draws

    pi_draw = inv_logit(fixed_part + hier_part)
    pred_mean = pi_draw.mean(axis=0)
    pred_lo = np.quantile(pi_draw, 0.025, axis=0)
    pred_hi = np.quantile(pi_draw, 0.975, axis=0)

    y_obs_prop = _coral_cover_proportion(test_df["average_coral_cover"].to_numpy())
    pred_mean_prop = _beta_to_proportion(pred_mean, n_train)
    pred_lo_prop = _beta_to_proportion(pred_lo, n_train)
    pred_hi_prop = _beta_to_proportion(pred_hi, n_train)

    rmse = float(np.sqrt(np.mean((y_obs_prop - pred_mean_prop) ** 2)))
    mae = float(np.mean(np.abs(y_obs_prop - pred_mean_prop)))
    r2 = float(r2_score(y_obs_prop, pred_mean_prop))
    coverage95 = float(
        np.mean((y_obs_prop >= pred_lo_prop) & (y_obs_prop <= pred_hi_prop))
    )

    y_obs_beta = transform_to_beta(y_obs_prop, n_train)
    y_obs_beta = np.clip(y_obs_beta, y_eps, 1.0 - y_eps)
    eps = 1e-9
    y_clamp = np.clip(y_obs_beta, eps, 1.0 - eps)
    log_scores = []
    for i in range(n_test):
        p_i = np.clip(pi_draw[:, i], eps, 1.0 - eps)
        sh1 = theta_draws * p_i
        sh2 = theta_draws * (1.0 - p_i)
        dens = beta_dist.pdf(y_clamp[i], sh1, sh2)
        log_scores.append(np.log(np.mean(np.maximum(dens, eps))))
    mean_log_score = float(np.mean(log_scores))

    return {
        "metrics": {
            "r2": r2,
            "rmse": rmse,
            "mae": mae,
            "coverage95": coverage95,
            "mean_log_score": mean_log_score,
        },
        "predictions": pd.DataFrame(
            {
                "row_id": test_df["row_id"].values,
                "y_obs": y_obs_prop,
                "y_pred": pred_mean_prop,
                "y_pred_lo95": pred_lo_prop,
                "y_pred_hi95": pred_hi_prop,
                "y_obs_beta": y_obs_beta,
                "y_pred_beta": pred_mean,
            }
        ),
    }
