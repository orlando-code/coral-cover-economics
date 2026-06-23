from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.special import expit as inv_logit
from scipy.stats import beta as beta_dist
from sklearn.metrics import r2_score

from src.models.hbb.data import standardize_train_test
from src.models.hbb.design import (
    build_design_matrix,
    inverse_transform_beta,
    transform_to_beta,
)

if TYPE_CHECKING:
    from src.models.hbb.model import HierarchicalBetaModel


def coral_cover_proportion(cover: np.ndarray) -> np.ndarray:
    y = np.asarray(cover, dtype=float)
    if np.nanmax(y) > 1.5:
        y = y / 100.0
    return np.clip(y, 0.0, 1.0)


def _beta_to_proportion(y_beta: np.ndarray, n_train: int) -> np.ndarray:
    return coral_cover_proportion(inverse_transform_beta(y_beta, n_train))


def _diversity_column(test_df: pd.DataFrame) -> str:
    return (
        "diversity.standardized"
        if "diversity.standardized" in test_df.columns
        else "diversity"
    )


def hierarchical_logit_offsets(
    post,
    test_df: pd.DataFrame,
    dense_info: dict[str, Any],
    *,
    use_site_hierarchy: bool,
    use_ecoregion_hierarchy: bool,
    use_diversity: bool,
) -> np.ndarray | float:
    """Per-draw hierarchical logit offsets with shape (n_draws, n_test), or 0.0."""
    use_hierarchy = use_site_hierarchy or use_ecoregion_hierarchy
    if not use_hierarchy:
        return 0.0

    n_test = len(test_df)
    mu_global_draws = post["mu_global"].stack(sample=("chain", "draw")).values
    n_draws = len(mu_global_draws)
    site_draws = (
        post["site_effect"].stack(sample=("chain", "draw")).values.T
        if use_site_hierarchy and "site_effect" in post
        else None
    )
    eco_draws = (
        post["ecoregion"].stack(sample=("chain", "draw")).values.T
        if use_ecoregion_hierarchy and "ecoregion" in post
        else None
    )
    beta_div_draws = (
        post["beta_diversity"].stack(sample=("chain", "draw")).values
        if use_ecoregion_hierarchy and use_diversity and "beta_diversity" in post
        else None
    )

    site_dense_map = dense_info["site_dense_map"]
    region_dense_map = dense_info["region_dense_map"]
    div_col = _diversity_column(test_df)

    hier_part = np.full((n_draws, n_test), np.nan, dtype=float)
    for i in range(n_test):
        s_i = site_dense_map.get(str(test_df["site"].iloc[i]))
        r_i = region_dense_map.get(str(test_df["region"].iloc[i]))
        d_i = test_df[div_col].iloc[i] if div_col in test_df.columns else np.nan

        if use_site_hierarchy and s_i is not None and site_draws is not None:
            hier_part[:, i] = site_draws[:, int(s_i)]
        elif use_ecoregion_hierarchy and r_i is not None and eco_draws is not None:
            hier_part[:, i] = eco_draws[:, int(r_i)]
        elif (
            use_ecoregion_hierarchy
            and use_diversity
            and beta_div_draws is not None
            and np.isfinite(d_i)
        ):
            hier_part[:, i] = mu_global_draws + beta_div_draws * d_i
        else:
            hier_part[:, i] = mu_global_draws
    return hier_part


def ecoregion_only_logit_offsets(
    post,
    test_df: pd.DataFrame,
    dense_info: dict[str, Any],
    *,
    use_diversity: bool,
) -> np.ndarray | None:
    """Per-draw ecoregion-only logit offsets, optionally removing diversity from ``g``.

    When ``use_diversity`` is False but ``beta_diversity`` is in the trace, subtracts
    ``beta_diversity * diversity[region]`` from the fitted ecoregion effect so
    ``g`` is counterfactually set to ``mu_global`` only.
    """
    if "ecoregion" not in post:
        return None

    n_test = len(test_df)
    mu_global_draws = post["mu_global"].stack(sample=("chain", "draw")).values
    n_draws = len(mu_global_draws)
    eco_draws = post["ecoregion"].stack(sample=("chain", "draw")).values.T
    beta_div_draws = (
        post["beta_diversity"].stack(sample=("chain", "draw")).values
        if "beta_diversity" in post
        else None
    )
    region_dense_map = dense_info["region_dense_map"]
    div_col = _diversity_column(test_df)

    hier_part = np.full((n_draws, n_test), np.nan, dtype=float)
    for i in range(n_test):
        r_i = region_dense_map.get(str(test_df["region"].iloc[i]))
        if r_i is None:
            hier_part[:, i] = mu_global_draws
            continue

        eco_r = eco_draws[:, int(r_i)]
        if use_diversity or beta_div_draws is None:
            hier_part[:, i] = eco_r
            continue

        d_i = test_df[div_col].iloc[i] if div_col in test_df.columns else np.nan
        if np.isfinite(d_i):
            hier_part[:, i] = eco_r - beta_div_draws * float(d_i)
        else:
            hier_part[:, i] = eco_r
    return hier_part


def _index_maps_from_work(work: pd.DataFrame) -> dict[str, dict[str, int]]:
    site_sr = work[["site", "site_idx"]].drop_duplicates("site")
    reg_sr = work[["region", "region_idx"]].drop_duplicates("region")
    return {
        "site_dense_map": dict(
            zip(site_sr["site"].astype(str), site_sr["site_idx"].astype(int))
        ),
        "region_dense_map": dict(
            zip(reg_sr["region"].astype(str), reg_sr["region_idx"].astype(int))
        ),
    }


def prepare_cv_fold_arrays(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_eps: float = 1e-6,
    *,
    variant: str | Variant = "reparam",
) -> dict[str, Any]:
    """Build arrays for one CV fold (train fit + test prediction).

    ``variant`` may be a variant name or a :class:`~src.models.hbb.variants.Variant`.
    """
    from src.models.hbb.variant_data import build_variant_data
    from src.models.hbb.variants import VARIANTS, Variant, predictors_for_variant

    if isinstance(variant, str):
        if variant not in VARIANTS:
            opts = ", ".join(sorted(VARIANTS))
            raise ValueError(f"Unknown beta variant '{variant}'. Expected one of: {opts}")
        var = VARIANTS[variant]
    else:
        var = variant
    std = standardize_train_test(train_df, test_df)
    tr_raw, te_raw = std["train"], std["test"]

    pack = build_variant_data(tr_raw, var)
    work = pack["df"].copy()
    work["site_idx"] = pack["site_idx"]
    work["region_idx"] = pack["region_idx"]
    n_train = len(work)
    predictors = predictors_for_variant(tr_raw, var)
    X_test, _ = build_design_matrix(
        te_raw, predictors=predictors, add_intercept=var.add_intercept
    )

    y_train = np.clip(pack["y"], y_eps, 1.0 - y_eps)
    index_maps = _index_maps_from_work(work)

    return {
        "X_train": pack["X"],
        "X_test": X_test,
        "y_train": y_train,
        "site_idx": pack["site_idx"],
        "region_idx": pack["region_idx"],
        "site_to_region": pack["site_to_region"],
        "diversity": pack["diversity"],
        "col_names": pack["col_names"],
        "dense_info": index_maps,
        "test_df": te_raw,
        "n_train": n_train,
        "reef_to_site_map": pack["reef_to_site_map"],
        "spec": var.spec,
        "variant": var.name,
        "use_site_hierarchy": var.use_site_hierarchy,
        "use_ecoregion_hierarchy": var.use_ecoregion_hierarchy,
        "use_hierarchy": var.use_hierarchy,
        "use_diversity": var.use_diversity,
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
    n_draws = beta_draws.shape[0]
    n_test = X_test.shape[0]
    fixed_part = beta_draws @ X_test.T

    use_site_hierarchy = getattr(model, "use_site_hierarchy", True)
    use_ecoregion_hierarchy = getattr(model, "use_ecoregion_hierarchy", True)
    hier_part = hierarchical_logit_offsets(
        post,
        test_df,
        dense_info,
        use_site_hierarchy=use_site_hierarchy,
        use_ecoregion_hierarchy=use_ecoregion_hierarchy,
        use_diversity=getattr(model, "use_diversity", True),
    )

    pi_draw = inv_logit(fixed_part + hier_part)
    pred_mean = pi_draw.mean(axis=0)
    pred_lo = np.quantile(pi_draw, 0.025, axis=0)
    pred_hi = np.quantile(pi_draw, 0.975, axis=0)

    y_obs_prop = coral_cover_proportion(test_df["average_coral_cover"].to_numpy())
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
