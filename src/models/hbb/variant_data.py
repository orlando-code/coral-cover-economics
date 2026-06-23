"""Build design matrices and hierarchical indices for a :class:`Variant`."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.models.hbb.design import build_design_matrix, transform_to_beta
from src.models.hbb.indices import prepare_hierarchical_indices
from src.models.hbb.variants import Variant, predictors_for_variant


def r_lexicographic_factor_codes(values: np.ndarray) -> np.ndarray:
    """Mimic ``as.integer(as.factor(as.character(values))) - 1``."""
    labels = sorted({str(int(v) + 1) for v in values})
    mapping = {label: i for i, label in enumerate(labels)}
    return np.array([mapping[str(int(v) + 1)] for v in values], dtype=int)


def infer_design_col_names(variant: Variant, n_beta: int) -> list[str]:
    """Reconstruct design-matrix column names for a variant (for post-hoc plots)."""
    from src.models.hbb._config import CV_PREDICTORS

    dummy = pd.DataFrame({col: [0.0] for col in CV_PREDICTORS})
    predictors = predictors_for_variant(dummy, variant)
    _, col_names = build_design_matrix(
        dummy, predictors=predictors, add_intercept=variant.add_intercept
    )
    if len(col_names) != n_beta:
        raise ValueError(
            f"Inferred {len(col_names)} columns for {variant.name}, "
            f"but trace has {n_beta} beta coefficients."
        )
    return col_names


def build_variant_data(df_std: pd.DataFrame, variant: Variant) -> dict[str, Any]:
    hier = prepare_hierarchical_indices(df_std, mode=variant.index_mode)
    work = hier.get("df", df_std).reset_index(drop=True)

    site_to_region = np.array(hier["site_to_region"], dtype=int)
    if variant.paper_factor_encoding:
        site_to_region = r_lexicographic_factor_codes(site_to_region)

    predictors = predictors_for_variant(work, variant)
    X, col_names = build_design_matrix(
        work, predictors=predictors, add_intercept=variant.add_intercept
    )
    y = transform_to_beta(work["average_coral_cover"].to_numpy(float), len(work))
    diversity = np.array(hier["diversity"], dtype=float)

    return {
        "df": work,
        "X": X,
        "y": y,
        "col_names": col_names,
        "site_idx": np.array(hier["site_idx"], dtype=int),
        "region_idx": np.array(hier["region_idx"], dtype=int),
        "site_to_region": site_to_region,
        "diversity": diversity,
        "reef_to_site_map": hier["reef_to_site_map"],
        "input_summary": {
            "variant": variant.name,
            "N": int(len(work)),
            "K": int(X.shape[1]),
            "Nre": int(len(np.unique(hier["site_idx"]))),
            "R": int(len(diversity)),
            "n_site": int(work["site"].nunique()) if "site" in work.columns else np.nan,
            "n_region": int(work["region"].nunique())
            if "region" in work.columns
            else np.nan,
            "n_erg": int(work["erg"].nunique()) if "erg" in work.columns else np.nan,
            "spec": variant.spec,
            "add_intercept": variant.add_intercept,
            "index_mode": variant.index_mode,
            "paper_factor_encoding": variant.paper_factor_encoding,
            "latitude_transform": variant.latitude_transform,
            "exclude_vars": "|".join(variant.exclude_vars),
            "use_hierarchy": variant.use_hierarchy,
            "use_diversity": variant.use_diversity,
            "X_columns": "|".join(col_names),
        },
        "use_hierarchy": variant.use_hierarchy,
        "use_diversity": variant.use_diversity,
    }
