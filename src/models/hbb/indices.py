from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

IndexMode = Literal["reparam", "legacy_r"]


def _reef_ecoregion_cols(df: pd.DataFrame) -> tuple[str, str]:
    reef = next(c for c in ("reef_id", "reef", "Reef_ID") if c in df.columns)
    eco = next(c for c in ("ecoregion", "Ecoregion") if c in df.columns)
    return reef, eco


def prepare_hierarchical_indices(
    df: pd.DataFrame,
    *,
    mode: IndexMode = "reparam",
) -> dict[str, Any]:
    """
    Build site/region indices for hierarchical random effects.

    Parameters
    ----------
    df : pd.DataFrame
        Model-ready observations.
    mode : str
        ``reparam`` — categorical codes on full data (non-centered pipeline).
        ``legacy_r`` — ``my_1_run_the_beta_model.Rmd`` factor order on distinct
        reef–ecoregion pairs (1-based in R, 0-based for PyMC).
    """
    if mode == "legacy_r":
        return prepare_hierarchical_indices_legacy(df)
    return prepare_hierarchical_indices_reparam(df)


def prepare_hierarchical_indices_reparam(df: pd.DataFrame) -> dict[str, Any]:
    """Dense site/region indices matching ``build_jags_data`` in native R."""
    reef, eco = _reef_ecoregion_cols(df)
    div = "diversity.standardized" if "diversity.standardized" in df.columns else "diversity"

    work = df.copy()
    if {"site", "region"}.issubset(work.columns):
        site_vals = sorted(work["site"].unique())
        site_map = {v: i for i, v in enumerate(site_vals)}
        work["site_idx"] = work["site"].map(site_map).astype(int)

        site_region = (
            work[["site", "region"]].drop_duplicates().sort_values("site")
        )
        if (site_region.groupby("site")["region"].nunique() != 1).any():
            raise ValueError("Sites map to multiple regions.")
        region_vals = sorted(site_region["region"].unique())
        region_map = {v: i for i, v in enumerate(region_vals)}
        work["ecoregion_idx"] = work["region"].map(region_map).astype(int)
    else:
        work["ecoregion_idx"] = pd.Categorical(work[eco]).codes
        work["site_idx"] = pd.Categorical(work[reef]).codes

    site_to_region = (
        work.groupby("site_idx", observed=True)["ecoregion_idx"]
        .first()
        .sort_index()
        .to_numpy(int)
    )
    diversity = (
        work.groupby("ecoregion_idx", observed=True)[div]
        .first()
        .sort_index()
        .to_numpy(float)
    )
    if not np.all(np.isfinite(diversity)):
        raise ValueError("Non-finite diversity.")
    return {
        "site_idx": work["site_idx"].to_numpy(int),
        "region_idx": work["ecoregion_idx"].to_numpy(int),
        "site_to_region": site_to_region,
        "diversity": diversity,
        "n_sites": len(site_to_region),
        "n_regions": len(diversity),
        "reef_to_site_map": dict(zip(work[reef], work["site_idx"])),
        "index_mode": "reparam",
    }


def prepare_hierarchical_indices_legacy(df: pd.DataFrame) -> dict[str, Any]:
    """
    Match ``my_1_run_the_beta_model.Rmd`` site/region IDs after the trusted
    data_for_maps ecoregion mapping has been applied.

    R builds ``sites_and_region_df`` from ``distinct(Reef_ID, Ecoregion)``, then::

        site  <- as.numeric(as.factor(Reef_ID))
        region <- as.numeric(as.factor(Ecoregion))

    ``factor(Reef_ID)`` / ``factor(Ecoregion)`` use R's default **alphabetical**
    level order (after ``my_1`` coerces ``Reef_ID`` on the full table, site codes
    use the full-data reef level set).
    Observations are attached with ``left_join(..., by = "Reef_ID")`` only, so a reef
    with multiple ecoregions in the pairs table can duplicate rows (same as R).

    JAGS uses ``region_for_each_site[i] <- sites_and_region_df$region[i]`` for
    ``i in 1:Nre`` (positional by row, not by the ``site`` column). PyMC stores that
    as ``site_to_region[site_id - 1]``.
    """
    reef, eco = _reef_ecoregion_cols(df)
    div = "diversity.standardized" if "diversity.standardized" in df.columns else "diversity"

    reef_str = df[reef].astype(str)
    eco_str = df[eco].astype(str)
    pairs = (
        pd.DataFrame({reef: reef_str, eco: eco_str})
        .drop_duplicates(ignore_index=True)
    )
    # R default factor() order (alphabetical); Reef_ID is factored on full data in my_1.
    reef_levels = sorted(reef_str.unique())
    eco_levels = sorted(pairs[eco].unique())
    pairs["site"] = pd.Categorical(pairs[reef], categories=reef_levels).codes + 1
    pairs["region"] = pd.Categorical(pairs[eco], categories=eco_levels).codes + 1

    n_sites = len(reef_levels)
    n_regions = len(eco_levels)
    if len(pairs) < n_sites:
        raise ValueError("Legacy pairs table shorter than unique site count.")
    # R: region_for_each_site[i] is the i-th row of sites_and_region_df (1-based).
    site_to_region = pairs["region"].iloc[:n_sites].to_numpy(int) - 1

    lookup = pairs[[reef, "site", "region"]]
    base = df.drop(columns=["site", "region"], errors="ignore").copy()
    base[reef] = reef_str
    work = base.merge(lookup, on=reef, how="left")
    if work["site"].isna().any():
        raise ValueError("Missing site/region mapping after legacy join.")

    site_idx = work["site"].to_numpy(int) - 1
    region_idx = work["region"].to_numpy(int) - 1

    eco_div = (
        df.assign(_eco=eco_str)
        .groupby("_eco", observed=True)[div]
        .agg(lambda s: float(s.dropna().iloc[0]) if s.notna().any() else np.nan)
    )
    diversity = np.array([eco_div[name] for name in eco_levels], dtype=float)
    if not np.all(np.isfinite(diversity)):
        raise ValueError("Non-finite diversity in legacy indexing.")

    reef_to_site_map = (
        pairs.drop_duplicates(subset=reef, keep="first")
        .set_index(reef)["site"]
        .astype(int)
        .sub(1)
        .to_dict()
    )

    out: dict[str, Any] = {
        "site_idx": site_idx,
        "region_idx": region_idx,
        "site_to_region": site_to_region,
        "diversity": diversity,
        "n_sites": n_sites,
        "n_regions": n_regions,
        "reef_to_site_map": reef_to_site_map,
        "index_mode": "legacy_r",
        "site": work["site"].to_numpy(int),
        "region": work["region"].to_numpy(int),
    }
    if len(work) != len(df):
        out["df"] = work.reset_index(drop=True)
    return out


def make_dense_site_region(df: pd.DataFrame) -> dict[str, Any]:
    work = df.copy()
    site_map = {str(s): i for i, s in enumerate(sorted(work["site"].unique()))}
    work["site_dense"] = work["site"].map(lambda s: site_map[str(s)]).astype(int)
    sr = work[["site", "region"]].drop_duplicates().sort_values("site")
    if (sr.groupby("site")["region"].nunique() != 1).any():
        raise ValueError("Sites map to multiple regions.")
    reg_map = {str(r): i for i, r in enumerate(sorted(sr["region"].unique()))}
    work["region_dense"] = work["region"].map(lambda r: reg_map[str(r)]).astype(int)
    site_level = (
        work[["site_dense", "region_dense"]]
        .drop_duplicates()
        .sort_values("site_dense")
        .reset_index(drop=True)
    )
    return {
        "data": work,
        "region_for_each_site": site_level["region_dense"].to_numpy(int),
        "n_sites": len(site_map),
        "n_regions": len(reg_map),
        "site_dense_map": site_map,
        "region_dense_map": reg_map,
    }


def build_region_diversity(df: pd.DataFrame, n_regions: int) -> np.ndarray:
    div = "diversity.standardized" if "diversity.standardized" in df.columns else "diversity"
    reg_div = df.groupby("region_dense", as_index=False)[div].mean().sort_values("region_dense")
    if len(reg_div) != n_regions:
        raise ValueError(f"Diversity length {len(reg_div)} != {n_regions}")
    return reg_div[div].to_numpy(float)
