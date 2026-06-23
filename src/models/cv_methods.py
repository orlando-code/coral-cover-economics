"""Cross-validation fold builders shared across models.

This module centralizes the definition of CV *regimes* (random, grouped, time-blocked,
spatial, etc.) so that all models in the project use consistent splitting logic and
metadata.

A *fold* is represented as a dict with keys:
- name: regime name
- fold: 1-based fold id
- train_idx: numpy array of row indices
- test_idx: numpy array of row indices
- meta: dict of regime-specific metadata (optional but recommended)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

ValidationRegime = str

ALL_CV_REGIMES: tuple[ValidationRegime, ...] = (
    "random_kfold",
    "site_group_kfold",
    "ecoregion_group_kfold",
    "forward_time_blocks",
    "forward_repeat_sites",
    # "spatial_kfold",
    "in_sample",
    "in_sample_multi_visit",
)

_EPOCH_19811231 = pd.Timestamp("1981-12-31")

IN_SAMPLE_REGIMES: frozenset[ValidationRegime] = frozenset(
    {"in_sample", "in_sample_multi_visit"}
)


def is_in_sample_regime(regime: str) -> bool:
    return regime in IN_SAMPLE_REGIMES


@dataclass(frozen=True)
class FoldSpec:
    name: ValidationRegime
    fold: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    meta: dict[str, Any]


def pick_first_existing(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def year_series(df: pd.DataFrame) -> pd.Series:
    """Calendar year per row (derived from ``days_since_19811231`` when needed)."""
    if "year" in df.columns:
        return df["year"].astype(int)
    if "Year" in df.columns:
        return df["Year"].astype(int)
    days_col = pick_first_existing(df, ["days_since_19811231"])
    if days_col is None:
        raise ValueError(
            "No year column found (expected 'year', 'Year', or 'days_since_19811231')."
        )
    return (
        _EPOCH_19811231 + pd.to_timedelta(df[days_col].astype(float), unit="D")
    ).dt.year.astype(int)


def _split_forward_repeat_sites(
    df: pd.DataFrame,
    *,
    cutoff_year: int,
    site_col: str,
    years: pd.Series,
) -> tuple[np.ndarray, np.ndarray, set[Any]]:
    """Train/test indices for one temporal cutoff (test = repeat-site future rows)."""
    year_arr = years.to_numpy()
    site_arr = df[site_col].to_numpy()
    train_mask = year_arr < cutoff_year
    test_future_mask = year_arr >= cutoff_year

    sites_with_train = set(site_arr[train_mask])
    sites_with_test = set(site_arr[test_future_mask])
    repeat_sites = sites_with_train & sites_with_test

    repeat_site_mask = np.isin(site_arr, list(repeat_sites))
    test_mask = test_future_mask & repeat_site_mask

    train_idx = np.flatnonzero(train_mask)
    test_idx = np.flatnonzero(test_mask)
    return train_idx, test_idx, repeat_sites


def make_folds_forward_repeat_sites(
    df: pd.DataFrame,
    *,
    site_col: str = "site",
    test_fraction: float = 0.2,
    regime_name: str = "forward_repeat_sites",
) -> list[FoldSpec]:
    """Temporal holdout on longitudinally sampled sites.

    Train on all rows before a calendar-year cutoff. Test on post-cutoff rows at
    sites with at least one pre-cutoff and one post-cutoff observation.

    The cutoff is chosen to make the eligible test rows ~``test_fraction`` of
    ``n_train + n_test`` (optimizing on repeat-site future rows only).
    """
    if site_col not in df.columns:
        raise ValueError(f"{regime_name} requires a '{site_col}' column")

    years = year_series(df)
    unique_years = sorted(years.unique())
    if len(unique_years) < 2:
        raise ValueError(f"{regime_name} requires observations in at least two years.")

    best: Optional[dict[str, Any]] = None
    for cutoff_year in unique_years[1:]:
        train_idx, test_idx, repeat_sites = _split_forward_repeat_sites(
            df,
            cutoff_year=int(cutoff_year),
            site_col=site_col,
            years=years,
        )
        n_train = len(train_idx)
        n_test = len(test_idx)
        if n_train == 0 or n_test == 0:
            continue

        eval_total = n_train + n_test
        frac_test = n_test / eval_total
        err = abs(frac_test - test_fraction)
        candidate = {
            "cutoff_year": int(cutoff_year),
            "train_idx": train_idx,
            "test_idx": test_idx,
            "repeat_sites": repeat_sites,
            "n_train": n_train,
            "n_test": n_test,
            "frac_test": frac_test,
            "err": err,
        }
        if best is None or candidate["err"] < best["err"]:
            best = candidate
        elif candidate["err"] == best["err"] and candidate["n_test"] > best["n_test"]:
            best = candidate

    if best is None:
        raise ValueError(
            f"{regime_name}: no cutoff produced non-empty train and repeat-site test sets."
        )

    repeat_sites = best["repeat_sites"]
    return [
        FoldSpec(
            name=regime_name,
            fold=1,
            train_idx=best["train_idx"],
            test_idx=best["test_idx"],
            meta={
                "time_col": "year",
                "site_col": site_col,
                "cutoff_year": best["cutoff_year"],
                "test_fraction_target": float(test_fraction),
                "test_fraction_actual": float(best["frac_test"]),
                "n_train": int(best["n_train"]),
                "n_test": int(best["n_test"]),
                "n_repeat_sites": int(len(repeat_sites)),
                "year_min": int(years.min()),
                "year_max": int(years.max()),
            },
        )
    ]


def make_folds_random(df: pd.DataFrame, k: int, seed: int) -> list[FoldSpec]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    fold_id = np.arange(len(idx)) % k
    folds: list[FoldSpec] = []
    for f in range(k):
        test = idx[fold_id == f]
        train = np.setdiff1d(np.arange(len(df)), test)
        folds.append(
            FoldSpec(
                name="random_kfold",
                fold=f + 1,
                train_idx=train,
                test_idx=test,
                meta={"seed": seed, "k": k},
            )
        )
    return folds


def make_folds_group(
    df: pd.DataFrame,
    group_col: str,
    k: int,
    seed: int,
    regime_name: str,
) -> list[FoldSpec]:
    rng = np.random.default_rng(seed)
    groups = rng.permutation(sorted(df[group_col].unique()))
    fold_id = np.arange(len(groups)) % k
    folds: list[FoldSpec] = []
    for f in range(k):
        test_groups = groups[fold_id == f]
        test = np.flatnonzero(df[group_col].isin(test_groups).to_numpy())
        train = np.setdiff1d(np.arange(len(df)), test)
        folds.append(
            FoldSpec(
                name=regime_name,
                fold=f + 1,
                train_idx=train,
                test_idx=test,
                meta={
                    "seed": seed,
                    "k": k,
                    "group_col": group_col,
                    "n_groups": int(pd.Series(groups).nunique()),
                    "n_test_groups": int(len(test_groups)),
                },
            )
        )
    return folds


def make_folds_forward_time(
    df: pd.DataFrame,
    k: int,
    time_col: str,
    regime_name: str = "forward_time_blocks",
) -> list[FoldSpec]:
    times = sorted(df[time_col].unique())
    block_id = pd.cut(range(len(times)), bins=k, labels=False) + 1
    time_block = dict(zip(times, block_id))
    row_block = df[time_col].map(time_block).to_numpy()

    folds: list[FoldSpec] = []
    # Use blocks 2..k as test; earlier blocks form training set.
    for b in range(2, k + 1):
        train = np.flatnonzero(row_block < b)
        test = np.flatnonzero(row_block == b)
        if len(train) == 0 or len(test) == 0:
            continue
        folds.append(
            FoldSpec(
                name=regime_name,
                fold=int(b),
                train_idx=train,
                test_idx=test,
                meta={"k": k, "time_col": time_col, "time_block": int(b)},
            )
        )
    return folds


def make_folds_spatial(
    df: pd.DataFrame,
    k: int,
    n_bins: int,
    seed: int,
    regime_name: str = "spatial_kfold",
) -> list[FoldSpec]:
    lon_col = pick_first_existing(df, ["longitude.degrees", "lon"])
    lat_col = pick_first_existing(df, ["latitude.degrees", "lat"])
    if lon_col is None or lat_col is None:
        raise ValueError("Spatial folds require longitude and latitude columns.")

    rng = np.random.default_rng(seed)
    lon_bin = pd.cut(df[lon_col], bins=n_bins, include_lowest=True, labels=False)
    lat_bin = pd.cut(df[lat_col], bins=n_bins, include_lowest=True, labels=False)
    block = lon_bin.astype(str) + "_" + lat_bin.astype(str)
    blocks = rng.permutation(block.unique())
    fold_id = np.arange(len(blocks)) % k
    block_to_fold = dict(zip(blocks, fold_id))
    row_fold = block.map(block_to_fold)

    folds: list[FoldSpec] = []
    for f in range(k):
        test = np.flatnonzero((row_fold == f).to_numpy())
        train = np.setdiff1d(np.arange(len(df)), test)
        folds.append(
            FoldSpec(
                name=regime_name,
                fold=f + 1,
                train_idx=train,
                test_idx=test,
                meta={
                    "seed": seed,
                    "k": k,
                    "n_bins": n_bins,
                    "lon_col": lon_col,
                    "lat_col": lat_col,
                },
            )
        )
    return folds


def make_folds_in_sample(
    df: pd.DataFrame, regime_name: str = "in_sample"
) -> list[FoldSpec]:
    """Single identity split: train and test on the full dataset."""
    idx = np.arange(len(df), dtype=int)
    return [
        FoldSpec(
            name=regime_name,
            fold=1,
            train_idx=idx,
            test_idx=idx.copy(),
            meta={"n_rows": int(len(df)), "split": "identity"},
        )
    ]


def make_folds_in_sample_multi_visit(
    df: pd.DataFrame,
    *,
    min_site_measurements: int = 1,
    site_col: str = "site",
    regime_name: str = "in_sample_multi_visit",
) -> list[FoldSpec]:
    """In-sample split restricted to rows at sites with > ``min_site_measurements`` visits.

    Default ``min_site_measurements=1`` keeps sites with at least two observations.
    Train and test indices are identical (as for ``in_sample``), but both are limited
    to the eligible subset.
    """
    if site_col not in df.columns:
        raise ValueError(f"{regime_name} requires a '{site_col}' column")
    if min_site_measurements < 1:
        raise ValueError(
            f"{regime_name}: min_site_measurements must be >= 1 "
            f"(got {min_site_measurements})"
        )

    site_counts = df.groupby(site_col, dropna=False).size()
    eligible_sites = site_counts[site_counts > min_site_measurements].index
    if len(eligible_sites) == 0:
        raise ValueError(
            f"{regime_name}: no sites with more than {min_site_measurements} measurements"
        )

    eligible_mask = df[site_col].isin(eligible_sites).to_numpy()
    idx = np.flatnonzero(eligible_mask)
    if len(idx) == 0:
        raise ValueError(f"{regime_name}: no rows at eligible sites")

    return [
        FoldSpec(
            name=regime_name,
            fold=1,
            train_idx=idx,
            test_idx=idx.copy(),
            meta={
                "n_rows": int(len(idx)),
                "n_rows_total": int(len(df)),
                "split": "identity",
                "site_col": site_col,
                "min_site_measurements": int(min_site_measurements),
                "min_site_rows_required": int(min_site_measurements + 1),
                "n_sites": int(len(eligible_sites)),
                "n_sites_total": int(site_counts.shape[0]),
            },
        )
    ]


def build_all_folds(
    df: pd.DataFrame,
    *,
    validation_regimes: list[str],
    k_folds: int,
    seed: int,
    spatial_bins: int = 4,
    time_col_candidates: Optional[list[str]] = None,
    in_sample_min_site_measurements: int = 1,
) -> tuple[list[FoldSpec], list[dict[str, Any]]]:
    """Build fold specs for the requested validation regimes.

    Notes
    -----
    - Some regimes may be skipped if required columns are missing.
    - forward_time_blocks yields fewer than k folds (blocks 2..k).
    - forward_repeat_sites yields one temporal holdout on repeat-visit sites.
    - in_sample yields one fold with identical train and test indices.
    - in_sample_multi_visit is in_sample on sites with
      > ``in_sample_min_site_measurements`` rows (default: at least 2 visits).
    """
    regimes = list(validation_regimes)
    time_col_candidates = time_col_candidates or [
        "days_since_19811231",
        "year",
        "Year",
    ]

    all_folds: list[FoldSpec] = []
    skipped: list[dict[str, Any]] = []

    if "random_kfold" in regimes:
        try:
            all_folds.extend(make_folds_random(df, k_folds, seed + 11))
        except Exception as exc:  # noqa: BLE001
            skipped.append({"regime": "random_kfold", "reason": str(exc)})

    if "site_group_kfold" in regimes:
        try:
            if "site" not in df.columns:
                raise ValueError("site_group_kfold requires a 'site' column")
            all_folds.extend(
                make_folds_group(df, "site", k_folds, seed + 23, "site_group_kfold")
            )
        except Exception as exc:  # noqa: BLE001
            skipped.append({"regime": "site_group_kfold", "reason": str(exc)})

    if "ecoregion_group_kfold" in regimes:
        try:
            if "region" not in df.columns:
                raise ValueError("ecoregion_group_kfold requires a 'region' column")
            all_folds.extend(
                make_folds_group(
                    df, "region", k_folds, seed + 37, "ecoregion_group_kfold"
                )
            )
        except Exception as exc:  # noqa: BLE001
            skipped.append({"regime": "ecoregion_group_kfold", "reason": str(exc)})

    if "forward_time_blocks" in regimes:
        try:
            time_col = pick_first_existing(df, time_col_candidates)
            if time_col is None:
                raise ValueError(
                    "No time column found (tried: "
                    + ", ".join(time_col_candidates)
                    + ")"
                )
            all_folds.extend(make_folds_forward_time(df, k_folds, time_col))
        except Exception as exc:  # noqa: BLE001
            skipped.append({"regime": "forward_time_blocks", "reason": str(exc)})

    if "forward_repeat_sites" in regimes:
        try:
            if "site" not in df.columns:
                raise ValueError("forward_repeat_sites requires a 'site' column")
            year_series(df)  # validate time is available
            all_folds.extend(make_folds_forward_repeat_sites(df))
        except Exception as exc:  # noqa: BLE001
            skipped.append({"regime": "forward_repeat_sites", "reason": str(exc)})

    if "spatial_kfold" in regimes:
        try:
            all_folds.extend(make_folds_spatial(df, k_folds, spatial_bins, seed + 53))
        except Exception as exc:  # noqa: BLE001
            skipped.append({"regime": "spatial_kfold", "reason": str(exc)})

    if "in_sample" in regimes:
        try:
            all_folds.extend(make_folds_in_sample(df))
        except Exception as exc:  # noqa: BLE001
            skipped.append({"regime": "in_sample", "reason": str(exc)})

    if "in_sample_multi_visit" in regimes:
        try:
            all_folds.extend(
                make_folds_in_sample_multi_visit(
                    df,
                    min_site_measurements=in_sample_min_site_measurements,
                )
            )
        except Exception as exc:  # noqa: BLE001
            skipped.append({"regime": "in_sample_multi_visit", "reason": str(exc)})

    if not all_folds:
        # If everything was skipped, surface a helpful error.
        reasons = "; ".join([f"{d['regime']}: {d['reason']}" for d in skipped])
        raise RuntimeError(f"No validation folds were created. Skips: {reasons}")

    return all_folds, skipped


def fold_manifest_dataframe(folds: list[FoldSpec]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for f in folds:
        rows.append(
            {
                "fold_tag": f"{f.name}__{f.fold}",
                "regime": f.name,
                "fold": int(f.fold),
                "n_train": int(len(f.train_idx)),
                "n_test": int(len(f.test_idx)),
                **{f"meta__{k}": v for k, v in (f.meta or {}).items()},
            }
        )
    return pd.DataFrame(rows)
