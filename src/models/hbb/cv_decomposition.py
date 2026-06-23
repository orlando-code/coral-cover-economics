"""Hierarchy and environmental explanatory-power decomposition for CV folds."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit as inv_logit
from sklearn.metrics import r2_score

from src.models.hbb.cv import (
    coral_cover_proportion,
    ecoregion_only_logit_offsets,
    hierarchical_logit_offsets,
)
from src.models.hbb.design import inverse_transform_beta

try:
    import arviz as az

    HAS_ARVIZ = True
except ImportError:
    HAS_ARVIZ = False


def _resolve_hierarchy_flags(
    variant: str | None,
    *,
    use_site_hierarchy: bool | None = None,
    use_ecoregion_hierarchy: bool | None = None,
    use_diversity: bool | None = None,
) -> tuple[bool, bool, bool]:
    from src.models.hbb.variants import VARIANTS

    var = VARIANTS.get(variant or "")
    use_site = (
        use_site_hierarchy
        if use_site_hierarchy is not None
        else (var.use_site_hierarchy if var else True)
    )
    use_eco = (
        use_ecoregion_hierarchy
        if use_ecoregion_hierarchy is not None
        else (var.use_ecoregion_hierarchy if var else True)
    )
    use_div = (
        use_diversity
        if use_diversity is not None
        else (var.use_diversity if var else True)
    )
    return use_site, use_eco, use_div


def flatten_decomposition_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Flatten one fold decomposition dict to a single CSV row."""
    row = {
        "fold_tag": summary.get("fold_tag"),
        "variant": summary.get("variant"),
        "n_test": summary.get("n_test"),
        "n_sites_test": summary.get("n_sites_test"),
        "n_sites_within_site_valid": summary.get("n_sites_within_site_valid"),
        "r2_within_site_unweighted_mean": summary.get("r2_within_site_unweighted_mean"),
        "r2_within_site_n_weighted_mean": summary.get("r2_within_site_n_weighted_mean"),
        "r2_within_site_median": summary.get("r2_within_site_median"),
    }
    for k, v in (summary.get("r2_global") or {}).items():
        row[f"r2_{k}"] = v
    for k, v in (summary.get("delta_r2") or {}).items():
        row[f"delta_{k}"] = v
    return row


def write_variant_decomposition_summary(
    summaries: list[dict[str, Any]],
    variant_dir: Path,
) -> None:
    """Write per-variant rollup CSV from in-memory fold summaries."""
    if not summaries:
        return
    pd.DataFrame([flatten_decomposition_summary(s) for s in summaries]).to_csv(
        Path(variant_dir) / "hierarchy_decomposition_summary.csv",
        index=False,
    )


def upsert_hierarchy_decomposition_rows(
    beta_glmm_root: Path,
    summaries: list[dict[str, Any]],
) -> None:
    """Merge fold decomposition summaries into the cross-variant rollup CSV."""
    if not summaries:
        return
    beta_glmm_root = Path(beta_glmm_root)
    path = beta_glmm_root / "hierarchy_decomposition_all_variants.csv"
    new_rows = pd.DataFrame([flatten_decomposition_summary(s) for s in summaries])
    if path.exists():
        existing = pd.read_csv(path)
        replace_keys = set(zip(new_rows["fold_tag"], new_rows["variant"]))
        keep = [
            (fold_tag, variant) not in replace_keys
            for fold_tag, variant in zip(existing["fold_tag"], existing["variant"])
        ]
        combined = pd.concat([existing.loc[keep], new_rows], ignore_index=True)
    else:
        combined = new_rows
    combined.to_csv(path, index=False)


def site_mean_baseline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    site_col: str = "site",
    cover_col: str = "average_coral_cover",
) -> np.ndarray:
    """Per-row predictions using each site's training-period mean cover."""
    train_means = (
        train_df.groupby(site_col, dropna=False)[cover_col]
        .apply(lambda s: coral_cover_proportion(s.to_numpy()).mean())
        .to_dict()
    )
    global_mean = float(coral_cover_proportion(train_df[cover_col].to_numpy()).mean())
    return np.array(
        [
            train_means.get(test_df[site_col].iloc[i], global_mean)
            for i in range(len(test_df))
        ],
        dtype=float,
    )


def random_effect_only_predictions(
    trace_path: Path,
    test_df: pd.DataFrame,
    dense_info: dict[str, Any],
    n_train: int,
    *,
    use_site_hierarchy: bool = True,
    use_ecoregion_hierarchy: bool = True,
    use_diversity: bool = True,
) -> np.ndarray | None:
    """Posterior mean cover from random effects only (no fixed-effect term)."""
    if not HAS_ARVIZ or not trace_path.exists():
        return None

    post = az.from_netcdf(trace_path).posterior
    hier_part = hierarchical_logit_offsets(
        post,
        test_df,
        dense_info,
        use_site_hierarchy=use_site_hierarchy,
        use_ecoregion_hierarchy=use_ecoregion_hierarchy,
        use_diversity=use_diversity,
    )
    if not isinstance(hier_part, np.ndarray):
        return None

    pred_mean = inv_logit(hier_part).mean(axis=0)
    return inverse_transform_beta(pred_mean, n_train)


def ecoregion_only_predictions(
    trace_path: Path,
    test_df: pd.DataFrame,
    dense_info: dict[str, Any],
    n_train: int,
    *,
    use_diversity: bool,
) -> np.ndarray | None:
    """Posterior mean cover from ecoregion random effects only (no site, no fixed effects)."""
    if not HAS_ARVIZ or not trace_path.exists():
        return None

    post = az.from_netcdf(trace_path).posterior
    hier_part = ecoregion_only_logit_offsets(
        post,
        test_df,
        dense_info,
        use_diversity=use_diversity,
    )
    if hier_part is None:
        return None

    pred_mean = inv_logit(hier_part).mean(axis=0)
    return inverse_transform_beta(pred_mean, n_train)


def fixed_effect_only_predictions(
    trace_path: Path,
    X_test: np.ndarray,
    n_train: int,
) -> np.ndarray | None:
    """Posterior mean cover from beta @ X only (no random effects)."""
    if not HAS_ARVIZ or not trace_path.exists():
        return None
    post = az.from_netcdf(trace_path).posterior
    beta_draws = post["beta"].stack(sample=("chain", "draw")).values.T
    fixed_part = beta_draws @ X_test.T
    pi_draw = inv_logit(fixed_part)
    pred_mean = pi_draw.mean(axis=0)
    return inverse_transform_beta(pred_mean, n_train)


def within_site_r2_table(
    merged: pd.DataFrame,
    *,
    site_col: str = "site",
    y_col: str = "y_obs",
    pred_cols: dict[str, str],
) -> pd.DataFrame:
    """Per-site R² for one or more prediction columns."""
    rows: list[dict[str, Any]] = []
    for site, grp in merged.groupby(site_col, dropna=False):
        row: dict[str, Any] = {
            "site": site,
            "n_test": int(len(grp)),
            "y_std": float(grp[y_col].std(ddof=0)) if len(grp) > 1 else 0.0,
        }
        for label, col in pred_cols.items():
            if len(grp) < 2:
                row[f"r2_{label}"] = float("nan")
            else:
                row[f"r2_{label}"] = r2_score(
                    grp[y_col].to_numpy(), grp[col].to_numpy()
                )
        rows.append(row)
    return pd.DataFrame(rows)


def variance_partition_from_trace(trace_path: Path) -> dict[str, float]:
    """Posterior means of hierarchy variance scales (when present)."""
    if not HAS_ARVIZ or not trace_path.exists():
        return {}
    post = az.from_netcdf(trace_path).posterior
    out: dict[str, float] = {}
    for var in ("sigma_site", "sigma", "sigma_ecoregion", "theta"):
        if var in post:
            out[f"{var}_mean"] = float(
                post[var].stack(sample=("chain", "draw")).values.mean()
            )
    if "sigma_site" in out and "sigma_ecoregion" in out:
        ss = out["sigma_site_mean"] ** 2
        se = out["sigma_ecoregion_mean"] ** 2
        out["icc_site"] = float(ss / (ss + se)) if (ss + se) > 0 else float("nan")
        out["icc_ecoregion"] = float(se / (ss + se)) if (ss + se) > 0 else float("nan")
    return out


def rebuild_fold_context(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    variant: str | None,
) -> dict[str, Any]:
    """Rebuild dense indices and X_test for post-hoc fold analysis."""
    from src.models.hbb.cv import prepare_cv_fold_arrays

    arrays = prepare_cv_fold_arrays(train_df, test_df, variant=variant or "reparam")
    return {
        "dense_info": arrays["dense_info"],
        "X_test": arrays["X_test"],
        "n_train": arrays["n_train"],
        "use_site_hierarchy": arrays["use_site_hierarchy"],
        "use_ecoregion_hierarchy": arrays["use_ecoregion_hierarchy"],
        "use_diversity": arrays["use_diversity"],
    }


def compute_fold_decomposition(
    *,
    predictions: pd.DataFrame,
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    fold_dir: Path,
    variant: str | None = None,
    n_train: int | None = None,
    dense_info: dict[str, Any] | None = None,
    X_test: np.ndarray | None = None,
    use_site_hierarchy: bool | None = None,
    use_ecoregion_hierarchy: bool | None = None,
    use_diversity: bool | None = None,
) -> dict[str, Any]:
    """Compute hierarchy / env decomposition metrics for one CV fold."""
    use_site, use_eco, use_div = _resolve_hierarchy_flags(
        variant,
        use_site_hierarchy=use_site_hierarchy,
        use_ecoregion_hierarchy=use_ecoregion_hierarchy,
        use_diversity=use_diversity,
    )

    if n_train is None:
        n_train = len(train_df)

    if not dense_info or X_test is None:
        ctx = rebuild_fold_context(train_df, test_df, variant)
        dense_info = dense_info or ctx["dense_info"]
        X_test = X_test if X_test is not None else ctx["X_test"]
        n_train = ctx["n_train"]
        if use_site_hierarchy is None:
            use_site = ctx["use_site_hierarchy"]
        if use_ecoregion_hierarchy is None:
            use_eco = ctx["use_ecoregion_hierarchy"]
        if use_diversity is None:
            use_div = ctx["use_diversity"]

    merged = predictions.merge(
        test_df[["row_id", "site", "region"]],
        on="row_id",
        how="left",
    )
    y = merged["y_obs"].to_numpy(dtype=float)

    site_mean_pred = site_mean_baseline(train_df, test_df)
    merged["y_pred_site_mean"] = site_mean_pred

    trace_path = fold_dir / "trace.nc"
    re_only_pred = random_effect_only_predictions(
        trace_path,
        test_df,
        dense_info or {},
        n_train,
        use_site_hierarchy=use_site,
        use_ecoregion_hierarchy=use_eco,
        use_diversity=use_div,
    )
    if re_only_pred is not None:
        merged["y_pred_re_only"] = re_only_pred

    if use_eco:
        re_eco_nodiv = ecoregion_only_predictions(
            trace_path,
            test_df,
            dense_info or {},
            n_train,
            use_diversity=False,
        )
        re_eco_div = ecoregion_only_predictions(
            trace_path,
            test_df,
            dense_info or {},
            n_train,
            use_diversity=True,
        )
        if re_eco_nodiv is not None:
            merged["y_pred_re_eco_nodiv"] = re_eco_nodiv
        if re_eco_div is not None:
            merged["y_pred_re_eco_div"] = re_eco_div

    if X_test is not None:
        fe_only_pred = fixed_effect_only_predictions(trace_path, X_test, n_train)
        if fe_only_pred is not None:
            merged["y_pred_fe_only"] = fe_only_pred

    pred_cols = {
        "model": "y_pred",
        "site_mean": "y_pred_site_mean",
    }
    if "y_pred_re_only" in merged.columns:
        pred_cols["re_only"] = "y_pred_re_only"
    if "y_pred_re_eco_nodiv" in merged.columns:
        pred_cols["re_eco_nodiv"] = "y_pred_re_eco_nodiv"
    if "y_pred_re_eco_div" in merged.columns:
        pred_cols["re_eco_div"] = "y_pred_re_eco_div"
    if "y_pred_fe_only" in merged.columns:
        pred_cols["fe_only"] = "y_pred_fe_only"

    per_site = within_site_r2_table(merged, pred_cols=pred_cols)
    per_site_valid = per_site[
        (per_site["n_test"] >= 2) & (per_site["y_std"] > 1e-4)
    ].copy()
    per_site.to_csv(fold_dir / "within_site_r2.csv", index=False)
    if len(per_site_valid):
        per_site_valid.to_csv(fold_dir / "within_site_r2_valid.csv", index=False)

    def _wmean(frame: pd.DataFrame, col: str, weight_col: str = "n_test") -> float:
        w = frame[weight_col].to_numpy(dtype=float)
        v = frame[col].to_numpy(dtype=float)
        mask = np.isfinite(v) & (w > 0)
        if not mask.any():
            return float("nan")
        return float(np.average(v[mask], weights=w[mask]))

    r2_global = {k: r2_score(y, merged[col].to_numpy()) for k, col in pred_cols.items()}
    r2_within_site_mean = (
        float(per_site_valid["r2_model"].mean())
        if len(per_site_valid)
        else float("nan")
    )
    r2_within_site_weighted = (
        _wmean(per_site_valid, "r2_model") if len(per_site_valid) else float("nan")
    )
    r2_within_site_median = (
        float(per_site_valid["r2_model"].median())
        if len(per_site_valid)
        else float("nan")
    )

    summary: dict[str, Any] = {
        "variant": variant,
        "n_test": int(len(merged)),
        "n_sites_test": int(merged["site"].nunique()),
        "r2_global": r2_global,
        "r2_within_site_unweighted_mean": r2_within_site_mean,
        "r2_within_site_n_weighted_mean": r2_within_site_weighted,
        "r2_within_site_median": r2_within_site_median,
        "n_sites_within_site_valid": int(len(per_site_valid)),
        "delta_r2": {},
        "variance_partition": variance_partition_from_trace(trace_path),
    }

    if "model" in r2_global and "site_mean" in r2_global:
        summary["delta_r2"]["model_minus_site_mean"] = (
            r2_global["model"] - r2_global["site_mean"]
        )
    if "model" in r2_global and "re_only" in r2_global:
        summary["delta_r2"]["model_minus_re_only"] = (
            r2_global["model"] - r2_global["re_only"]
        )
        summary["delta_r2"]["re_only_minus_site_mean"] = (
            r2_global["re_only"] - r2_global["site_mean"]
        )
    if "re_eco_nodiv" in r2_global and "site_mean" in r2_global:
        summary["delta_r2"]["re_eco_nodiv_minus_site_mean"] = (
            r2_global["re_eco_nodiv"] - r2_global["site_mean"]
        )
    if "re_eco_div" in r2_global and "re_eco_nodiv" in r2_global:
        summary["delta_r2"]["re_eco_div_minus_nodiv"] = (
            r2_global["re_eco_div"] - r2_global["re_eco_nodiv"]
        )
    if "model" in r2_global and "fe_only" in r2_global:
        summary["delta_r2"]["model_minus_fe_only"] = (
            r2_global["model"] - r2_global["fe_only"]
        )
        summary["delta_r2"]["fe_only_minus_site_mean"] = (
            r2_global["fe_only"] - r2_global["site_mean"]
        )

    (fold_dir / "hierarchy_decomposition.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n"
    )
    merged.to_csv(fold_dir / "predictions_decomposition.csv", index=False)
    return summary


def load_fold_train_test_frames(
    fold_tag: str,
    regime: str,
    *,
    seed: int = 42,
    k_folds: int = 5,
    in_sample_min_site_measurements: int | None = None,
    fold_manifest_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild train/test frames from cached model-ready data and CV fold manifest."""
    from src.dataloading.build_sully_model_ready_data import to_hbb_frame
    from src.models.coral_data import load_model_ready_data
    from src.models.cv_methods import build_all_folds

    min_site = in_sample_min_site_measurements
    if (
        min_site is None
        and fold_manifest_path is not None
        and fold_manifest_path.exists()
    ):
        manifest = pd.read_csv(fold_manifest_path)
        row = manifest.loc[manifest["fold_tag"] == fold_tag]
        meta_col = "meta__min_site_measurements"
        if (
            not row.empty
            and meta_col in row.columns
            and pd.notna(row.iloc[0][meta_col])
        ):
            min_site = int(row.iloc[0][meta_col])
    if min_site is None:
        min_site = 1

    df = load_model_ready_data()
    folds, _ = build_all_folds(
        df,
        validation_regimes=[regime],
        k_folds=k_folds,
        seed=seed,
        in_sample_min_site_measurements=min_site,
    )
    fold = next(
        (f for f in folds if f"{f.name}__{f.fold}" == fold_tag),
        None,
    )
    if fold is None:
        raise ValueError(f"Fold {fold_tag!r} not found for regime {regime!r}")

    hbb = to_hbb_frame(df)
    train_df = hbb.iloc[fold.train_idx].reset_index(drop=True)
    test_df = hbb.iloc[fold.test_idx].reset_index(drop=True)
    return train_df, test_df


def _read_fold_csv(path: Path) -> pd.DataFrame:
    """Read a fold CSV with one retry for slow/cloud-backed paths."""
    import time

    last_err: Exception | None = None
    for attempt in range(2):
        try:
            return pd.read_csv(path)
        except (TimeoutError, OSError) as exc:
            last_err = exc
            if attempt == 0:
                time.sleep(1.0)
                continue
            raise
    raise last_err  # pragma: no cover


def _load_fold_frames(
    fold_dir: Path,
    *,
    fold_tag: str,
    regime: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train/test; prefer manifest rebuild over fold CSVs on OneDrive."""
    try:
        return load_fold_train_test_frames(
            fold_tag,
            regime,
            fold_manifest_path=fold_dir.parent.parent / "fold_manifest.csv",
        )
    except Exception:
        pass

    train_path = fold_dir / "train_df.csv"
    test_path = fold_dir / "test_df.csv"
    variant_dir = fold_dir.parent.parent
    if not train_path.exists():
        train_path = variant_dir / "train_df.csv"
    if not test_path.exists():
        test_path = variant_dir / "test_df.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Missing train/test CSVs under {fold_dir} and manifest rebuild failed."
        )
    return _read_fold_csv(train_path), _read_fold_csv(test_path)


DIVERSITY_SPLIT_COLS = ("y_pred_re_eco_nodiv", "y_pred_re_eco_div")


def _decomposition_has_diversity_split(out_path: Path) -> bool:
    if not out_path.exists():
        return False
    cols = pd.read_csv(out_path, nrows=0).columns
    return all(col in cols for col in DIVERSITY_SPLIT_COLS)


def run_decomposition_for_fold(
    variant_dir: Path,
    fold_tag: str,
    *,
    regime: str = "forward_repeat_sites",
    skip_if_exists: bool = True,
    require_diversity_split: bool = False,
) -> dict[str, Any] | None:
    """Run decomposition for one variant fold."""
    variant_dir = Path(variant_dir)
    fold_dir = variant_dir / "folds" / fold_tag
    pred_path = fold_dir / "predictions.csv"
    out_path = fold_dir / "predictions_decomposition.csv"

    if skip_if_exists and out_path.exists():
        if not require_diversity_split or _decomposition_has_diversity_split(out_path):
            return None
    if not pred_path.exists():
        raise FileNotFoundError(
            f"Missing fold predictions required for decomposition: {pred_path}"
        )

    predictions = pd.read_csv(pred_path)
    train_df, test_df = _load_fold_frames(fold_dir, fold_tag=fold_tag, regime=regime)
    test_df = test_df[test_df["row_id"].isin(predictions["row_id"])].copy()

    fit_stats_path = fold_dir / "fit_statistics.json"
    variant = fold_tag_name = None
    n_train = len(train_df)
    if fit_stats_path.exists():
        fit_stats = json.loads(fit_stats_path.read_text())
        variant = fit_stats.get("metrics", {}).get("variant")
        fold_tag_name = fit_stats.get("fold_tag")
        n_train = int(fit_stats.get("metrics", {}).get("n_train", n_train))

    summary = compute_fold_decomposition(
        predictions=predictions,
        test_df=test_df,
        train_df=train_df,
        fold_dir=fold_dir,
        variant=variant,
        n_train=n_train,
    )
    summary["fold_tag"] = fold_tag_name or fold_tag
    return summary


def ensure_waterfall_decomposition(
    beta_glmm_root: Path,
    *,
    variants: list[str],
    fold_tag: str,
    regime: str = "forward_repeat_sites",
    eco_variant: str = "reparam_ecoregion_only",
) -> None:
    """Run decomposition only for waterfall-required variants and one fold."""
    beta_glmm_root = Path(beta_glmm_root)
    summaries: list[dict[str, Any]] = []
    for variant in variants:
        summary = run_decomposition_for_fold(
            beta_glmm_root / variant,
            fold_tag,
            regime=regime,
            skip_if_exists=True,
            require_diversity_split=(variant == eco_variant),
        )
        out_path = (
            beta_glmm_root
            / variant
            / "folds"
            / fold_tag
            / "predictions_decomposition.csv"
        )
        ready = out_path.exists() and (
            variant != eco_variant or _decomposition_has_diversity_split(out_path)
        )
        if not ready:
            pred_path = (
                beta_glmm_root / variant / "folds" / fold_tag / "predictions.csv"
            )
            if not pred_path.exists():
                failures_path = beta_glmm_root / variant / "failures.csv"
                hint = f" See {failures_path}." if failures_path.exists() else ""
                raise FileNotFoundError(
                    f"Missing CV fold outputs for {variant}/{fold_tag}.{hint} "
                    "Re-run with --run-models."
                )
            raise RuntimeError(
                f"Hierarchy decomposition did not produce {out_path}. "
                "Check fold trace/diagnostics and re-run with --force-rerun."
            )

        if summary is not None:
            summaries.append(summary)
        else:
            json_path = (
                beta_glmm_root
                / variant
                / "folds"
                / fold_tag
                / "hierarchy_decomposition.json"
            )
            if json_path.exists():
                cached = json.loads(json_path.read_text())
                cached["fold_tag"] = fold_tag
                summaries.append(cached)

    upsert_hierarchy_decomposition_rows(beta_glmm_root, summaries)


def run_decomposition_for_variant(variant_dir: Path) -> list[dict[str, Any]]:
    """Run decomposition for every fold under a beta_glmm variant directory."""
    variant_dir = Path(variant_dir)
    summaries: list[dict[str, Any]] = []
    folds_root = variant_dir / "folds"
    if not folds_root.is_dir():
        return summaries

    for fold_dir in sorted(p for p in folds_root.iterdir() if p.is_dir()):
        fold_tag = fold_dir.name
        regime = fold_tag.rsplit("__", 1)[0]
        summary = run_decomposition_for_fold(
            variant_dir,
            fold_tag,
            regime=regime,
            skip_if_exists=False,
        )
        if summary is not None:
            summaries.append(summary)

    write_variant_decomposition_summary(summaries, variant_dir)
    return summaries


def run_decomposition_for_beta_glmm_root(beta_glmm_root: Path) -> pd.DataFrame:
    """Run decomposition across all variant subdirectories."""
    beta_glmm_root = Path(beta_glmm_root)
    all_rows: list[dict[str, Any]] = []

    variant_dirs = [
        p for p in beta_glmm_root.iterdir() if p.is_dir() and (p / "folds").is_dir()
    ]
    if not variant_dirs and (beta_glmm_root / "folds").is_dir():
        variant_dirs = [beta_glmm_root]

    for vdir in variant_dirs:
        for summary in run_decomposition_for_variant(vdir):
            all_rows.append(summary)

    if not all_rows:
        return pd.DataFrame()

    combined = pd.DataFrame([flatten_decomposition_summary(s) for s in all_rows])
    combined.to_csv(
        beta_glmm_root / "hierarchy_decomposition_all_variants.csv", index=False
    )
    return combined
