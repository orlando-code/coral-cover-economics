"""Fit beta-GLMM variants and write investigation diagnostics."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.special import expit as inv_logit, logit

from src.models.hbb.analysis import save_in_sample_application_outputs
from src.models.hbb.model import HierarchicalBetaModel, resolve_pymc_ncores
from src.models.hbb.variant_data import build_variant_data
from src.models.hbb.variants import (
    COEF_LABELS,
    COEF_LABEL_BY_COLUMN,
    KEY_OTHER_PARAMS,
    VARIANTS,
    Variant,
    coefficient_labels,
)

def _posterior_array(idata: az.InferenceData, var_name: str) -> np.ndarray | None:
    if var_name not in idata.posterior:
        return None
    return idata.posterior[var_name].values


def _flat_samples(idata: az.InferenceData, var_name: str) -> np.ndarray | None:
    arr = _posterior_array(idata, var_name)
    if arr is None:
        return None
    if arr.ndim == 2:
        return arr.reshape(-1)
    return arr.reshape(-1, arr.shape[-1])


def coefficient_summary(
    idata: az.InferenceData,
    col_names: list[str],
    *,
    use_diversity: bool = True,
) -> pd.DataFrame:
    beta = _flat_samples(idata, "beta")
    if beta is None:
        raise ValueError("Trace missing beta.")
    beta_div = _flat_samples(idata, "beta_diversity") if use_diversity else None

    rows: list[dict[str, Any]] = []
    beta_labels = list(col_names)
    if beta_labels and beta_labels[0] == "Intercept":
        beta_for_plot = beta[:, 1:]
        labels = coefficient_labels(beta_labels[1:])
    else:
        beta_for_plot = beta
        labels = coefficient_labels(beta_labels)

    for label, samples in zip(labels, beta_for_plot.T):
        rows.append(summary_row(label, samples))
    if beta_div is not None:
        if beta_div.ndim != 1:
            beta_div = beta_div.reshape(-1)
        rows.append(summary_row("Diversity", beta_div))
    return pd.DataFrame(rows)


def intercept_summary(
    idata: az.InferenceData, col_names: list[str]
) -> pd.DataFrame | None:
    beta = _flat_samples(idata, "beta")
    if beta is None or not col_names or col_names[0] != "Intercept":
        return None
    return pd.DataFrame([summary_row("Intercept", beta[:, 0])])


def summary_row(name: str, samples: np.ndarray) -> dict[str, Any]:
    return {
        "variable": name,
        "mean": float(np.mean(samples)),
        "sd": float(np.std(samples, ddof=1)),
        "lower_2.5": float(np.quantile(samples, 0.025)),
        "upper_97.5": float(np.quantile(samples, 0.975)),
        "lower_25": float(np.quantile(samples, 0.25)),
        "upper_75": float(np.quantile(samples, 0.75)),
    }


def logit_beta_to_delta_cover(beta: float, pi_ref: float = 0.3) -> float:
    """Map a logit-scale coefficient to Δcover at reference mean cover ``pi_ref``."""
    pi_ref = float(np.clip(pi_ref, 1e-4, 1.0 - 1e-4))
    return float(inv_logit(logit(pi_ref) + beta) - pi_ref)


def enrich_coefficients_delta_cover(
    beta_df: pd.DataFrame,
    pi_refs: tuple[float, ...] = (0.2, 0.3, 0.4),
) -> pd.DataFrame:
    """Add Δcover columns (and transformed interval bounds) for gamma coefficients."""
    out = beta_df.copy()
    for pi in pi_refs:
        label = f"delta_cover_pi{pi:.1f}"
        out[label] = out["mean"].map(lambda b: logit_beta_to_delta_cover(b, pi))
        out[f"{label}_lo"] = out["lower_2.5"].map(lambda b: logit_beta_to_delta_cover(b, pi))
        out[f"{label}_hi"] = out["upper_97.5"].map(lambda b: logit_beta_to_delta_cover(b, pi))
    return out


def filter_df_by_abs_lat_cap(df: pd.DataFrame, cap: float | None) -> pd.DataFrame:
    """Keep rows with absolute latitude (degrees) <= ``cap``; ``None`` keeps all rows."""
    if cap is None:
        return df
    if "lat" in df.columns:
        abs_lat = df["lat"].astype(float)
    elif "lat_signed_degrees" in df.columns:
        abs_lat = df["lat_signed_degrees"].astype(float).abs()
    elif "latitude.degrees" in df.columns:
        abs_lat = df["latitude.degrees"].astype(float).abs()
    else:
        raise ValueError("Could not find a latitude column for abs-lat filtering.")
    return df.loc[abs_lat <= float(cap)].copy()


def run_latitude_cap_sensitivity(
    df_std: pd.DataFrame,
    *,
    caps: Iterable[float | None] = (None, 30, 25, 20, 15, 10),
    variant: Variant | None = None,
    draws: int = 400,
    tune: int = 400,
    chains: int = 2,
    ncores: int = 1,
    target_accept: float = 0.95,
    max_treedepth: int = 15,
    seed: int = 42,
    progressbar: bool = False,
) -> pd.DataFrame:
    """Refit ``reparam`` while progressively excluding high-|latitude| observations."""
    import tempfile

    variant = variant or VARIANTS["reparam"]
    rows: list[dict[str, Any]] = []
    for i, cap in enumerate(caps):
        subset = filter_df_by_abs_lat_cap(df_std, cap)
        with tempfile.TemporaryDirectory(prefix="lat_cap_") as tmp:
            beta_df, _, summary = fit_variant(
                variant=variant,
                df_std=subset,
                output_dir=Path(tmp),
                draws=draws,
                tune=tune,
                chains=chains,
                ncores=ncores,
                target_accept=target_accept,
                max_treedepth=max_treedepth,
                seed=seed + i,
                progressbar=progressbar,
            )
        enriched = enrich_coefficients_delta_cover(beta_df)
        for _, row in enriched.iterrows():
            rows.append(
                {
                    "abs_lat_cap": np.nan if cap is None else float(cap),
                    "cap_label": "full" if cap is None else f"<= {int(cap)}°",
                    "n_obs": summary["N"],
                    "n_sites": summary["Nre"],
                    "n_regions": summary["R"],
                    **row.to_dict(),
                }
            )
    return pd.DataFrame(rows)


def ecoregion_predictor_contributions(
    idata: az.InferenceData,
    *,
    col_names: list[str],
    diversity: np.ndarray,
    region_predictors: pd.DataFrame,
    pi_ref: float = 0.3,
) -> pd.DataFrame:
    """Per-ecoregion logit contributions for diversity, latitude, and historical SST."""
    beta = _flat_samples(idata, "beta")
    beta_div = _flat_samples(idata, "beta_diversity")
    mu_global = _flat_samples(idata, "mu_global")
    ecoregion = _flat_samples(idata, "ecoregion")
    if beta is None or beta_div is None or mu_global is None or ecoregion is None:
        raise ValueError("Trace missing beta, beta_diversity, mu_global, or ecoregion.")

    label_to_col = {v: k for k, v in COEF_LABEL_BY_COLUMN.items()}
    wanted = {
        "Latitude": label_to_col["Latitude"],
        "Historical_SST_max": label_to_col["Historical_SST_max"],
    }
    beta_mean = beta.mean(axis=0)
    beta_div_mean = float(beta_div.mean())
    mu_global_mean = float(mu_global.mean())
    ecoregion_mean = ecoregion.mean(axis=0)

    rows: list[dict[str, Any]] = []
    for r in range(len(diversity)):
        reg_row = region_predictors.loc[region_predictors["region_idx"] == r].iloc[0]
        div_logit = beta_div_mean * float(diversity[r])
        g_logit = mu_global_mean + div_logit
        random_residual = float(ecoregion_mean[r] - g_logit)
        lat_logit = float(beta_mean[col_names.index(wanted["Latitude"])] * reg_row["lat_stzd"])
        hist_logit = float(
            beta_mean[col_names.index(wanted["Historical_SST_max"])] * reg_row["historical_sst_max_stzd"]
        )
        rows.append(
            {
                "region_idx": r,
                "ecoregion": reg_row.get("ecoregion", r),
                "n_sites": int(reg_row["n_sites"]),
                "mean_abs_lat": float(reg_row["abs_lat"]),
                "diversity": float(diversity[r]),
                "diversity_logit": div_logit,
                "latitude_logit": lat_logit,
                "historical_sst_max_logit": hist_logit,
                "ecoregion_random_residual_logit": random_residual,
                "diversity_delta_cover": logit_beta_to_delta_cover(div_logit, pi_ref),
                "latitude_delta_cover": logit_beta_to_delta_cover(lat_logit, pi_ref),
                "historical_sst_max_delta_cover": logit_beta_to_delta_cover(hist_logit, pi_ref),
                "ecoregion_random_residual_delta_cover": logit_beta_to_delta_cover(
                    random_residual, pi_ref
                ),
            }
        )
    return pd.DataFrame(rows)


def convergence_summary(
    idata: az.InferenceData, var_names: Iterable[str]
) -> pd.DataFrame:
    available = [v for v in var_names if v in idata.posterior]
    summary = az.summary(idata, var_names=available, hdi_prob=0.95)
    summary = summary.rename(columns={"ess_bulk": "n.eff"})
    cols = [
        c
        for c in ["mean", "sd", "hdi_2.5%", "hdi_97.5%", "r_hat", "n.eff"]
        if c in summary
    ]
    return summary[cols]


def plot_coefficients(beta_df: pd.DataFrame, path: Path, title: str) -> None:
    from src.plots.hb_beta_plots import plot_coefficient_forest_df

    plot_coefficient_forest_df(
        beta_df,
        path,
        title=title,
        label_col="variable",
    )


def draws_dataframe(
    idata: az.InferenceData, var_names: list[str], col_names: list[str]
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    posterior = idata.posterior
    for var in var_names:
        if var == "beta" and "beta" in posterior:
            vals = posterior["beta"].values  # chain, draw, beta_dim
            for idx, label in enumerate(col_names):
                pieces.append(
                    pd.DataFrame(
                        {
                            "chain": np.repeat(np.arange(vals.shape[0]), vals.shape[1]),
                            "draw": np.tile(np.arange(vals.shape[1]), vals.shape[0]),
                            "param": label,
                            "value": vals[:, :, idx].reshape(-1),
                        }
                    )
                )
        elif var in posterior:
            vals = posterior[var].values
            if vals.ndim == 2:
                pieces.append(
                    pd.DataFrame(
                        {
                            "chain": np.repeat(np.arange(vals.shape[0]), vals.shape[1]),
                            "draw": np.tile(np.arange(vals.shape[1]), vals.shape[0]),
                            "param": var,
                            "value": vals.reshape(-1),
                        }
                    )
                )
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def save_trace_nc(idata: az.InferenceData, path: Path) -> None:
    """Save ArviZ InferenceData and verify a non-empty posterior round-trips."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not idata.posterior.data_vars:
        raise ValueError(f"Refusing to save empty posterior to {path}")
    idata.to_netcdf(path)
    loaded = az.from_netcdf(path)
    if not loaded.posterior.data_vars:
        raise RuntimeError(f"trace.nc at {path} failed post-save validation.")


def load_trace_nc(path: Path) -> az.InferenceData:
    """Load ``trace.nc`` written by :func:`save_trace_nc`.

    ArviZ stores InferenceData as a multi-group NetCDF file. Use this helper
    (or ``arviz.from_netcdf``) — ``xarray.open_dataset`` only opens the empty
    root group and will look like a 0-byte dataset.
    """
    path = Path(path)
    idata = az.from_netcdf(path)
    if not idata.posterior.data_vars:
        raise ValueError(
            f"No posterior variables in {path}. "
            "Load with arviz.from_netcdf(), not xarray.open_dataset()."
        )
    return idata


def save_facet_plot(df: pd.DataFrame, path: Path, kind: str, title: str) -> None:
    if df.empty:
        return
    n_params = df["param"].nunique()
    height = max(3.0, 2.0 * np.ceil(n_params / 4))
    if kind == "trace":
        g = sns.relplot(
            df,
            x="draw",
            y="value",
            hue="chain",
            col="param",
            col_wrap=4,
            kind="line",
            facet_kws={"sharey": False, "sharex": True},
            height=2.0,
            aspect=1.35,
            linewidth=0.6,
        )
    elif kind == "density":
        g = sns.displot(
            df,
            x="value",
            hue="chain",
            col="param",
            col_wrap=4,
            kind="kde",
            facet_kws={"sharex": False, "sharey": False},
            height=2.0,
            aspect=1.35,
        )
    else:
        raise ValueError(kind)
    g.figure.set_size_inches(10.8, height)
    g.figure.suptitle(title, y=1.02)
    g.figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(g.figure)


def rhat_plot_frame(
    conv: pd.DataFrame,
    col_names: list[str] | None = None,
) -> pd.DataFrame:
    """Build a readable R-hat summary: fixed effects individually, RE groups aggregated."""
    import re

    conv = conv.copy()
    conv.index = conv.index.astype(str)
    if "r_hat" not in conv.columns:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    beta_pattern = re.compile(r"^beta\[(\d+)\]$")
    beta_items = sorted(
        (int(m.group(1)), name)
        for name in conv.index
        if (m := beta_pattern.match(name))
    )

    labels = list(col_names or [])
    if labels and labels[0] == "Intercept":
        if any(i == 0 for i, _ in beta_items):
            rows.append(
                {
                    "param": "Intercept",
                    "r_hat": float(conv.loc["beta[0]", "r_hat"]),
                    "r_hat_std": np.nan,
                    "kind": "fixed",
                }
            )
        env_labels = coefficient_labels(labels[1:])
        beta_items = [(i, name) for i, name in beta_items if i > 0]
    else:
        env_labels = coefficient_labels(labels) if labels else []

    for j, (i, name) in enumerate(beta_items):
        label = env_labels[j] if j < len(env_labels) else name
        rows.append(
            {
                "param": label,
                "r_hat": float(conv.loc[name, "r_hat"]),
                "r_hat_std": np.nan,
                "kind": "fixed",
            }
        )

    scalar_labels = {
        "beta_diversity": "Diversity",
        "mu_global": "mu_global",
        "sigma": "sigma",
        "sigma_site": "sigma_site",
        "sigma_ecoregion": "sigma_ecoregion",
        "theta": "theta",
    }
    for var in KEY_OTHER_PARAMS:
        if var in conv.index:
            rows.append(
                {
                    "param": scalar_labels.get(var, var),
                    "r_hat": float(conv.loc[var, "r_hat"]),
                    "r_hat_std": np.nan,
                    "kind": "hyper",
                }
            )

    for group, prefix in [("ecoregion", "ecoregion["), ("site_effect", "site_effect[")]:
        sub = conv[conv.index.str.startswith(prefix)]
        if sub.empty:
            continue
        n = len(sub)
        rows.append(
            {
                "param": f"{group} (n={n})",
                "r_hat": float(sub["r_hat"].mean()),
                "r_hat_std": float(sub["r_hat"].std(ddof=1)) if n > 1 else 0.0,
                "kind": "random",
            }
        )

    return pd.DataFrame(rows)


def save_rhat_plot(
    conv: pd.DataFrame,
    path: Path,
    title: str,
    *,
    col_names: list[str] | None = None,
) -> None:
    df = rhat_plot_frame(conv, col_names)
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(df))))
    y = np.arange(len(df))
    colours = np.full(len(df), "steelblue", dtype=object)
    colours[df["kind"] == "hyper"] = "teal"
    colours[df["kind"] == "random"] = "mediumpurple"
    colours[(df["r_hat"] > 1.05) & (df["r_hat"] <= 1.10)] = "orange"
    colours[df["r_hat"] > 1.10] = "red"

    ax.barh(y, df["r_hat"], color=colours, height=0.55, alpha=0.9, zorder=2)
    err_mask = df["r_hat_std"].notna() & (df["r_hat_std"] > 0)
    if err_mask.any():
        ax.errorbar(
            df.loc[err_mask, "r_hat"],
            y[err_mask.to_numpy()],
            xerr=df.loc[err_mask, "r_hat_std"],
            fmt="none",
            ecolor="black",
            capsize=3,
            linewidth=1.2,
            zorder=3,
        )

    ax.axvline(1.0, color="gray", linestyle="-", linewidth=0.8)
    ax.axvline(1.05, color="orange", linestyle="--")
    ax.axvline(1.10, color="red", linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(df["param"])
    ax.set_xlabel("R-hat (bars = fixed/hyper; purple = mean ± SD over RE levels)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_variant_outputs(
    *,
    idata: az.InferenceData,
    variant: Variant,
    output_dir: Path,
    col_names: list[str],
    input_summary: dict[str, Any],
    elapsed_s: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    diag_dir = output_dir / "diagnostics"
    log_dir = output_dir / "logs"
    diag_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    save_trace_nc(idata, output_dir / "trace.nc")
    (output_dir / "model_spec.json").write_text(
        json.dumps(asdict(variant), indent=2) + "\n"
    )
    pd.DataFrame([input_summary]).to_csv(
        output_dir / "model_input_summary.csv", index=False
    )

    beta_df = coefficient_summary(
        idata, col_names, use_diversity=variant.use_diversity
    )
    beta_df.to_csv(output_dir / "beta_est.csv", index=False)
    intercept_df = intercept_summary(idata, col_names)
    if intercept_df is not None:
        intercept_df.to_csv(output_dir / "intercept_beta.csv", index=False)

    other_params = [
        p
        for p in KEY_OTHER_PARAMS
        if p != "beta_diversity" or variant.use_diversity
    ]
    re_params: list[str] = []
    if variant.use_ecoregion_hierarchy:
        re_params.append("ecoregion")
    if variant.use_site_hierarchy:
        re_params.append("site_effect")
    conv = convergence_summary(idata, ["beta", *other_params, *re_params])
    conv.to_csv(output_dir / "convergence_diagnostics.csv")
    conv.to_csv(diag_dir / "convergence_full.csv")

    plot_coefficients(
        beta_df, diag_dir / "coeff_forest.png", f"Beta coefficients: {variant.name}"
    )
    beta_draws = draws_dataframe(idata, ["beta"], col_names)
    other_draws = draws_dataframe(idata, other_params, col_names)
    save_facet_plot(
        beta_draws,
        diag_dir / "trace_betas.png",
        "trace",
        f"Beta traces: {variant.name}",
    )
    save_facet_plot(
        other_draws,
        diag_dir / "trace_other.png",
        "trace",
        f"Other traces: {variant.name}",
    )
    save_facet_plot(
        beta_draws,
        diag_dir / "density_betas.png",
        "density",
        f"Beta densities: {variant.name}",
    )
    save_facet_plot(
        other_draws,
        diag_dir / "density_other.png",
        "density",
        f"Other densities: {variant.name}",
    )
    save_rhat_plot(
        conv, diag_dir / "rhat_summary.png", f"R-hat: {variant.name}", col_names=col_names
    )

    rhat = conv["r_hat"] if "r_hat" in conv else pd.Series(dtype=float)
    neff = conv["n.eff"] if "n.eff" in conv else pd.Series(dtype=float)
    run_log = {
        "variant": variant.name,
        "elapsed_seconds": elapsed_s,
        "max_rhat": float(rhat.max(skipna=True)) if not rhat.empty else None,
        "n_rhat_gt_1.05": int((rhat > 1.05).sum()) if not rhat.empty else None,
        "n_rhat_gt_1.10": int((rhat > 1.10).sum()) if not rhat.empty else None,
        "min_neff": float(neff.min(skipna=True)) if not neff.empty else None,
        "median_neff": float(neff.median(skipna=True)) if not neff.empty else None,
    }
    (log_dir / "run_log.json").write_text(json.dumps(run_log, indent=2) + "\n")
    pd.Series(run_log).to_csv(log_dir / "run_log.txt", header=False)
    beta_df["variant"] = variant.name
    return beta_df, conv


def plot_comparison(combined: pd.DataFrame, path: Path) -> None:
    if combined["variant"].nunique() < 2:
        return
    preferred = [
        *COEF_LABELS,
        "sin(latitude)",
        "cos(latitude)",
        "Diversity",
    ]
    present = list(dict.fromkeys(combined["variable"].astype(str)))
    order = [v for v in preferred if v in present] + [
        v for v in present if v not in preferred
    ]
    df = combined.copy()
    df["variable"] = pd.Categorical(df["variable"], categories=order, ordered=True)
    df = df.sort_values("variable")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.pointplot(
        data=df,
        x="mean",
        y="variable",
        hue="variant",
        dodge=0.55,
        join=False,
        errorbar=None,
        ax=ax,
    )
    for _, row in df.iterrows():
        y = order.index(row["variable"])
        ax.plot(
            [row["lower_2.5"], row["upper_97.5"]], [y, y], color="black", alpha=0.25
        )
    ax.axvline(0, color="gray", linestyle="--")
    ax.set_xlabel("Estimated gamma coefficients")
    ax.set_ylabel("")
    ax.set_title("Beta coefficient comparison")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def fit_variant(
    *,
    variant: Variant,
    df_std: pd.DataFrame,
    output_dir: Path,
    draws: int,
    tune: int,
    chains: int,
    ncores: int,
    target_accept: float,
    max_treedepth: int,
    seed: int,
    progressbar: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    built = build_variant_data(df_std, variant)
    model = HierarchicalBetaModel()
    start = time.time()
    model.fit(
        built["X"],
        built["y"],
        built["site_idx"],
        built["region_idx"],
        built["site_to_region"],
        reef_to_site_map=built["reef_to_site_map"],
        diversity=built["diversity"],
        col_names=built["col_names"],
        n_samples=draws,
        n_tune=tune,
        n_chains=chains,
        target_accept=target_accept,
        max_treedepth=max_treedepth,
        random_seed=seed,
        spec=variant.spec,
        ncores=ncores,
        progressbar=progressbar,
        use_site_hierarchy=variant.use_site_hierarchy,
        use_ecoregion_hierarchy=variant.use_ecoregion_hierarchy,
        use_diversity=variant.use_diversity,
    )
    elapsed = time.time() - start
    beta_df, conv = save_variant_outputs(
        idata=model.trace,
        variant=variant,
        output_dir=output_dir,
        col_names=built["col_names"],
        input_summary=built["input_summary"],
        elapsed_s=elapsed,
    )
    save_in_sample_application_outputs(
        model=model,
        df=built["df"],
        X=built["X"],
        site_idx=built["site_idx"],
        output_dir=output_dir,
    )
    return beta_df, conv, built["input_summary"]
