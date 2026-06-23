"""Data preparation helpers for the paper-vs-reparam investigation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit as inv_logit

from src.models.hbb._config import VARS_TO_STANDARDIZE
from src.models.hbb.variants import DIVERSITY_ECOREGION_ALIASES

def add_latitude_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add absolute and trigonometric latitude encodings before standardization."""
    out = df.copy()
    if "latitude.degrees" in out.columns:
        signed_latitude = out["latitude.degrees"].astype(float)
    else:
        signed_latitude = out["lat"].astype(float)
    out["lat_signed_degrees"] = signed_latitude
    out["lat"] = signed_latitude.abs()
    radians = np.deg2rad(signed_latitude)
    out["lat_sin"] = np.sin(radians)
    out["lat_cos"] = np.cos(radians)
    return out


def standardization_vars() -> list[str]:
    vars_to_standardize = list(VARS_TO_STANDARDIZE)
    for name in ("lat_sin", "lat_cos"):
        if name not in vars_to_standardize:
            vars_to_standardize.append(name)
    return vars_to_standardize


def normalize_ecoregion_name(value: Any) -> str:
    import re

    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def ecoregion_column(df: pd.DataFrame) -> str:
    for name in ("ecoregion", "Ecoregion", "ERName"):
        if name in df.columns:
            return name
    raise ValueError("Could not find an ecoregion column.")


def replace_diversity_from_ecoregions(
    df: pd.DataFrame,
    *,
    path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replace model diversity with standardized species counts from ecoregions.csv."""
    eco = pd.read_csv(path)
    required = {"ecoregion_name", "total_species_number"}
    missing = required - set(eco.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")

    out = df.copy()
    eco = eco.copy()
    eco["total_species_number"] = pd.to_numeric(
        eco["total_species_number"], errors="coerce"
    )
    mean = eco["total_species_number"].mean(skipna=True)
    sd = eco["total_species_number"].std(skipna=True)
    eco["ecoregions_diversity_standardized"] = (
        eco["total_species_number"] - mean
    ) / sd
    eco["_norm_name"] = eco["ecoregion_name"].map(normalize_ecoregion_name)
    lookup = eco.dropna(subset=["total_species_number"]).drop_duplicates(
        "_norm_name", keep="first"
    )

    eco_col = ecoregion_column(out)
    mapping_rows: list[dict[str, Any]] = []
    values: list[float] = []
    for model_name in out[eco_col].astype(str):
        target_name = DIVERSITY_ECOREGION_ALIASES.get(model_name, model_name)
        norm_name = normalize_ecoregion_name(target_name)
        match = lookup.loc[lookup["_norm_name"] == norm_name]
        if match.empty:
            values.append(np.nan)
            matched_name = None
            raw_species = np.nan
        else:
            row = match.iloc[0]
            values.append(float(row["ecoregions_diversity_standardized"]))
            matched_name = str(row["ecoregion_name"])
            raw_species = float(row["total_species_number"])
        mapping_rows.append(
            {
                "model_ecoregion": model_name,
                "mapped_ecoregions_name": matched_name,
                "aliased_to": target_name if target_name != model_name else "",
                "total_species_number": raw_species,
            }
        )

    out["diversity.standardized"] = values
    mapping = (
        pd.DataFrame(mapping_rows)
        .drop_duplicates("model_ecoregion")
        .sort_values("model_ecoregion")
        .reset_index(drop=True)
    )
    unmatched = mapping.loc[mapping["mapped_ecoregions_name"].isna(), "model_ecoregion"]
    if not unmatched.empty:
        raise ValueError(
            "Could not map ecoregions.csv diversity for: "
            + ", ".join(unmatched.astype(str))
        )
    return out, mapping


@dataclass(frozen=True)
class CoverSimConfig:
    """Stochastic cover simulation with a strong mean cosine-latitude gradient."""

    intercept: float = 0.0
    lat_strength: float = 2.0
    precision: float = 50.0
    site_logit_sd: float = 0.15
    obs_logit_sd: float = 0.15
    seed: int = 42
    standardize_cos_lat: bool = True


def _signed_latitude_degrees(df: pd.DataFrame) -> np.ndarray:
    if "lat_signed_degrees" in df.columns:
        return df["lat_signed_degrees"].astype(float).to_numpy()
    if "latitude.degrees" in df.columns:
        return df["latitude.degrees"].astype(float).to_numpy()
    return df["lat"].astype(float).to_numpy()


def simulate_cover(
    df: pd.DataFrame,
    *,
    mode: str,
    config: CoverSimConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    """Simulate coral cover with a cosine-latitude mean and beta observation noise.

    The legacy deterministic ``cos(lat)`` response saturated the beta likelihood
    and let site intercepts absorb the entire latitudinal gradient.  This version
    keeps a strong average ``cos(lat)`` relationship on the logit scale while
    adding modest site- and observation-level noise so NUTS can identify fixed
    effects and variance components.
    """
    if mode == "observed":
        return df, None
    if mode != "cosine_latitude":
        raise ValueError(f"Unknown cover simulation: {mode}")

    cfg = config or CoverSimConfig()
    out = df.copy()
    signed_latitude = _signed_latitude_degrees(out)
    cos_lat = np.cos(np.deg2rad(signed_latitude))
    cos_lat = np.clip(cos_lat, 1e-6, 1.0 - 1e-6)
    cos_lat_predictor = cos_lat.copy()
    if cfg.standardize_cos_lat:
        cos_lat_predictor = (cos_lat - cos_lat.mean()) / cos_lat.std()

    rng = np.random.default_rng(cfg.seed)
    n = len(out)
    eta = cfg.intercept + cfg.lat_strength * cos_lat_predictor

    if "site" in out.columns:
        site_codes, _ = pd.factorize(out["site"], sort=True)
        site_offsets = rng.normal(0.0, cfg.site_logit_sd, int(site_codes.max()) + 1)
        eta = eta + site_offsets[site_codes]
    else:
        site_codes = None

    eta = eta + rng.normal(0.0, cfg.obs_logit_sd, n)
    pi = np.clip(inv_logit(eta), 1e-4, 1.0 - 1e-4)
    alpha = cfg.precision * pi
    beta = cfg.precision * (1.0 - pi)
    simulated = rng.beta(alpha, beta)
    simulated = np.clip(simulated, 1e-4, 1.0 - 1e-4)

    out["observed_average_coral_cover"] = out["average_coral_cover"]
    out["average_coral_cover"] = simulated
    out["sim_cos_lat"] = cos_lat
    out["sim_cos_lat_predictor"] = cos_lat_predictor
    out["sim_logit_mean"] = eta

    site_means = None
    if site_codes is not None:
        site_df = pd.DataFrame(
            {
                "site": site_codes,
                "cos_lat": cos_lat,
                "cover": simulated,
            }
        )
        site_means = site_df.groupby("site", as_index=False).mean(numeric_only=True)

    metadata: dict[str, Any] = {
        **asdict(cfg),
        "mode": mode,
        "description": (
            "logit_mean = intercept + lat_strength * cos_lat_predictor + site_offset + obs_noise; "
            "cos_lat_predictor is dataset-standardized cos(lat) when standardize_cos_lat=True; "
            "cover ~ Beta(precision * pi, precision * (1 - pi))"
        ),
        "n_obs": n,
        "n_sites": int(len(np.unique(site_codes))) if site_codes is not None else None,
        "cos_lat_range": [float(cos_lat.min()), float(cos_lat.max())],
        "simulated_cover_range": [float(simulated.min()), float(simulated.max())],
        "corr_cos_lat_obs": float(np.corrcoef(cos_lat, simulated)[0, 1]),
        "corr_cos_lat_predictor_obs": float(
            np.corrcoef(cos_lat_predictor, simulated)[0, 1]
        ),
        "corr_cos_lat_site_means": (
            float(np.corrcoef(site_means["cos_lat"], site_means["cover"])[0, 1])
            if site_means is not None
            else None
        ),
    }
    return out, metadata


def _binned_mean_curve(
    x: np.ndarray, y: np.ndarray, *, n_bins: int = 30
) -> tuple[np.ndarray, np.ndarray]:
    """Return bin centres and mean y for evenly spaced x bins."""
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    edges = np.linspace(x_sorted.min(), x_sorted.max(), n_bins + 1)
    centres: list[float] = []
    means: list[float] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (x_sorted >= lo) & (x_sorted <= hi if hi == edges[-1] else x_sorted < hi)
        if not mask.any():
            continue
        centres.append(float(np.mean(x_sorted[mask])))
        means.append(float(np.mean(y_sorted[mask])))
    return np.asarray(centres), np.asarray(means)


def plot_simulated_cover_vs_latitude(
    df: pd.DataFrame,
    path: Path,
    *,
    config: CoverSimConfig | None = None,
) -> None:
    """Plot simulated cover against latitude with the DGP mean overlay."""
    cfg = config or CoverSimConfig()
    signed_lat = _signed_latitude_degrees(df)
    abs_lat = np.abs(signed_lat)
    cover = df["average_coral_cover"].astype(float).to_numpy()

    cos_lat = np.clip(np.cos(np.deg2rad(signed_lat)), 1e-6, 1.0 - 1e-6)
    cos_pred = cos_lat.copy()
    if cfg.standardize_cos_lat:
        cos_pred = (cos_lat - cos_lat.mean()) / cos_lat.std()
    pi_mean = inv_logit(cfg.intercept + cfg.lat_strength * cos_pred)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)

    panels = (
        (axes[0], signed_lat, "Signed latitude (°)", "Cosine DGP: peak cover near equator"),
        (axes[1], abs_lat, "Absolute latitude (°)", "Model predictor: |lat| increases toward poles"),
    )
    for ax, x, xlabel, subtitle in panels:
        order = np.argsort(x)
        ax.scatter(
            x,
            cover,
            s=7,
            alpha=0.10,
            color="steelblue",
            edgecolors="none",
            rasterized=True,
            label="Simulated observations",
        )
        bin_x, bin_y = _binned_mean_curve(x, cover)
        ax.plot(
            bin_x,
            bin_y,
            color="black",
            lw=1.8,
            marker="o",
            ms=4,
            label="Binned mean cover",
        )
        ax.plot(
            x[order],
            pi_mean[order],
            color="crimson",
            lw=2.2,
            label="DGP mean (no site/obs noise)",
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Simulated coral cover")
        ax.set_title(subtitle, fontsize=10)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(
        "Simulated cover vs latitude "
        f"(lat_strength={cfg.lat_strength:g}, precision={cfg.precision:g})",
        y=1.08,
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_cover_simulation_diagnostics(
    df: pd.DataFrame,
    inv_dir: Path,
    *,
    config: CoverSimConfig,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Write simulation metadata and latitude diagnostic plots."""
    diag_dir = inv_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    plot_path = diag_dir / "cover_vs_latitude.png"
    plot_simulated_cover_vs_latitude(df, plot_path, config=config)
    metadata = dict(metadata)
    metadata["cover_vs_latitude_plot"] = str(plot_path)
    (inv_dir / "cover_simulation.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    return metadata

