"""Beta-GLMM variant definitions and parsing (investigation + cross-validation)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from src.models.hbb._config import CV_PREDICTORS, ModelSpec

COEF_LABELS = [
    "Latitude",
    "Depth",
    "Human_pop",
    "Cyclone",
    "SST_mean",
    "SSTA_Mean",
    "SSTA_min",
    "SSTA_freqstdev",
    "SSTA_dhwmax",
    "TSA_max",
    "TSA_freqstdev",
    "Turbidity_mean",
    "Historical_SST_max",
]

COEF_LABEL_BY_COLUMN = {
    "lat_stzd": "Latitude",
    "lat_sin_stzd": "sin(latitude)",
    "lat_cos_stzd": "cos(latitude)",
    "depth_stzd": "Depth",
    "human_pop_stzd": "Human_pop",
    "cyclone_stzd": "Cyclone",
    "sst_mean_stzd": "SST_mean",
    "ssta_mean_stzd": "SSTA_Mean",
    "ssta_min_stzd": "SSTA_min",
    "ssta_freqstdev_stzd": "SSTA_freqstdev",
    "ssta_dhwmax_stzd": "SSTA_dhwmax",
    "tsa_max_stzd": "TSA_max",
    "tsa_freqstdev_stzd": "TSA_freqstdev",
    "turbidity_mean_stzd": "Turbidity_mean",
    "historical_sst_max_stzd": "Historical_SST_max",
}

KEY_OTHER_PARAMS = [
    "beta_diversity",
    "mu_global",
    "sigma",
    "sigma_site",
    "sigma_ecoregion",
    "theta",
]

DIVERSITY_ECOREGION_ALIASES = {
    "Banda Sea and Molucca Islands": "Banda Sea and Moluccas",
    "Birds Head Peninsula, Papua": "Raja Ampat, Papua",
    "Central and northern Great Barrier Reef": "Great Barrier Reef north-central",
    "Colombia, Ecuador and Chile, Pacific coast": "Colombia and Ecuador, Pacific coast",
    "Cook Islands, south-west Pacific": "Cook Islands, central Pacific",
    "Eastern Hawaii": "Hawaii east",
    "Eastern coast South Africa": "South Africa east",
    "Gilbert Islands, west Kiribati": "Kiribati west, Gilbert Islands",
    "Gulf of Tomini, Indonesia": "Gulf of Tomini, Sulawesi",
    "Kenya and Tanzania coast": "Kenya and Tanzania",
    "Lakshadweep Islands": "Lakshadweep",
    "Makassar Strait, Indonesia": "Makassar Strait",
    "Maldive Islands": "Maldives",
    "Moreton Bay, eastern Australia": "Moreton Bay, east Australia",
    "North Madagascar": "Madagascar north",
    "North Mozambique coast": "Mozambique north",
    "North Myanmar and Bangladesh": "Myanmar north and Bangladesh",
    "North Philippines": "Philippines north",
    "North Ryukyu Islands, Japan": "Ryukyu Islands north",
    "North Sri Lanka and east India": "Sri Lanka north and India east",
    "North Vietnam": "Vietnam north",
    "North and central Red Sea": "Red Sea north-central",
    "Northern Seychelles": "Seychelles north",
    "South Java": "Java south",
    "South Madagascar": "Madagascar south",
    "South Mozambique coast": "Mozambique south",
    "South Red Sea": "Red Sea south",
    "South Ryukyu Islands, Japan": "Ryukyu Islands south",
    "South Vietnam": "Vietnam south",
    "South-east Philippines": "Philippines south-east",
    "Southern Great Barrier Reef": "Great Barrier Reef south",
    "Southern Seychelles": "Seychelles south",
    "Strait of Malacca": "Malacca Strait",
    "Sunda Shelf, south-east Asia": "Sunda Shelf",
    "West Sumatra": "Sumatra west",
    "Western Mexico and Revillagigedo Islands": "Mexico west and Revillagigedo Islands",
    "Western Tuamotu Archipelago, central Pacific": "Tuamotu Archipelago west, central Pacific",
}

BETA_VARIANT_LABELS: dict[str, str] = {
    "reparam": "Beta-GLMM (reparam)",
    "paper_reproduction": "Beta-GLMM (paper)",
    "paper_region_fixed": "Beta-GLMM (fixed index)",
    "reparam_centered": "Beta-GLMM (centered)",
    "original_no_intercept": "Beta-GLMM (no intercept)",
    "reparam_no_latitude": "Beta-GLMM (no lat)",
    "reparam_latitude_trig": "Beta-GLMM (trig lat)",
    "reparam_flat": "Beta-GLMM (flat)",
    "reparam_site_only": "Beta-GLMM (site only)",
    "reparam_ecoregion_only": "Beta-GLMM (ecoregion only)",
    "reparam_no_diversity": "Beta-GLMM (no diversity)",
}


@dataclass(frozen=True)
class Variant:
    name: str
    subdir: str
    spec: ModelSpec
    add_intercept: bool
    index_mode: str
    paper_factor_encoding: bool = False
    exclude_vars: tuple[str, ...] = ()
    latitude_transform: Literal["abs", "trig"] = "abs"
    use_site_hierarchy: bool = True
    use_ecoregion_hierarchy: bool = True
    use_diversity: bool = True

    @property
    def use_hierarchy(self) -> bool:
        """True when any random-intercept hierarchy is enabled."""
        return self.use_site_hierarchy or self.use_ecoregion_hierarchy


VARIANTS: dict[str, Variant] = {
    "original_no_intercept": Variant(
        name="original_no_intercept",
        subdir="03_no_intercept_centered",
        spec="centered",
        add_intercept=False,
        index_mode="reparam",
    ),
    "paper_reproduction": Variant(
        name="paper_reproduction",
        subdir="01_paper_reproduction",
        spec="legacy_r",
        add_intercept=True,
        index_mode="legacy_r",
        paper_factor_encoding=True,
    ),
    "paper_region_fixed": Variant(
        name="paper_region_fixed",
        subdir="02_paper_region_fixed",
        spec="legacy_r",
        add_intercept=True,
        index_mode="reparam",
    ),
    "reparam_centered": Variant(
        name="reparam_centered",
        subdir="04_reparam_centered",
        spec="centered",
        add_intercept=False,
        index_mode="reparam",
    ),
    "reparam": Variant(
        name="reparam",
        subdir="05_reparam_noncentered",
        spec="reparam",
        add_intercept=False,
        index_mode="reparam",
    ),
    "reparam_no_latitude": Variant(
        name="reparam_no_latitude",
        subdir="06_reparam_no_latitude",
        spec="reparam",
        add_intercept=False,
        index_mode="reparam",
        exclude_vars=("lat",),
    ),
    "reparam_latitude_trig": Variant(
        name="reparam_latitude_trig",
        subdir="07_reparam_latitude_trig",
        spec="reparam",
        add_intercept=False,
        index_mode="reparam",
        latitude_transform="trig",
    ),
    "reparam_flat": Variant(
        name="reparam_flat",
        subdir="08_reparam_flat",
        spec="reparam",
        add_intercept=True,
        index_mode="reparam",
        use_site_hierarchy=False,
        use_ecoregion_hierarchy=False,
        use_diversity=False,
    ),
    "reparam_site_only": Variant(
        name="reparam_site_only",
        subdir="10_reparam_site_only",
        spec="reparam",
        add_intercept=False,
        index_mode="reparam",
        use_site_hierarchy=True,
        use_ecoregion_hierarchy=False,
        use_diversity=False,
    ),
    "reparam_ecoregion_only": Variant(
        name="reparam_ecoregion_only",
        subdir="11_reparam_ecoregion_only",
        spec="reparam",
        add_intercept=False,
        index_mode="reparam",
        use_site_hierarchy=False,
        use_ecoregion_hierarchy=True,
        use_diversity=True,
    ),
    "reparam_no_diversity": Variant(
        name="reparam_no_diversity",
        subdir="09_reparam_no_diversity",
        spec="reparam",
        add_intercept=False,
        index_mode="reparam",
        use_site_hierarchy=True,
        use_ecoregion_hierarchy=True,
        use_diversity=False,
    ),
}

FULL_INVESTIGATION_VARIANTS = (
    "paper_reproduction",
    "paper_region_fixed",
    "original_no_intercept",
    "reparam_centered",
    "reparam",
)


def parse_csv_list(value: str | None) -> list[str]:
    """Parse comma-separated CLI tokens."""
    value = (value or "").strip()
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


def parse_variant_names(text: str | None, *, default: list[str] | None = None) -> list[str]:
    """Parse comma-separated variant names; validate against :data:`VARIANTS`."""
    names = parse_csv_list(text) if text else list(default or ["reparam"])
    if not names:
        raise ValueError("At least one beta variant is required.")
    bad = [n for n in names if n not in VARIANTS]
    if bad:
        opts = ", ".join(sorted(VARIANTS))
        raise ValueError(
            f"Unknown beta variant(s): {', '.join(bad)}. Expected one of: {opts}"
        )
    return names


def parse_variants(text: str | None, *, default: list[Variant]) -> list[Variant]:
    """Parse investigation variant selection (supports ``all`` / ``full`` aliases)."""
    if not text:
        return default
    requested = parse_csv_list(text)
    if len(requested) == 1 and requested[0] in {"all", "full"}:
        requested = list(FULL_INVESTIGATION_VARIANTS)
    elif len(requested) == 1 and requested[0] in {"available", "list"}:
        raise ValueError(f"Available variants: {', '.join(VARIANTS)}")
    requested = [
        "paper_reproduction" if v == "paper_faithful" else v for v in requested
    ]
    names = parse_variant_names(",".join(requested))
    return [VARIANTS[name] for name in names]


def parse_excluded_vars(text: str | None) -> tuple[str, ...]:
    if not text:
        return ()
    return tuple(v.strip() for v in text.split(",") if v.strip())


def normalize_excluded_var(name: str) -> str:
    key = name.strip().lower()
    aliases = {
        "latitude": "lat",
        "absolute_latitude": "lat",
        "abs_latitude": "lat",
        "lat_stzd": "lat",
        "sin_latitude": "lat_sin",
        "lat_sin_stzd": "lat_sin",
        "cos_latitude": "lat_cos",
        "lat_cos_stzd": "lat_cos",
    }
    if key.endswith("_stzd"):
        key = key.removesuffix("_stzd")
    return aliases.get(key, key)


def exclusion_to_columns(name: str) -> set[str]:
    key = normalize_excluded_var(name)
    if key == "lat":
        return {"lat_stzd", "lat_sin_stzd", "lat_cos_stzd"}
    return {f"{key}_stzd"}


def parse_optional_bool(value: str | None) -> bool | None:
    """Parse CLI ``true``/``false`` tokens; ``None`` means unset."""
    if value is None:
        return None
    key = value.strip().lower()
    if key in {"", "none"}:
        return None
    if key in {"1", "true", "yes", "on"}:
        return True
    if key in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Expected true/false, got {value!r}")


def apply_variant_options(
    variants: list[Variant],
    *,
    exclude_vars: tuple[str, ...],
    latitude_transform: Literal["abs", "trig"] | None,
    use_site_hierarchy: bool | None = None,
    use_ecoregion_hierarchy: bool | None = None,
) -> list[Variant]:
    if (
        not exclude_vars
        and latitude_transform is None
        and use_site_hierarchy is None
        and use_ecoregion_hierarchy is None
    ):
        return variants
    out: list[Variant] = []
    for variant in variants:
        merged_exclusions = tuple(dict.fromkeys((*variant.exclude_vars, *exclude_vars)))
        next_variant = variant
        if exclude_vars:
            suffix = "_exclude_" + "_".join(normalize_excluded_var(v) for v in exclude_vars)
            next_variant = replace(
                next_variant,
                name=f"{next_variant.name}{suffix}",
                subdir=f"{next_variant.subdir}{suffix}",
                exclude_vars=merged_exclusions,
            )
        else:
            next_variant = replace(next_variant, exclude_vars=merged_exclusions)
        if latitude_transform is not None:
            suffix = f"_lat_{latitude_transform}"
            next_variant = replace(
                next_variant,
                name=f"{next_variant.name}{suffix}",
                subdir=f"{next_variant.subdir}{suffix}",
                latitude_transform=latitude_transform,
            )
        site_h = (
            use_site_hierarchy
            if use_site_hierarchy is not None
            else next_variant.use_site_hierarchy
        )
        eco_h = (
            use_ecoregion_hierarchy
            if use_ecoregion_hierarchy is not None
            else next_variant.use_ecoregion_hierarchy
        )
        if (
            use_site_hierarchy is not None
            or use_ecoregion_hierarchy is not None
        ) and (site_h != next_variant.use_site_hierarchy or eco_h != next_variant.use_ecoregion_hierarchy):
            suffix = f"_site_{site_h}_eco_{eco_h}"
            next_variant = replace(
                next_variant,
                name=f"{next_variant.name}{suffix}",
                subdir=f"{next_variant.subdir}{suffix}",
                use_site_hierarchy=site_h,
                use_ecoregion_hierarchy=eco_h,
            )
        out.append(next_variant)
    return out


def unique_variants(variants: list[Variant]) -> list[Variant]:
    seen: set[tuple[str, str]] = set()
    out: list[Variant] = []
    for variant in variants:
        key = (variant.name, variant.subdir)
        if key in seen:
            continue
        seen.add(key)
        out.append(variant)
    return out


def coefficient_labels(col_names: list[str]) -> list[str]:
    return [COEF_LABEL_BY_COLUMN.get(name, name.removesuffix("_stzd")) for name in col_names]


def display_coefficient_label(name: str) -> str:
    """Map design-column, summary-table, or trace names to verbose plot labels."""
    from src.plots.plot_config import COVARIATE_LABELS

    key = str(name)
    if key in COVARIATE_LABELS:
        return COVARIATE_LABELS[key]
    if key in COEF_LABEL_BY_COLUMN:
        col = key
        return COVARIATE_LABELS.get(col, COEF_LABEL_BY_COLUMN[col])
    for col, short in COEF_LABEL_BY_COLUMN.items():
        if short == key:
            return COVARIATE_LABELS.get(col, short)
    if key in {"Diversity", "beta_diversity"}:
        return COVARIATE_LABELS.get("beta_diversity", "Beta diversity")
    if key == "Intercept":
        return "Intercept"
    return key


def predictors_for_variant(work, variant: Variant) -> list[str]:
    import pandas as pd

    if not isinstance(work, pd.DataFrame):
        raise TypeError("work must be a pandas DataFrame")
    predictors = [p for p in CV_PREDICTORS if p in work.columns]
    if variant.latitude_transform == "trig":
        predictors = [p for p in predictors if p != "lat_stzd"]
        for name in ("lat_sin_stzd", "lat_cos_stzd"):
            if name in work.columns and name not in predictors:
                predictors.append(name)
    excluded_columns: set[str] = set()
    for name in variant.exclude_vars:
        excluded_columns |= exclusion_to_columns(name)
    return [p for p in predictors if p not in excluded_columns]


def variant_plot_title(name: str, *, short: bool = False) -> str:
    """Human-readable beta variant label for plot titles / axis labels."""
    v = VARIANTS.get(name)
    if v is None:
        return BETA_VARIANT_LABELS.get(name, name)

    idx = v.index_mode
    if v.paper_factor_encoding:
        idx = f"{idx}, lexicographic factor"

    intercept = "intercept" if v.add_intercept else "no intercept"
    if short:
        parts = [f"β-GLMM {name}", f"({v.spec}, {intercept})"]
        if not v.use_hierarchy:
            parts[0] = f"β-regression {name}"
        return "\n".join(parts)

    if not v.use_hierarchy:
        hierarchy = "fixed effects only"
    elif v.use_site_hierarchy and v.use_ecoregion_hierarchy:
        hierarchy = (
            "ecoregion+site, no diversity" if not v.use_diversity else "full hierarchy"
        )
    elif v.use_site_hierarchy:
        hierarchy = "site only"
    else:
        hierarchy = (
            "ecoregion only, no diversity"
            if not v.use_diversity
            else "ecoregion only"
        )
    return (
        f"Beta-GLMM · {name}\n"
        f"{v.spec} · {idx} indexing · {intercept} · {hierarchy}"
    )


def beta_variant_output_dir(output_dir, variant: str | Variant, n_variants: int):
    """Per-variant CV output path (flat when only one variant is run)."""
    from pathlib import Path

    variant_name = variant.name if isinstance(variant, Variant) else variant
    base = Path(output_dir) / "beta_glmm"
    return base if n_variants == 1 else base / variant_name
