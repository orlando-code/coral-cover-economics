#!/usr/bin/env python3
"""Generate lat_cap_sensitivity.csv and ecoregion_contributions.csv for cover.ipynb."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.models.hbb.data import load_model_data_from_pipeline, standardize_variables
from src.models.hbb.run_investigation import (
    VARIANTS,
    add_latitude_features,
    build_variant_data,
    ecoregion_predictor_contributions,
    fit_variant,
    load_trace_nc,
    run_latitude_cap_sensitivity,
    standardization_vars,
)


def main() -> None:
    df = load_model_data_from_pipeline(
        config.data_dir / "sully_og", force_rebuild=False, index_source="paper_factor"
    )
    df = add_latitude_features(df)
    df_std, _ = standardize_variables(df, standardization_vars())
    out = config.data_dir / "sully_og" / "output_python" / "investigation"
    out.mkdir(parents=True, exist_ok=True)

    print("Running latitude cap sensitivity (6 caps)...")
    sens = run_latitude_cap_sensitivity(
        df_std,
        caps=[None, 30, 25, 20, 15, 10],
        draws=300,
        tune=300,
        chains=2,
        ncores=1,
        progressbar=True,
    )
    sens_path = out / "lat_cap_sensitivity.csv"
    sens.to_csv(sens_path, index=False)
    print(f"Wrote {sens_path}")
    print(
        sens.loc[sens.variable.eq("Latitude"), ["cap_label", "mean", "delta_cover_pi0.3", "n_obs"]]
        .round(4)
        .to_string(index=False)
    )

    built = build_variant_data(df_std, VARIANTS["reparam"])
    work = built["df"].copy()
    work["region_idx"] = built["region_idx"]
    region_predictors = (
        work.groupby("region_idx", as_index=False)
        .agg(
            abs_lat=("lat", "mean"),
            lat_stzd=("lat_stzd", "mean"),
            historical_sst_max_stzd=("historical_sst_max_stzd", "mean"),
            n_sites=("site", "nunique"),
            ecoregion=("ecoregion", "first"),
        )
        .sort_values("region_idx")
    )

    print("Fitting reparam model for ecoregion decomposition...")
    with tempfile.TemporaryDirectory(prefix="eco_decomp_") as tmp:
        fit_variant(
            variant=VARIANTS["reparam"],
            df_std=df_std,
            output_dir=Path(tmp),
            draws=300,
            tune=300,
            chains=2,
            ncores=1,
            target_accept=0.95,
            max_treedepth=15,
            seed=42,
            progressbar=True,
        )
        idata = load_trace_nc(Path(tmp) / "trace.nc")

    eco = ecoregion_predictor_contributions(
        idata,
        col_names=built["col_names"],
        diversity=built["diversity"],
        region_predictors=region_predictors,
        pi_ref=0.3,
    )
    eco_path = out / "ecoregion_contributions.csv"
    eco.to_csv(eco_path, index=False)
    print(f"Wrote {eco_path}")


if __name__ == "__main__":
    main()
