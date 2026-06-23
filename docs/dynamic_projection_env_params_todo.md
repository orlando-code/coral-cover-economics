# Dynamic projection — environmental parameters to add

Priority list for improving longitudinal coral-cover projection beyond the current
Sully covariate set. Ordered by expected incremental value for **within-reef
forecasting** and **scenario projection**.

## Tier 1 — highest priority (dynamic stress & memory)

These directly address the gap your decomposition exposed: persistence dominates,
but **recent thermal history** should explain deviations from site baselines.

| Parameter | Rationale | Source ideas |
|---|---|---|
| **Cumulative DHW to date** | Integrates bleaching memory; better than static `Historical_SST_max` alone | NOAA Coral Reef Watch, UNEP-LIVE |
| **Rolling 3–5 yr SST anomaly mean** | Captures recent warming trend at site | CRW SST, HadISST/OISST at reef coords |
| **Years since last major bleaching event** | Recovery dynamics; reefs behave differently post-bleach | Derived from DHW time series (>4 °C-weeks) |
| **SSTA trend (slope over prior 5 yr)** | Direction of local thermal change | Fit from monthly SST at site |
| **Acute heat exposure in prior 12 months** | Short-horizon forcing for next survey | CRW `ssta_dhwmax` extended backward |

## Tier 2 — disturbance & local context

| Parameter | Rationale | Source ideas |
|---|---|---|
| **Cyclone exposure index (cumulative)** | Your model has `Cyclone` point values; cumulative tracks may help projections | IBTrACS buffer around reef |
| **Crown-of-thorns outbreak history** | Major driver of cover change on GBR / Pacific | AIMS, local monitoring |
| **Fishing pressure / market gravity** | Finer than `Human_pop` alone | Cinner et al. fisheries accessibility |
| **Marine protected area status** | Management moderates decline/recovery | WDPA, Allen et al. |
| **Sediment / river plume exposure** | Turbidity_mean is static; plume frequency matters | Remote sensing (MODIS turbidity) |

## Tier 3 — reef state & connectivity (static but informative for new sites)

Useful mainly for **spatial extrapolation** (`site_group_kfold`), less for repeat-site projection.

| Parameter | Rationale | Source ideas |
|---|---|---|
| **Reef geomorphology** (atoll vs fringing vs barrier) | Controls recovery rate and stress sensitivity | UNEP atlas, Allen morphometrics |
| **Connectivity / larval supply proxy** | Recovery potential after disturbance | Ocean circulation models (Connolly et al.) |
| **Calcification saturation state (Ωarag)** | Long-run viability under acidification | GLODAP / Earth system model output |
| **Wave energy exposure** | Mechanical disturbance, turbidity resuspension | GEBCO + wave reanalysis |

## Tier 4 — scenario-linked futures (for projection application)

Required for SSP/RCP scenario maps once the dynamic model is validated.

| Parameter | Scenario years | Notes |
|---|---|---|
| **SST_mean** (already present) | 2050, 2100 | Extend to SSP1-2.6 / SSP2-4.5 / SSP5-8.5 |
| **Human population** (already present) | 2050, 2100 | UN WPP coastal projections |
| **Turbidity under land-use change** | 2050, 2100 | Coupled catchment models if available |
| **Cyclone intensity frequency** | 2050, 2100 | CMIP6 downscaled cyclone projections |
| **Ocean acidification (pH / Ω)** | 2100 | From ESMs at reef depth |

## Features to deprioritise

| Parameter | Reason |
|---|---|
| **Ecoregion diversity** | ~0 incremental R² once site effects included; not a future scenario variable |
| **Nested ecoregion RE** | Redundant with site RE on repeat-site CV |
| **Static `Historical_SST_max` alone** | Already in model; supersede with dynamic cumulative metrics |

## Suggested acquisition workflow

1. **Extract at reef coordinates** for all Tier 1 variables (same `Reef_ID` join as `model_ready_data.csv`).
2. **Build rolling / cumulative features** in `build_sully_model_ready_data.py` with train-only normalization in CV.
3. **Re-run** `forward_repeat_sites` CV comparing:
   - `survey_mean` (persistence)
   - `reparam` (Sully corrected)
   - `persist_ar_env` / `persist_residual_xgboost` (dynamic)
4. **Report** level R², ΔR² vs persistence, and `prior_only_r2` (rows with lagged cover).

## Open methodological items

- [ ] Bayesian hierarchical AR(1) on logit scale (full uncertainty propagation)
- [ ] Conformal / bootstrap prediction intervals for XGBoost residual models
- [ ] Pseudo-projection holdout: predict most recent survey from prior + scenario covariates
- [ ] Spatial block CV for unseen-reef generalization
- [ ] Reef × warming interaction terms
