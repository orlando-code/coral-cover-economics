# Coral cover change and variability — analysis of run `031_cover_variability`

**Pipeline:** `python -m src.models.cover_variability_analysis --run-slug cover_variability`  
**Run directory:** [`outputs/031_cover_variability/`](../outputs/031_cover_variability/)  
**Self-contained export:** [`cover_variability_analysis.html`](cover_variability_analysis.html) (figures embedded; open in a browser → Print → Save as PDF)  
**Data:** Reef Check survey frame (1997–2025), 17,152 surveys at 5,051 unique sites  
**Sites with ≥3 repeat surveys:** 1,658 (used for trend and variability estimates)

This note interprets the observed (not model-projected) coral cover dynamics produced by the cover-variability pipeline. All cover values are reported in **percent** (%); slopes are in **percentage points per year** (pp yr⁻¹).

---

## Executive summary

1. **Global mean cover is not a simple monotonic decline.** Survey-weighted mean cover fell ~5.5 pp from 1997 to 2025 (33% → 27%), but the path is highly non-linear: a **2022 peak (~40%)** is followed by a **sharp 2025 drop (~27%)** that drives the endpoint decline. The global OLS trend over all years is slightly positive (+0.06 pp yr⁻¹, R² ≈ 0.04), reflecting mid-period recovery rather than sustained loss.

2. **Most individually tracked sites look “stable” under formal trend tests.** Among 1,658 multi-visit sites, 85% are classified stable (Kendall test, α = 0.05); only 4.2% declining and 5.4% increasing. Site-level median slope is −0.12 pp yr⁻¹, but the distribution is wide (5th–95th percentile: −8.4 to +8.1 pp yr⁻¹).

3. **Variability is substantial and mostly between sites, not within sites.** Overall SD ≈ 19.8 pp. Variance decomposition shows ~75% of total variance is **between-site** and ~25% **within-site** (repeat-visit noise around a site mean). That ratio is similar pre- and post-2018 (~78% / ~22%).

4. **Geographic and environmental patterns are heterogeneous.** Declining slopes concentrate in parts of the **Caribbean, South-East Asia, and the South Pacific** (e.g. Vanuatu, Taiwan, Sunda Shelf). Increasing slopes appear in **South-East Asia and the Western Pacific** (Philippines, Fiji, GBR, Java Sea). **Depth** and **historical SST max** show modest mean-slope differences; **latitude** (signed) correlates weakly with trend direction (more negative toward the equator in the northern hemisphere sample).

5. **Cover state transitions are dominated by persistence in the “moderate” class (10–50%).** Half of analysed sites remain moderate at both first and last visit; ~19% move from moderate to low; ~10% from high to moderate.

6. **MPA status is common but creation effects are weak / negative on average.** ~56% of surveys fall inside a WDPA marine/coastal polygon. Inside-MPA surveys have only ~1.3 pp higher mean cover than outside (Welch *p* ≪ 0.001). Among 94 sites with surveys both before and after mid-window MPA designation, median post−pre cover shift is **−2.1 pp**; ecoregion-matched DiD is similarly negative (median −1.7 pp).

---

## 1. Global cover through time

![Global coral cover time series](assets/cover_variability/global_cover_timeseries.png)

*Figure 1. Survey-weighted mean coral cover (solid blue) with ±1 SD ribbon; dashed red = site-balanced mean (each site contributes equally per year).*

| Metric | Value |
|--------|------:|
| Overall mean cover (all surveys) | 33.0% |
| Overall SD | 19.8 pp |
| 1997 survey-weighted mean | 32.9% |
| 2025 survey-weighted mean | 27.4% |
| Endpoint change (1997 → 2025) | −5.5 pp |
| Pre-2018 era mean (yearly averages) | 32.8% |
| Post-2018 era mean | 33.9% |
| 2022 peak | 39.7% |
| 2025 trough | 27.4% |
| Global OLS slope (yearly means) | +0.06 pp yr⁻¹ (R² = 0.04) |

**Interpretation.** The time series is dominated by **inter-annual and sampling fluctuations**, not a steady long-term trend. Cover was relatively flat or slightly depressed in the early 2000s (~29–31%), recovered through the 2010s (~35% in 2012–2015), dipped around 2017–2019, spiked in **2022**, then fell sharply in **2025**. The 2025 value should be treated cautiously: only **232 surveys at 153 sites** (vs 544/456 in 2022), so it may partly reflect **sampling composition** and incomplete year coverage rather than a fully representative global state.

Survey-weighted and site-balanced means track closely, suggesting the global curve is not driven solely by oversampling of high- or low-cover regions in particular years.

---

## 2. Survey effort

![Survey effort through time](assets/cover_variability/survey_effort.png)

*Figure 2. Number of surveys (bars) and unique sites surveyed (line) by year.*

Reef Check effort grows from a few hundred surveys in the late 1990s to **700–850 surveys/year** in the mid-2010s. The post-2020 period shows partial recovery after a 2020 dip (397 surveys), with 2022–2024 again above 540 surveys. Expanded effort increases the number of sites with ≥3 visits (1,658 in this run) but also changes which ecoregions dominate each year’s global mean — an important caveat when reading year-to-year shifts.

---

## 3. Site-level trends

![Distribution of site OLS slopes](assets/cover_variability/site_slope_histogram.png)

*Figure 3. Per-site OLS slope (% cover yr⁻¹) for sites with ≥3 surveys.*

| Site-level statistic | Value |
|---------------------|------:|
| Sites analysed | 1,658 |
| Mean slope | −0.23 pp yr⁻¹ |
| Median slope | −0.12 pp yr⁻¹ |
| Fraction declining (Kendall p < 0.05, slope < 0) | 4.2% |
| Fraction increasing | 5.4% |
| Fraction stable | 85.2% |
| Insufficient temporal resolution | 5.2% |

**Interpretation.** The histogram is centred slightly below zero but with **long tails**: a minority of sites lose or gain cover at several pp yr⁻¹. Formal trend classification is conservative — many sites have noisy, irregular revisit schedules, so Kendall tests rarely reject “no trend” even when OLS slopes are negative. The median site is effectively **flat** over its observation window.

This site-level picture **differs from the global endpoint change** (−5.5 pp): endpoint change mixes **which sites are sampled in early vs late years**, **regional compositional shift**, and **recent global downturn (2025)**, whereas site slopes fit each reef’s own timeline.

---

## 4. Within-site variability

![Within-site cover SD](assets/cover_variability/site_variability_histogram.png)

*Figure 4. Distribution of within-site cover standard deviation (pp) across multi-visit sites.*

| Variability metric | Value |
|-------------------|------:|
| Mean within-site SD | 9.7 pp |
| Median within-site SD | 8.7 pp |
| 95th percentile within-site SD | 20.5 pp |

![Mean cover vs within-site SD](assets/cover_variability/mean_vs_variability_scatter.png)

*Figure 5. Each point is a site; colour = OLS slope. Sites with higher mean cover do not systematically show lower visit-to-visit SD, but extreme slopes (red/blue tails) appear at all cover levels.*

**Variance decomposition** (all surveys):

| Component | Variance (pp²) | Fraction |
|-----------|---------------:|---------:|
| Total | 391.5 | 100% |
| Between-site | 293.4 | 75.0% |
| Within-site | 98.0 | 25.0% |

Pre-2018 vs extension eras show nearly identical between/within ratios (~78% / ~22%). **Spatial heterogeneity dominates**: reefs differ greatly in average cover; repeat visits add roughly a quarter of total spread.

---

## 5. Ecoregion patterns

![Mean site slope by ecoregion](assets/cover_variability/ecoregion_mean_slopes.png)

*Figure 6. Mean site OLS slope by ecoregion (ecoregions with largest declines and increases shown).*

![Trend class composition by ecoregion](assets/cover_variability/ecoregion_trend_classes.png)

*Figure 7. Stacked fraction of declining / stable / increasing sites within each ecoregion (top 15 by site count).*

![Ecoregion slope map](assets/cover_variability/ecoregion_slope_map.png)

*Figure 8. Choropleth of mean site slope by MEOW/COTW ecoregion.*

![Ecoregion × year heatmap](assets/cover_variability/ecoregion_year_heatmap.png)

*Figure 9. Site-balanced mean cover (%) for the 20 ecoregions with most site-years.*

### Regions with notably negative mean slopes (≥10 sites)

| Ecoregion | Sites | Mean slope (pp yr⁻¹) | Mean cover (%) |
|-----------|------:|---------------------:|---------------:|
| Vanuatu | 21 | −6.0 | 27.5 |
| Taiwan and coastal China | 45 | −2.8 | 29.3 |
| Gulf of Thailand | 20 | −2.0 | 37.4 |
| Sunda Shelf, SE Asia | 218 | −1.0 | 45.7 |
| Hispaniola, Puerto Rico & Lesser Antilles | 89 | −1.4 | 17.9 |
| Belize and west Caribbean | 47 | −1.5 | 19.9 |

Caribbean ecoregions combine **low mean cover** (~12–20%) with **negative mean slopes**. South-East Asian shelves (Sunda, Sulu) have **higher mean cover** (~38–46%) but still slightly negative mean slopes.

### Regions with notably positive mean slopes (≥10 sites)

| Ecoregion | Sites | Mean slope (pp yr⁻¹) | Mean cover (%) |
|-----------|------:|---------------------:|---------------:|
| Fiji | 52 | +3.2 | 36.5 |
| South-east Philippines | 78 | +2.8 | 35.1 |
| Central & northern GBR | 99 | +1.1 | 34.4 |
| Java Sea | 19 | +1.3 | 41.0 |
| Andaman Sea | 26 | +1.5 | 38.1 |

The heatmap (Figure 9) shows **ecoregion-specific trajectories** decoupled from the global curve: e.g. some Western Pacific ecoregions brighten in the 2010s–2022 while Caribbean rows stay pale (low cover) throughout.

**Caution:** ecoregions with few sites (1–4) show extreme slopes (e.g. North Vietnam −21 pp yr⁻¹) that are **not robust**; maps and ranked tables should be read with sample-size in mind.

---

## 6. Latitude

![Mean slope by latitude bin](assets/cover_variability/latitude_mean_slopes.png)

*Figure 10. Mean site slope by 10° absolute-latitude bin.*

| Lat bin (°) | Sites | Mean slope (pp yr⁻¹) | Mean cover (%) |
|------------|------:|---------------------:|---------------:|
| 0–10 | 648 | −0.16 | 38.1 |
| 10–20 | 584 | +0.11 | 29.8 |
| 20–30 | 418 | −0.75 | 32.5 |
| 30–40 | 8 | −2.98 | 38.1 |

Tropical sites (0–10°) carry the largest sample. The 20–30° bin shows the most negative mean slope; the 30–40° bin has only **8 sites** and is unreliable. Signed-latitude correlations (Table below) suggest **slightly more negative trends at low northern latitudes** in this dataset, but effect sizes are small.

---

## 7. Depth

![Mean slope by depth bin](assets/cover_variability/depth_mean_slopes.png)

*Figure 11. Mean site slope by median survey depth at each site.*

| Depth bin | Sites | Mean slope (pp yr⁻¹) | Mean cover (%) | Mean within-site SD (pp) |
|-----------|------:|---------------------:|---------------:|-------------------------:|
| 0–5 m | 438 | −0.63 | 34.6 | 9.7 |
| 5–10 m | 987 | +0.01 | 33.6 | 10.0 |
| 10–20 m | 233 | −0.48 | 32.8 | 8.4 |

Shallow (0–5 m) and deeper (10–20 m) sites show modestly negative mean slopes; the **5–10 m band** (majority of sites) is near zero on average. Depth is **not strongly correlated** with slope (Pearson r ≈ 0, p ≈ 0.98). Within-site SD is similar across bins (~8–10 pp).

---

## 8. Historical maximum SST

![Mean slope by historic SST max quartile](assets/cover_variability/historical_sst_max_mean_slopes.png)

*Figure 12. Mean site slope by quartile of site-level median `historical_sst_max` (CMIP historic ensemble, 1870–1980).*

| SST max quartile | Sites | Mean slope (pp yr⁻¹) | Mean cover (%) | Mean within-site SD (pp) |
|-----------------|------:|---------------------:|---------------:|-------------------------:|
| Q1 (coolest) | 415 | −0.24 | 32.8 | 8.6 |
| Q2 | 412 | −0.40 | 26.9 | 8.7 |
| Q3 | 398 | +0.02 | 35.8 | 10.0 |
| Q4 (warmest) | 433 | −0.27 | 39.3 | 11.5 |

Warmest-quartile sites have **higher mean cover** but also **higher visit-to-visit SD** (11.5 vs 8.6 pp in Q1). Mean slopes differ only slightly across quartiles; **`historical_sst_max` does not predict site trend direction** (Pearson r = −0.009, p = 0.73). It **does** predict variability: warmer historic maxima associate with higher within-site SD (r = +0.21, p < 10⁻¹⁶).

---

## 9. Cover state transitions

![Cover state transitions](assets/cover_variability/cover_state_transitions.png)

*Figure 13. Transitions between cover states from first to last visit (low < 10%, moderate 10–50%, high > 50%).*

| First → Last | Sites | Fraction |
|--------------|------:|---------:|
| Moderate → Moderate | 828 | 49.9% |
| High → High | 197 | 11.9% |
| High → Moderate | 170 | 10.3% |
| Moderate → Low | 153 | 9.2% |
| Moderate → High | 150 | 9.0% |
| Low → Moderate | 73 | 4.4% |
| Low → Low | 72 | 4.3% |
| High → Low | 8 | 0.5% |

**~50% of sites stay moderate** throughout their window. Downward transitions (moderate→low, high→moderate) outnumber upward ones slightly (~22% vs ~14% excluding stable moderate/moderate). Direct **high→low** collapse is rare (8 sites, 0.5%).

---

## 10. Environmental correlates of trend and variability

From [`env_trend_correlations.csv`](../outputs/031_cover_variability/env_trend_correlations.csv) (1,572–1,658 sites):

| Predictor | Response | Pearson r | p-value | Takeaway |
|-----------|----------|----------:|--------:|----------|
| `historical_sst_max` | Slope | −0.009 | 0.73 | No linear trend relationship |
| `historical_sst_max` | Within-site SD | +0.21 | < 10⁻¹⁶ | Warmer historic max → noisier sites |
| `depth_m` | Slope | −0.001 | 0.98 | No trend relationship |
| `latitude` (signed) | Slope | −0.10 | 0.0001 | Lower latitudes (NH) → slightly more negative slopes |
| `latitude` | Total change (first→last) | −0.09 | 0.0004 | Same direction, small effect |
| `abs_lat` | Within-site SD | −0.14 | < 10⁻⁸ | Higher latitudes → slightly lower SD |

Ecological takeaway: **thermal history predicts how much cover fluctuates at a site, not which direction it trends.** Geographic trend patterns are better explained by **region-specific disturbance and recovery histories** (visible in ecoregion figures) than by simple SST-max or depth gradients alone.

---

## 11. Marine protected areas (WDPA)

MPA status is attached by point-in-polygon join to **WDPA June 2026** marine and coastal polygons (`join_mpa_protected`), using each survey’s signed latitude/longitude. When multiple polygons overlap a site, the earliest `STATUS_YR` is retained as the designation year.

Site exposure classes (relative to each reef’s own survey window):

| Class | Definition | Sites |
|-------|------------|------:|
| Never MPA | Outside all marine/coastal WDPA polygons (during observation) | 2,938 |
| Always MPA | Inside an MPA designated before the site’s first survey | 1,694 |
| Switched into MPA | Designation year falls after first survey and ≤ last survey | 245 |
| MPA (unknown year) | Protected but `STATUS_YR` missing/invalid | 174 |

### Inside vs outside

![MPA cover comparison](assets/cover_variability/mpa_cover_comparison.png)

*Figure 14. Survey-level mean coral cover inside vs outside WDPA marine/coastal MPAs.*

| Status | Surveys | Sites | Mean cover (%) | Median (%) |
|--------|--------:|------:|---------------:|-----------:|
| Outside MPA | 7,571 | 2,335 | 32.2 | 30.0 |
| Inside MPA | 9,581 | 2,788 | 33.5 | 31.3 |

Difference (inside − outside) = **+1.3 pp** (Welch *t* = 4.29, *p* = 1.8×10⁻⁵). The association is statistically clear but **small** relative to overall cover SD (~20 pp), and is **not causal** — MPAs are non-randomly sited.

### Trends by MPA class

![MPA class slopes](assets/cover_variability/mpa_class_slopes.png)

*Figure 15. Mean site OLS slope by MPA exposure class (sites with ≥3 surveys).*

| Class | Sites (≥3 surveys) | Mean slope (pp yr⁻¹) | Mean cover (%) |
|-------|-------------------:|---------------------:|---------------:|
| Always MPA | 682 | −0.36 | 34.4 |
| Never MPA | 790 | −0.25 | 33.3 |
| Switched into MPA | 154 | +0.28 | 33.9 |
| MPA (unknown year) | 32 | +0.47 | 30.2 |

Sites that were **already protected** before monitoring show slightly **more negative** mean slopes than never-MPA sites. Sites that **gained** an MPA mid-window have a near-zero/slightly positive mean slope over their full record — but that mixes pre- and post-designation years (see creation effects below).

### Effect of MPA creation within the sample window

For switched sites with ≥2 surveys before and ≥2 after designation (**n = 94**):

![MPA creation shifts](assets/cover_variability/mpa_creation_shifts.png)

*Figure 16. Left: post − pre mean cover at newly designated MPA sites. Right: difference-in-differences vs never-MPA sites in the same ecoregion (same break year).*

| Metric | Value |
|--------|------:|
| Sites with enough pre/post surveys | 94 |
| Mean level shift (post − pre) | −2.3 pp |
| Median level shift | −2.1 pp |
| Fraction with positive shift | 42.6% |
| Sites with ecoregion-matched controls | 60 |
| Mean DiD (treated − control) | −4.7 pp |
| Median DiD | −1.7 pp |
| Fraction with positive DiD | 43.3% |

**Interpretation.** On average, coral cover is **lower after** mid-window MPA designation than before at the same sites. Relative to never-MPA reefs in the same ecoregion (using the treated site’s designation year as a shared break), the DiD estimate is also negative. This does **not** imply that MPAs cause decline: designation often follows bleaching, fishing pressure, or other crises; enforcement lag is common; and sample sizes per site are modest. The distribution is wide (level shifts from about −52 to +28 pp), so some sites do improve after designation.

![MPA event study](assets/cover_variability/mpa_event_study.png)

*Figure 17. Event study: site-balanced mean cover by years relative to MPA designation (switched sites, ±10 years).*

The event-study curve shows no clear step-up at year 0; cover around designation is noisy and often flat-to-declining, consistent with the negative average level shift.

### MPA takeaways

1. **Inside MPAs have modestly higher mean cover** (+1.3 pp) but similar within-site variability.
2. **Always-protected sites are not trending better** than unprotected ones in this sample.
3. **Mid-window MPA creation** is associated with a small **negative** average before→after cover change and a negative ecoregion-matched DiD — consistent with designation amid stress rather than a detectable recovery signal in Reef Check revisit data.
4. Causal claims would need stronger designs (e.g. matching on pre-trends, IUCN/no-take strata, enforcement timing).

---

## 12. Synthesis and limitations

### What the data support

- Coral cover in the Reef Check global sample is **highly variable in space** (~75% between-site variance).
- **Within-site revisit SD** (~9.7 pp on average) is non-trivial and sets a floor on detectable trends without long, frequent time series.
- **Regional stories differ**: Caribbean low-cover, slightly declining; parts of SE Asia and South Pacific with negative mean slopes; Western Pacific and some Australian/Indonesian ecoregions with positive mean slopes.
- The **2022 global high** and **2025 drop** are the dominant features of the recent global curve and warrant separate investigation (bleaching years, survey gaps, site turnover).
- **MPA presence** is associated with slightly higher mean cover, but **MPA creation mid-window** shows no average recovery signal (median post−pre ≈ −2 pp).

### Limitations

1. **Site selection and effort** — Reef Check sites are not a random global sample; expanded effort post-2003 and uneven 2025 coverage bias aggregate means.
2. **Minimum 3 surveys** — Trend estimates exclude 3,393 sites with fewer visits; excluded sites may differ systematically.
3. **Trend tests** — Irregular revisit intervals and same-year repeats weaken Kendall/OLS power; 85% “stable” may understate real change.
4. **Confounding** — Ecoregion, depth, SST, and MPA status correlate with geography; multivariate models would be needed to isolate drivers.
5. **Endpoint vs slope** — Short 2025 sampling makes “1997–2025 decline” sensitive to the last year included.
6. **MPA join** — Strict point-in-polygon against WDPA polygons; sites just outside boundaries are coded unprotected; `STATUS_YR` can be missing or outdated; no enforcement intensity.

### Suggested follow-ups

- Align with [`period_break_analysis`](../MODELING.md) (2018 era shift) on the same frame for pre/post-2018 level shifts vs full-series slopes.
- Stratify the 2022–2025 downturn by ecoregion and bleaching year.
- Restrict to **sites with ≥5 visits and span ≥10 years** for a robust trend subset.
- Compare this observed trajectory to HBB **projected** cover used in economics (different construct).
- Stratify MPA creation effects by **IUCN category / no-take** and require parallel pre-trends vs matched controls.

---

## Figure index

| Figure | File |
|--------|------|
| 1 Global time series | [`global_cover_timeseries.png`](assets/cover_variability/global_cover_timeseries.png) |
| 2 Survey effort | [`survey_effort.png`](assets/cover_variability/survey_effort.png) |
| 3 Site slope histogram | [`site_slope_histogram.png`](assets/cover_variability/site_slope_histogram.png) |
| 4 Within-site SD histogram | [`site_variability_histogram.png`](assets/cover_variability/site_variability_histogram.png) |
| 5 Mean vs variability | [`mean_vs_variability_scatter.png`](assets/cover_variability/mean_vs_variability_scatter.png) |
| 6 Ecoregion slopes | [`ecoregion_mean_slopes.png`](assets/cover_variability/ecoregion_mean_slopes.png) |
| 7 Ecoregion trend classes | [`ecoregion_trend_classes.png`](assets/cover_variability/ecoregion_trend_classes.png) |
| 8 Ecoregion map | [`ecoregion_slope_map.png`](assets/cover_variability/ecoregion_slope_map.png) |
| 9 Ecoregion × year heatmap | [`ecoregion_year_heatmap.png`](assets/cover_variability/ecoregion_year_heatmap.png) |
| 10 Latitude slopes | [`latitude_mean_slopes.png`](assets/cover_variability/latitude_mean_slopes.png) |
| 11 Depth slopes | [`depth_mean_slopes.png`](assets/cover_variability/depth_mean_slopes.png) |
| 12 Historic SST max slopes | [`historical_sst_max_mean_slopes.png`](assets/cover_variability/historical_sst_max_mean_slopes.png) |
| 13 Cover state transitions | [`cover_state_transitions.png`](assets/cover_variability/cover_state_transitions.png) |
| 14 MPA inside vs outside | [`mpa_cover_comparison.png`](assets/cover_variability/mpa_cover_comparison.png) |
| 15 MPA class slopes | [`mpa_class_slopes.png`](assets/cover_variability/mpa_class_slopes.png) |
| 16 MPA creation shifts / DiD | [`mpa_creation_shifts.png`](assets/cover_variability/mpa_creation_shifts.png) |
| 17 MPA event study | [`mpa_event_study.png`](assets/cover_variability/mpa_event_study.png) |

**Data tables:** [`cover_variability_report.json`](../outputs/031_cover_variability/cover_variability_report.json), [`site_variability_stats.csv`](../outputs/031_cover_variability/site_variability_stats.csv), [`site_mpa_classification.csv`](../outputs/031_cover_variability/site_mpa_classification.csv), [`mpa_creation_effects.csv`](../outputs/031_cover_variability/mpa_creation_effects.csv), and sibling CSVs in the run directory.
