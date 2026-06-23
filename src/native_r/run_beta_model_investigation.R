#!/usr/bin/env Rscript

# Systematic comparison: original centered beta-GLMM vs reparameterized implementation.
#
# Hypotheses tested:
#   1. Extra fixed intercept (beta[1]) alongside mu_global hurts identification/convergence
#   2. Centered vs non-centered hierarchical parameterization
#   3. Diversity vector ordering / R indexing (n_erg vs n_regions, alphabetical vs aligned)
#
# Outputs:
#   data/sully_og/output/original/          — original spec (intercept + centered)
#   data/sully_og/output/reparam/           — reparam spec (no intercept + non-centered)
#   data/sully_og/output/investigation/     — ablation variants + comparison plot
#
# Usage:
#   Rscript run_beta_model_investigation.R
#   INV_SMOKE=1 Rscript run_beta_model_investigation.R   # tiny subsample, short MCMC
#   BETA_JAGS_PARALLEL=0 Rscript run_beta_model_investigation.R  # opt-out of jags.parallel

suppressPackageStartupMessages({
  library(dplyr)
  library(R2jags)
  library(ggplot2)
})

script_args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", script_args, value = TRUE)
script_dir <- if (length(file_arg)) {
  dirname(normalizePath(sub("^--file=", "", file_arg[1]), wins = FALSE, mustWork = FALSE))
} else {
  getwd()
}
source(file.path(script_dir, "beta_model_reparam_utils.R"))

default_data_dir <- "/Users/rt582/Library/CloudStorage/OneDrive-UniversityofCambridge/cambridge/phd/Paper_Conferences/reef_cover_economics/data/sully_og"
default_output_root <- file.path(default_data_dir, "output")

is_inv_smoke <- function() {
  identical(Sys.getenv("INV_SMOKE"), "1") ||
    identical(Sys.getenv("BETA_SMOKE"), "1") ||
    identical(Sys.getenv("RPARAM_SMOKE"), "1")
}

cfg <- list(
  data_dir = Sys.getenv("BETA_DATA_DIR", unset = default_data_dir),
  output_root = Sys.getenv("BETA_OUTPUT_ROOT", unset = default_output_root),
  seed = 20260529L,
  y_eps = 1e-6,
  use_parallel = resolve_jags_parallel(default = TRUE),
  monitor_params = DEFAULT_MONITOR_PARAMS,
  mcmc = list(
    n_chains = 6L,
    # Paper small run: n_burnin=4000, n_iter=15000 (→ 11000 post-burnin / 10 = 1100 per chain).
    # Large run (saved_parallel_model_large.RData): 6 chains, 10000 burnin, 20000 total.
    n_burnin = 10000L,
    n_iter   = 20000L,
    n_thin   = 10L
  )
)

if (is_inv_smoke()) {
  smoke <- smoke_mcmc_settings()
  cfg$mcmc$n_chains <- smoke$n_chains
  cfg$mcmc$n_burnin <- smoke$n_burnin
  cfg$mcmc$n_iter <- smoke$n_iter
  cfg$mcmc$n_thin <- smoke$n_thin
  Sys.setenv(BETA_SMOKE = "1")
}

cfg$mcmc$use_parallel <- cfg$use_parallel
msg("JAGS parallel: %s (set BETA_JAGS_PARALLEL=0 to disable)", cfg$use_parallel)

MODEL_VARIANTS <- list(
  paper_reproduction = list(
    label = "paper_reproduction",
    output_subdir = "investigation/01_paper_reproduction",
    include_intercept = TRUE,
    parameterization = "centered",
    diversity_ordering = "region_factor",
    r_index = "n_erg",
    model_kind = "original_centered",
    region_encoding = "paper_factor",
    primary = TRUE
  ),
  paper_region_fixed = list(
    label = "paper_region_fixed",
    output_subdir = "investigation/02_paper_region_fixed",
    include_intercept = TRUE,
    parameterization = "centered",
    diversity_ordering = "region_factor",
    r_index = "n_erg",
    model_kind = "original_centered",
    region_encoding = "dense",
    primary = TRUE
  ),
  original_no_intercept = list(
    label = "original_no_intercept",
    output_subdir = "investigation/03_no_intercept_centered",
    include_intercept = FALSE,
    parameterization = "centered",
    diversity_ordering = "region_factor",
    r_index = "n_erg",
    model_kind = "original_centered",
    region_encoding = "dense",
    primary = TRUE
  ),
  original_paper_diversity = list(
    label = "original_paper_diversity",
    output_subdir = "investigation/ablation_paper_diversity",
    include_intercept = TRUE,
    parameterization = "centered",
    diversity_ordering = "paper_alphabetical",
    r_index = "n_erg",
    model_kind = "original_centered",
    region_encoding = "dense",
    primary = FALSE
  ),
  original_aligned_diversity = list(
    label = "original_aligned_diversity",
    output_subdir = "investigation/ablation_aligned_diversity",
    include_intercept = TRUE,
    parameterization = "centered",
    diversity_ordering = "region_dense",
    r_index = "n_regions",
    model_kind = "original_centered",
    region_encoding = "dense",
    primary = FALSE
  ),
  reparam_centered = list(
    label = "reparam_centered",
    output_subdir = "investigation/04_reparam_centered",
    include_intercept = FALSE,
    parameterization = "centered",
    diversity_ordering = "region_dense",
    r_index = "n_regions",
    model_kind = "reparam_centered",
    region_encoding = "dense",
    primary = TRUE
  ),
  reparam = list(
    label = "reparam",
    output_subdir = "investigation/05_reparam_noncentered",
    include_intercept = FALSE,
    parameterization = "noncentered",
    diversity_ordering = "region_dense",
    r_index = "n_regions",
    model_kind = "reparam",
    region_encoding = "dense",
    primary = TRUE
  )
)

run_variant <- function(variant, df, cfg, init_seed) {
  output_dir <- file.path(cfg$output_root, variant$output_subdir)
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  start_time <- Sys.time()

  msg(
    "Fitting variant '%s' -> %s",
    variant$label,
    output_dir
  )

  if (identical(variant$model_kind, "reparam") &&
      !variant$include_intercept &&
      variant$diversity_ordering == "region_dense" &&
      variant$r_index == "n_regions") {
    pkg <- build_jags_data(df, y_eps = cfg$y_eps)
    pkg$spec <- list(
      include_intercept = FALSE,
      parameterization = "noncentered",
      diversity_ordering = "region_dense",
      r_index = "n_regions",
      region_encoding = variant$region_encoding %||% "dense"
    )
  } else {
    pkg <- build_investigation_jags_data(
      df,
      include_intercept  = variant$include_intercept,
      parameterization   = variant$parameterization,
      diversity_ordering = variant$diversity_ordering,
      r_index            = variant$r_index,
      y_eps              = cfg$y_eps,
      region_encoding    = variant$region_encoding %||% "dense"
    )
  }

  msg(
    "  N=%d K=%d Nre=%d R=%d | intercept=%s param=%s diversity=%s r_index=%s encoding=%s",
    pkg$win.data$N,
    pkg$win.data$K,
    pkg$win.data$Nre,
    pkg$win.data$R,
    variant$include_intercept,
    variant$parameterization,
    variant$diversity_ordering,
    variant$r_index,
    variant$region_encoding %||% "dense"
  )

  input_summary <- data.frame(
    variant = variant$label,
    N = pkg$win.data$N,
    K = pkg$win.data$K,
    Nre = pkg$win.data$Nre,
    R = pkg$win.data$R,
    n_site = length(unique(pkg$data$site)),
    n_region = length(unique(pkg$data$region)),
    n_erg = if ("ERG" %in% names(pkg$data)) length(unique(pkg$data$ERG)) else NA_integer_,
    include_intercept = variant$include_intercept,
    parameterization = variant$parameterization,
    diversity_ordering = variant$diversity_ordering,
    r_index = variant$r_index,
    region_encoding = variant$region_encoding %||% "dense",
    X_columns = paste(colnames(pkg$X), collapse = "|"),
    stringsAsFactors = FALSE
  )
  write.csv(input_summary, file.path(output_dir, "model_input_summary.csv"), row.names = FALSE)

  fit_res <- run_investigation_jags_fit(
    pkg = pkg,
    cfg = cfg,
    output_dir = output_dir,
    model_kind = variant$model_kind,
    init_seed = init_seed
  )

  beta_df <- write_beta_fit_outputs(
    fit = fit_res$fit,
    pkg = pkg,
    output_dir = output_dir,
    variant_label = variant$label,
    include_intercept = variant$include_intercept,
    monitor_params = fit_res$monitor_params,
    start_time = start_time,
    mcmc_cfg = cfg$mcmc
  )

  conv <- read.csv(
    file.path(output_dir, "convergence_diagnostics.csv"),
    row.names = 1,
    check.names = FALSE
  )
  max_rhat <- max(conv$Rhat, na.rm = TRUE)
  msg("  max Rhat = %.4f", max_rhat)

  list(
    variant = variant$label,
    output_dir = output_dir,
    fit = fit_res$fit,
    beta_df = beta_df,
    max_rhat = max_rhat,
    spec = pkg$spec,
    input_summary = input_summary
  )
}

write_investigation_report <- function(results, diversity_diag, report_path) {
  lines <- c(
    "# Beta model investigation summary",
    "",
    sprintf("Generated: %s", Sys.time()),
    "",
    "## Diversity alignment diagnostics",
    sprintf("- n_regions (dense): %d", diversity_diag$n_regions),
    sprintf("- n_erg: %s", diversity_diag$n_erg),
    sprintf(
      "- Misaligned entries (paper alphabetical vs region_dense, first min(R) regions): %s",
      diversity_diag$n_misaligned_paper
    ),
    "",
    "## Model variants",
    ""
  )

  for (res in results) {
    lines <- c(
      lines,
      sprintf("### %s", res$variant),
      sprintf("- output: `%s`", res$output_dir),
      sprintf("- max Rhat: %.4f", res$max_rhat),
      sprintf(
        "- spec: intercept=%s, param=%s, diversity=%s, r_index=%s, region_encoding=%s",
        res$spec$include_intercept,
        res$spec$parameterization,
        res$spec$diversity_ordering,
        res$spec$r_index,
        res$spec$region_encoding %||% "dense"
      ),
      ""
    )
  }

  lines <- c(
    lines,
    "## Comparison design",
    "",
    "All variants are now built from the same paper-style filtered data. The shared loader",
    "uses the curated `data_for_maps.csv` ecoregion/diversity lookup because the original",
    "diversity dataset is not available, then reconstructs `site` and `region` exactly as",
    "`my_1_run_the_beta_model.Rmd` does: `site = as.numeric(as.factor(Reef_ID))` and",
    "`region = as.numeric(as.factor(Ecoregion))` after filtering.",
    "",
    "The main comparison sequence is:",
    "",
    "1. `paper_reproduction`: paper-style data, fixed intercept in X, centered hierarchy,",
    "   and the paper's `as.factor(as.character(region))` site-to-ecoregion encoding.",
    "2. `paper_region_fixed`: same model, but corrected integer `region_for_each_site`.",
    "3. `original_no_intercept`: removes the fixed intercept while keeping centered effects.",
    "4. `reparam_centered`: uses the reparameterized no-intercept design with centered effects.",
    "5. `reparam`: no fixed intercept plus non-centered site/ecoregion effects.",
    "",
    "The files `coefficient_shift_vs_paper.csv`, `convergence_comparison.csv`, and",
    "`model_input_summary.csv` quantify which step changes coefficient values and which step",
    "improves convergence.",
    "",
    "### Known paper-model quirks",
    "",
    "1. **region_for_each_site encoding**: The paper passes",
    "   `region_for_each_site = as.factor(as.character(sites_and_region_df$region))`.",
    "   When JAGS coerces this factor to integer it uses alphabetically-sorted level codes,",
    "   not the original region integers. For 83 regions this scrambles the site-to-ecoregion",
    "   mapping (e.g. region 2 maps to ecoregion[12] because '2' is the 12th level in",
    "   lexicographic order of '1','10','11',...).",
    "",
    "2. **JAGS model string**: The published BUGS code reuses loop index `i` in three loops and",
    "   uses `logit(pi[i])` where `pi` is a JAGS built-in constant. The investigation uses",
    "   distinct loop indices (k, j, i) and renames pi -> prob.",
    "",
    "3. **Intercept identification**: The paper model includes both beta[1] (the fixed intercept",
    "   in `model.matrix(~ ...)`) and `mu_global` in the ecoregion hierarchy. These are two",
    "   baseline terms competing to explain the same location shift.",
    "",
    "4. **Parameterization**: The reparam model keeps the likelihood structure but uses a",
    "   non-centered hierarchy, which is expected to improve mixing for nested random effects.",
    "",
    "## Diagnostics written",
    "",
    "Each variant directory contains `beta_est.csv`, `convergence_diagnostics.csv`,",
    "`model_input_summary.csv`, `model_spec.json`, `logs/run_log.txt`, and a `diagnostics/`",
    "directory with coefficient forests, beta traces, hyperparameter traces, densities,",
    "autocorrelation plots, R-hat plots, and full convergence tables.",
    ""
  )

  writeLines(lines, report_path)
}

# ---- Main ----
set.seed(cfg$seed)
inv_dir <- file.path(cfg$output_root, "investigation")
dir.create(inv_dir, recursive = TRUE, showWarnings = FALSE)

msg("Loading data from %s", cfg$data_dir)
df <- load_model_data_from_pipeline(cfg$data_dir)
msg(
  "Loaded %d rows | %d sites | %d regions | %d ERG codes",
  nrow(df),
  length(unique(df$site)),
  length(unique(df$region)),
  length(unique(df$ERG))
)

div_diag <- diagnose_diversity_alignment(df)
write.csv(
  div_diag$summary,
  file.path(inv_dir, "diversity_alignment.csv"),
  row.names = FALSE
)
msg(
  "Diversity alignment: n_regions=%d, n_erg=%d, paper-vs-dense mismatches=%s",
  div_diag$n_regions,
  div_diag$n_erg,
  div_diag$n_misaligned_paper
)

variants_to_run <- MODEL_VARIANTS
inv_variants <- Sys.getenv("INV_VARIANTS", unset = "")
if (nzchar(inv_variants)) {
  keep <- strsplit(inv_variants, ",", fixed = TRUE)[[1]]
  keep <- trimws(keep)
  keep[keep == "paper_faithful"] <- "paper_reproduction"
  variants_to_run <- MODEL_VARIANTS[intersect(names(MODEL_VARIANTS), keep)]
  if (length(variants_to_run) == 0) {
    stop("INV_VARIANTS did not match any known variant names.")
  }
}

results <- list()
beta_frames <- list()
input_summaries <- list()
init_seed <- cfg$seed

for (nm in names(variants_to_run)) {
  variant <- variants_to_run[[nm]]
  init_seed <- init_seed + 1L
  res <- tryCatch(
    run_variant(variant, df, cfg, init_seed = init_seed),
    error = function(e) {
      msg("ERROR fitting variant '%s': %s", variant$label, conditionMessage(e))
      list(
        variant = variant$label,
        output_dir = file.path(cfg$output_root, variant$output_subdir),
        error = conditionMessage(e),
        spec = list(
          include_intercept = variant$include_intercept,
          parameterization = variant$parameterization,
          diversity_ordering = variant$diversity_ordering,
          r_index = variant$r_index
        ),
        max_rhat = NA_real_
      )
    }
  )
  results[[nm]] <- res
  if (!is.null(res$beta_df)) {
    beta_frames[[nm]] <- res$beta_df
  }
  if (!is.null(res$input_summary)) {
    input_summaries[[nm]] <- res$input_summary
  }
}

if (length(input_summaries) > 0) {
  write.csv(
    dplyr::bind_rows(input_summaries),
    file.path(inv_dir, "model_input_summary.csv"),
    row.names = FALSE
  )
}

if (length(beta_frames) >= 2) {
  comparison_csv <- file.path(inv_dir, "beta_coeff_comparison.csv")
  combined <- dplyr::bind_rows(beta_frames)
  write.csv(combined, comparison_csv, row.names = FALSE)

  reference_variant <- if ("paper_reproduction" %in% combined$variant) {
    "paper_reproduction"
  } else {
    unique(combined$variant)[[1]]
  }
  ref <- combined %>%
    dplyr::filter(.data$variant == reference_variant) %>%
    dplyr::select(variable, paper_mean = mean)
  shifts <- combined %>%
    dplyr::left_join(ref, by = "variable") %>%
    dplyr::mutate(
      mean_shift_vs_paper = .data$mean - .data$paper_mean,
      abs_shift_vs_paper = abs(.data$mean_shift_vs_paper)
    ) %>%
    dplyr::arrange(.data$variable, .data$variant)
  write.csv(
    shifts,
    file.path(inv_dir, "coefficient_shift_vs_paper.csv"),
    row.names = FALSE
  )

  plot_path <- file.path(inv_dir, "beta_coeff_comparison.png")
  plot_beta_coefficient_comparison(
    beta_frames,
    plot_path,
    title = "Beta coefficients: original vs reparam and ablations"
  )
  msg("Comparison plot written to: %s", plot_path)

  # Also copy comparison plot to output root for convenience
  file.copy(
    plot_path,
    file.path(cfg$output_root, "beta_coeff_comparison.png"),
    overwrite = TRUE
  )
}

conv_summaries <- lapply(results, function(res) {
  if (!is.null(res$error)) {
    return(data.frame(
      variant = res$variant,
      status = "error",
      max_rhat = NA_real_,
      n_rhat_gt_1.05 = NA_integer_,
      n_rhat_gt_1.10 = NA_integer_,
      min_neff = NA_real_,
      median_neff = NA_real_,
      error = res$error,
      stringsAsFactors = FALSE
    ))
  }
  conv_path <- file.path(res$output_dir, "convergence_diagnostics.csv")
  if (!file.exists(conv_path)) return(NULL)
  conv <- read.csv(conv_path, row.names = 1, check.names = FALSE)
  rhat <- if ("Rhat" %in% names(conv)) conv$Rhat else rep(NA_real_, nrow(conv))
  neff <- if ("n.eff" %in% names(conv)) conv$n.eff else rep(NA_real_, nrow(conv))
  data.frame(
    variant = res$variant,
    status = "ok",
    max_rhat = max(rhat, na.rm = TRUE),
    n_rhat_gt_1.05 = sum(rhat > 1.05, na.rm = TRUE),
    n_rhat_gt_1.10 = sum(rhat > 1.10, na.rm = TRUE),
    min_neff = min(neff, na.rm = TRUE),
    median_neff = stats::median(neff, na.rm = TRUE),
    error = NA_character_,
    stringsAsFactors = FALSE
  )
})
conv_summaries <- Filter(Negate(is.null), conv_summaries)
if (length(conv_summaries) > 0) {
  write.csv(
    dplyr::bind_rows(conv_summaries),
    file.path(inv_dir, "convergence_comparison.csv"),
    row.names = FALSE
  )
}

write_investigation_report(
  results = results,
  diversity_diag = div_diag,
  report_path = file.path(inv_dir, "investigation_report.md")
)

msg("Done. Primary outputs:")
msg("  paper reproduction: %s", file.path(inv_dir, "01_paper_reproduction"))
msg("  reparam:            %s", file.path(inv_dir, "05_reparam_noncentered"))
msg("  investigation:      %s", inv_dir)
