#!/usr/bin/env Rscript

# Cross-validation for reparameterized beta-GLMM
#
# NOTE: Python port available — prefer:
#   python -m src.models.run_beta_model_cross_validation
#   RCV_SMOKE=1 python -m src.models.run_beta_model_cross_validation
#
# Validation regimes (reproducible via fixed seeds):
# - random_kfold
# - site_group_kfold
# - ecoregion_group_kfold
# - forward_time_blocks
# - spatial_kfold
#
# Usage:
#   Rscript run_beta_model_cross_validation.R
#   RCV_SMOKE=1 Rscript run_beta_model_cross_validation.R   # 1 regime, short MCMC

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

cfg <- list(
  data_dir = "/Users/rt582/Library/CloudStorage/OneDrive-UniversityofCambridge/cambridge/phd/Paper_Conferences/reef_cover_economics/data/sully_og",
  output_dir = "/Users/rt582/Library/CloudStorage/OneDrive-UniversityofCambridge/cambridge/phd/Paper_Conferences/reef_cover_economics/data/sully_og/output/cross_validation",
  seed = 20260529L,
  k_folds = 5L,
  spatial_bins = 4L,
  min_train_rows = 500L,
  y_eps = 1e-6,
  use_parallel = TRUE,
  monitor_params = DEFAULT_MONITOR_PARAMS,
  mcmc = list(
    n_chains = 3L,
    n_burnin = 3000L,
    n_iter = 9000L,
    n_thin = 6L
  ),
  validation_regimes = c(
    "random_kfold",
    "site_group_kfold",
    "ecoregion_group_kfold",
    "forward_time_blocks",
    "spatial_kfold"
  )
)

if (identical(Sys.getenv("RCV_SMOKE"), "1")) {
  cfg$k_folds <- 2L
  cfg$validation_regimes <- c("random_kfold")
  cfg$min_train_rows <- 200L
  cfg$mcmc$n_burnin <- 500L
  cfg$mcmc$n_iter <- 1500L
  cfg$mcmc$n_thin <- 2L
  cfg$output_dir <- file.path(cfg$output_dir, "smoke")
  cfg$use_parallel <- FALSE
}

dir.create(cfg$output_dir, recursive = TRUE, showWarnings = FALSE)
set.seed(cfg$seed)

# ---- Fold constructors ----
make_folds_random <- function(df, k, seed) {
  set.seed(seed)
  idx <- sample(seq_len(nrow(df)))
  fold_id <- ((seq_along(idx) - 1L) %% k) + 1L
  lapply(seq_len(k), function(f) {
    test <- idx[fold_id == f]
    list(name = "random_kfold", fold = f, train_idx = setdiff(seq_len(nrow(df)), test), test_idx = test)
  })
}

make_folds_group <- function(df, group_col, k, seed, regime_name) {
  set.seed(seed)
  groups <- sample(sort(unique(df[[group_col]])))
  fold_id <- ((seq_along(groups) - 1L) %% k) + 1L
  lapply(seq_len(k), function(f) {
    test_groups <- groups[fold_id == f]
    test <- which(df[[group_col]] %in% test_groups)
    list(name = regime_name, fold = f, train_idx = setdiff(seq_len(nrow(df)), test), test_idx = test)
  })
}

make_folds_forward_time <- function(df, k, time_col) {
  times <- sort(unique(df[[time_col]]))
  block_id <- cut(seq_along(times), breaks = k, labels = FALSE)
  time_block <- setNames(block_id, times)
  row_block <- as.integer(time_block[as.character(df[[time_col]])])
  out <- list()
  for (b in 2:k) {
    train <- which(row_block < b)
    test <- which(row_block == b)
    if (length(train) == 0 || length(test) == 0) next
    out[[length(out) + 1L]] <- list(
      name = "forward_time_blocks", fold = b,
      train_idx = train, test_idx = test
    )
  }
  out
}

make_folds_spatial <- function(df, k, n_bins, seed) {
  set.seed(seed)
  lon_bin <- cut(df$Longitude.Degrees, breaks = n_bins, include.lowest = TRUE, labels = FALSE)
  lat_bin <- cut(df$Latitude.Degrees, breaks = n_bins, include.lowest = TRUE, labels = FALSE)
  block <- paste(lon_bin, lat_bin, sep = "_")
  blocks <- sample(unique(block))
  fold_id <- ((seq_along(blocks) - 1L) %% k) + 1L
  lapply(seq_len(k), function(f) {
    test_blocks <- blocks[fold_id == f]
    test <- which(block %in% test_blocks)
    list(name = "spatial_kfold", fold = f, train_idx = setdiff(seq_len(nrow(df)), test), test_idx = test)
  })
}

fit_fold_model <- function(train_df, test_df, cfg_local, fold_tag) {
  std <- standardize_train_test(train_df, test_df)
  tr <- std$train
  te <- std$test

  N_train <- nrow(tr)
  pkg <- build_jags_data(tr, N_for_transform = N_train, y_eps = cfg_local$y_eps)
  X_test <- build_design_matrix(te)

  model_path <- file.path(cfg_local$output_dir, sprintf("model_%s.txt", fold_tag))
  fit <- run_jags_fit(
    pkg$win.data, cfg_local, model_path,
    init_seed = cfg_local$seed + as.integer(abs(stats::runif(1, 1, 1e6)))
  )

  pred <- predict_from_posterior(fit, X_test, te, pkg$dense, y_eps = cfg_local$y_eps)

  summ <- fit$BUGSoutput$summary
  rhat_col <- intersect("Rhat", colnames(summ))
  neff_col <- intersect("n.eff", colnames(summ))
  max_rhat <- if (length(rhat_col) == 1L) max(summ[, rhat_col], na.rm = TRUE) else NA_real_
  min_neff <- if (length(neff_col) == 1L) min(summ[, neff_col], na.rm = TRUE) else NA_real_

  list(
    metrics = data.frame(
      fold_tag = fold_tag,
      n_train = nrow(tr),
      n_test = nrow(te),
      rmse = pred$metrics["rmse"],
      mae = pred$metrics["mae"],
      coverage95 = pred$metrics["coverage95"],
      mean_log_score = pred$metrics["mean_log_score"],
      max_rhat = max_rhat,
      min_neff = min_neff,
      stringsAsFactors = FALSE
    ),
    predictions = cbind(fold_tag = fold_tag, pred$predictions),
    fit_summary = summ
  )
}

# ---- Load data ----
msg("Loading data from %s (paper pipeline)", cfg$data_dir)
df <- load_model_data_from_pipeline(cfg$data_dir)
time_col <- pick_first_existing(df, c("days_since_19811231", "Year", "year"))
if (is.null(time_col)) {
  msg("No explicit time column found; forward_time_blocks will be skipped.")
}
msg("Loaded %d rows | %d sites | %d regions", nrow(df), length(unique(df$site)), length(unique(df$region)))

# ---- Build folds ----
all_folds <- list()
if ("random_kfold" %in% cfg$validation_regimes) {
  all_folds <- c(all_folds, make_folds_random(df, cfg$k_folds, cfg$seed + 11L))
}
if ("site_group_kfold" %in% cfg$validation_regimes) {
  all_folds <- c(all_folds, make_folds_group(df, "site", cfg$k_folds, cfg$seed + 23L, "site_group_kfold"))
}
if ("ecoregion_group_kfold" %in% cfg$validation_regimes) {
  all_folds <- c(all_folds, make_folds_group(df, "region", cfg$k_folds, cfg$seed + 37L, "ecoregion_group_kfold"))
}
if ("forward_time_blocks" %in% cfg$validation_regimes && !is.null(time_col)) {
  all_folds <- c(all_folds, make_folds_forward_time(df, cfg$k_folds, time_col))
}
if ("spatial_kfold" %in% cfg$validation_regimes) {
  all_folds <- c(all_folds, make_folds_spatial(df, cfg$k_folds, cfg$spatial_bins, cfg$seed + 53L))
}
if (length(all_folds) == 0) stop("No validation folds were created.")

fold_manifest <- do.call(rbind, lapply(all_folds, function(f) {
  data.frame(
    fold_tag = paste(f$name, f$fold, sep = "__"),
    regime = f$name, fold = f$fold,
    n_train = length(f$train_idx), n_test = length(f$test_idx),
    stringsAsFactors = FALSE
  )
}))
write.csv(fold_manifest, file.path(cfg$output_dir, "fold_manifest.csv"), row.names = FALSE)

# ---- Run CV ----
all_metrics <- list()
all_predictions <- list()
all_failures <- list()

for (f in all_folds) {
  fold_tag <- paste(f$name, f$fold, sep = "__")
  msg("Running %s", fold_tag)

  train_df <- df[f$train_idx, , drop = FALSE]
  test_df <- df[f$test_idx, , drop = FALSE]

  if (nrow(train_df) < cfg$min_train_rows || nrow(test_df) == 0) {
    msg("Skipping %s (train=%d, test=%d).", fold_tag, nrow(train_df), nrow(test_df))
    next
  }

  res <- tryCatch(
    fit_fold_model(train_df, test_df, cfg, fold_tag),
    error = function(e) {
      all_failures[[fold_tag]] <<- data.frame(
        fold_tag = fold_tag, regime = f$name, fold = f$fold,
        n_train = nrow(train_df), n_test = nrow(test_df),
        error = conditionMessage(e), stringsAsFactors = FALSE
      )
      msg("Fold %s failed: %s", fold_tag, conditionMessage(e))
      NULL
    }
  )
  if (is.null(res)) next

  res$metrics$regime <- f$name
  res$metrics$fold <- f$fold
  all_metrics[[fold_tag]] <- res$metrics
  all_predictions[[fold_tag]] <- res$predictions
  write.csv(res$fit_summary, file.path(cfg$output_dir, sprintf("summary_%s.csv", fold_tag)), row.names = TRUE)
}

metrics_df <- bind_rows(all_metrics)
pred_df <- bind_rows(all_predictions)
if (nrow(metrics_df) == 0) stop("No folds were successfully fit.")

if (length(all_failures) > 0) {
  write.csv(bind_rows(all_failures), file.path(cfg$output_dir, "validation_failures.csv"), row.names = FALSE)
}

write.csv(metrics_df, file.path(cfg$output_dir, "validation_metrics_by_fold.csv"), row.names = FALSE)
write.csv(pred_df, file.path(cfg$output_dir, "validation_predictions.csv"), row.names = FALSE)

metrics_regime <- metrics_df %>%
  group_by(regime) %>%
  summarise(
    folds = n(),
    rmse_mean = mean(rmse, na.rm = TRUE),
    rmse_sd = sd(rmse, na.rm = TRUE),
    mae_mean = mean(mae, na.rm = TRUE),
    coverage95_mean = mean(coverage95, na.rm = TRUE),
    mean_log_score = mean(mean_log_score, na.rm = TRUE),
    max_rhat_mean = mean(max_rhat, na.rm = TRUE),
    min_neff_mean = mean(min_neff, na.rm = TRUE),
    .groups = "drop"
  )
write.csv(metrics_regime, file.path(cfg$output_dir, "validation_metrics_by_regime.csv"), row.names = FALSE)

p <- ggplot(metrics_df, aes(x = regime, y = rmse)) +
  geom_boxplot(outlier.alpha = 0.4) +
  coord_flip() +
  theme_minimal(base_size = 12) +
  labs(title = "RMSE by validation regime", x = "", y = "RMSE")
ggsave(file.path(cfg$output_dir, "rmse_by_regime.png"), p, width = 9, height = 5, dpi = 300)

msg("Done. Outputs written to: %s", cfg$output_dir)
