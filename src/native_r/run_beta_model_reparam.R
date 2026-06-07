#!/usr/bin/env Rscript

# Reparameterized beta-GLMM: full-data fit
#
# Model changes vs original pipeline:
# - No fixed intercept in X (~ 0 + ...); global baseline via mu_global
# - Non-centered hierarchical effects for ecoregion and site
# - Monitors mu_global, sigma, sigma_ecoregion
#
# Usage:
#   Rscript run_beta_model_reparam.R
#   RPARAM_SMOKE=1 Rscript run_beta_model_reparam.R   # short pilot run

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
source("/Users/rt582/Library/CloudStorage/OneDrive-UniversityofCambridge/cambridge/phd/Paper_Conferences/reef_cover_economics/src/native_r/beta_model_reparam_utils.R")

default_data_dir <- "/Users/rt582/Library/CloudStorage/OneDrive-UniversityofCambridge/cambridge/phd/Paper_Conferences/reef_cover_economics/data/sully_og"
default_output_dir <- file.path(default_data_dir, "output", "reparam")

cfg <- list(
  data_dir = Sys.getenv("BETA_DATA_DIR", unset = default_data_dir),
  output_dir = Sys.getenv("BETA_OUTPUT_DIR", unset = default_output_dir),
  seed = 20260529L,
  y_eps = 1e-6,
  use_parallel = TRUE,
  monitor_params = DEFAULT_MONITOR_PARAMS,
  mcmc = list(
    n_chains = 6L,
    n_burnin = 10000L,
    n_iter = 20000L,
    n_thin = 10L
  )
)
# cfg <- list(
#   data_dir = Sys.getenv("BETA_DATA_DIR", unset = default_data_dir),
#   output_dir = Sys.getenv("BETA_OUTPUT_DIR", unset = default_output_dir),
#   seed = 20260529L,
#   y_eps = 1e-6,
#   use_parallel = TRUE,
#   monitor_params = DEFAULT_MONITOR_PARAMS,
#   mcmc = list(
#     n_chains = 3L,
#     n_burnin = 10000L,
#     n_iter = 20000L,
#     n_thin = 10L
#   )
# )

if (is_smoke_mode()) {
  smoke <- smoke_mcmc_settings()
  cfg$mcmc$n_chains <- smoke$n_chains
  cfg$mcmc$n_burnin <- smoke$n_burnin
  cfg$mcmc$n_iter <- smoke$n_iter
  cfg$mcmc$n_thin <- smoke$n_thin
  cfg$use_parallel <- FALSE
}

dir.create(cfg$output_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(cfg$output_dir, "logs"), recursive = TRUE, showWarnings = FALSE)
set.seed(cfg$seed)

msg("Loading data from %s (paper pipeline)", cfg$data_dir)
df <- load_model_data_from_pipeline(cfg$data_dir)
msg("Loaded %d rows | %d sites | %d regions", nrow(df), length(unique(df$site)), length(unique(df$region)))

std <- standardize_vars(df, FEATURE_VARS)
df <- std$data

pkg <- build_jags_data(df, y_eps = cfg$y_eps)
win.data <- pkg$win.data

msg("Fitting reparameterized model (N=%d, K=%d, Nre=%d, R=%d)...", win.data$N, win.data$K, win.data$Nre, win.data$R)
model_path <- file.path(cfg$output_dir, "GLMM_coral_cover_reparam.txt")
fit <- run_jags_fit(win.data, cfg, model_path, init_seed = cfg$seed)

out <- fit$BUGSoutput
sims <- out$sims.list

# ---- Coefficient summary (aligned with paper labels) ----
beta_mat <- sims$beta
beta_div <- sims$beta_diversity

beta_df <- data.frame(
  variable = COEF_LABELS,
  mean = colMeans(beta_mat),
  sd = apply(beta_mat, 2, sd),
  lower_2.5 = apply(beta_mat, 2, stats::quantile, probs = 0.025),
  upper_97.5 = apply(beta_mat, 2, stats::quantile, probs = 0.975),
  lower_25 = apply(beta_mat, 2, stats::quantile, probs = 0.25),
  upper_75 = apply(beta_mat, 2, stats::quantile, probs = 0.75),
  stringsAsFactors = FALSE
)
beta_df <- rbind(
  beta_df,
  data.frame(
    variable = "Diversity",
    mean = mean(beta_div),
    sd = sd(beta_div),
    lower_2.5 = stats::quantile(beta_div, 0.025),
    upper_97.5 = stats::quantile(beta_div, 0.975),
    lower_25 = stats::quantile(beta_div, 0.25),
    upper_75 = stats::quantile(beta_div, 0.75),
    stringsAsFactors = FALSE
  )
)

write.csv(beta_df, file.path(cfg$output_dir, "beta_est_reparam.csv"), row.names = FALSE)

# ---- Convergence diagnostics ----
key_params <- c(
  paste0("beta[", seq_len(win.data$K), "]"),
  "beta_diversity", "mu_global", "theta", "sigma", "sigma_ecoregion"
)
conv <- summarize_convergence(fit, key_params)
write.csv(conv, file.path(cfg$output_dir, "convergence_diagnostics.csv"), row.names = TRUE)
print(conv)

# ---- Hyperparameter summary ----
hyper <- summarize_convergence(fit, c("mu_global", "sigma", "sigma_ecoregion", "theta", "beta_diversity"))
write.csv(hyper, file.path(cfg$output_dir, "hyperparameter_summary.csv"), row.names = TRUE)

# ---- Posterior predictive check ----
if (all(c("Fit", "FitNew") %in% names(sims))) {
  ppc <- data.frame(
    metric = c("Fit_mean", "FitNew_mean", "bayes_p_value"),
    value = c(
      mean(sims$Fit),
      mean(sims$FitNew),
      mean(sims$FitNew > sims$Fit)
    )
  )
  write.csv(ppc, file.path(cfg$output_dir, "posterior_predictive_check.csv"), row.names = FALSE)
  msg("Bayesian p-value: %.3f", ppc$value[3])
}

# ---- Coefficient plot ----
beta_df$color <- "gray"
beta_df$color[beta_df$mean > 0 & beta_df$lower_2.5 >= 0] <- "blue"
beta_df$color[beta_df$mean < 0 & beta_df$upper_97.5 <= 0] <- "red"

p <- ggplot(beta_df, aes(x = reorder(variable, mean), y = mean)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray") +
  geom_errorbar(aes(ymin = lower_2.5, ymax = upper_97.5), width = 0, linewidth = 0.5) +
  geom_errorbar(aes(ymin = lower_25, ymax = upper_75), width = 0, linewidth = 1.3) +
  geom_point(size = 3, shape = 21, fill = beta_df$color, color = "black") +
  coord_flip() +
  theme_gray(base_size = 14) +
  labs(x = "", y = expression(paste("Estimated ", gamma, " coefficients"))) +
  theme(legend.position = "none")

ggsave(file.path(cfg$output_dir, "Beta_coeff_plot_reparam.png"), p, width = 9, height = 7, dpi = 300)

# ---- Convergence diagnostic plots (chains, ACF, posterior histograms) ----
if (!is_smoke_mode()) {
  save_reparam_convergence_plots(
    out = out,
    K = win.data$K,
    log_root = file.path(cfg$output_dir, "logs"),
    prefix = "reparam",
    mcmc_support_path = file.path(cfg$data_dir, "MCMCSupportHighstatV4.R")
  )
}

msg("Done. Outputs written to: %s", cfg$output_dir)
