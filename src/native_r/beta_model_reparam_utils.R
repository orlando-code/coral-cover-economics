# Shared helpers for reparameterized beta-GLMM scripts.
# Sourced by run_beta_model_reparam.R and run_beta_model_cross_validation.R

FEATURE_VARS <- c(
  "lat", "Depth", "Human_pop", "Cyclone", "SST_mean", "SSTA_Mean",
  "SSTA_min", "SSTA_freqstdev", "SSTA_dhwmax", "TSA_max",
  "TSA_freqstdev", "Turbidity_mean", "Historical_SST_max"
)

COEF_LABELS <- c(
  "Latitude", "Depth", "Human_pop", "Cyclone", "SST_mean", "SSTA_Mean",
  "SSTA_min", "SSTA_freqstdev", "SSTA_dhwmax", "TSA_max",
  "TSA_freqstdev", "Turbidity_mean", "Historical_SST_max"
)

DEFAULT_MONITOR_PARAMS <- c(
  "beta", "beta_diversity", "mu_global", "sigma", "sigma_ecoregion",
  "theta", "a", "ecoregion", "Fit", "FitNew"
)

msg <- function(...) cat(sprintf(...), "\n")

is_smoke_mode <- function() {
  identical(Sys.getenv("BETA_SMOKE"), "1") || identical(Sys.getenv("RPARAM_SMOKE"), "1")
}

smoke_mcmc_settings <- function() {
  list(
    n_chains = 2L,
    n_burnin = as.integer(Sys.getenv("BETA_SMOKE_N_BURNIN", unset = "5")),
    n_iter = as.integer(Sys.getenv("BETA_SMOKE_N_ITER", unset = "20")),
    n_thin = 1L
  )
}

validate_mcmc_cfg <- function(mcmc_cfg) {
  # R2jags: n.iter is total iterations (burn-in + post-burn-in), not post-burn-in alone.
  n_iter <- as.integer(mcmc_cfg$n_iter)
  n_burnin <- as.integer(mcmc_cfg$n_burnin)
  n_thin <- as.integer(mcmc_cfg$n_thin)
  n_chains <- as.integer(mcmc_cfg$n_chains)

  if (length(n_iter) != 1L || !is.finite(n_iter) || n_iter <= 0L) {
    stop("mcmc$n_iter must be a positive integer (R2jags total iterations, including burn-in).")
  }
  if (length(n_burnin) != 1L || !is.finite(n_burnin) || n_burnin < 0L) {
    stop("mcmc$n_burnin must be a non-negative integer.")
  }
  if (n_iter <= n_burnin) {
    stop(sprintf(
      paste(
        "mcmc$n_iter (%d) must be greater than mcmc$n_burnin (%d).",
        "In R2jags, n.iter counts total iterations; post-burn-in = n.iter - n.burnin."
      ),
      n_iter,
      n_burnin
    ))
  }
  if (length(n_thin) != 1L || !is.finite(n_thin) || n_thin <= 0L) {
    stop("mcmc$n_thin must be a positive integer.")
  }
  if (length(n_chains) != 1L || !is.finite(n_chains) || n_chains <= 0L) {
    stop("mcmc$n_chains must be a positive integer.")
  }
  invisible(TRUE)
}

resolve_jags_parallel <- function(default = TRUE) {
  flag <- tolower(trimws(Sys.getenv("BETA_JAGS_PARALLEL", unset = "auto")))
  if (flag %in% c("1", "true", "yes")) {
    return(TRUE)
  }
  if (flag %in% c("0", "false", "no")) {
    return(FALSE)
  }
  isTRUE(default)
}

assign_jags_parallel_data <- function(win.data) {
  # jags.parallel exports names(data) from envir; BUGS data (notably `re`) must exist there.
  for (var in names(win.data)) {
    assign(var, win.data[[var]], envir = .GlobalEnv)
  }
  invisible(names(win.data))
}

call_r2jags <- function(jags_args, use_parallel, progress_bar = "text") {
  if (!isTRUE(use_parallel)) {
    jags_args$progress.bar <- progress_bar
    return(do.call(R2jags::jags, jags_args))
  }

  jags_args$envir <- .GlobalEnv
  if (is.character(jags_args$model.file) && length(jags_args$model.file) == 1L) {
    jags_args$model.file <- normalizePath(jags_args$model.file, wins = FALSE, mustWork = TRUE)
  }
  do.call(R2jags::jags.parallel, jags_args)
}

maybe_subsample_smoke <- function(df, seed = 20260529L) {
  if (!is_smoke_mode()) {
    return(df)
  }
  max_sites <- as.integer(Sys.getenv("BETA_SMOKE_MAX_SITES", unset = "12"))
  site_col <- if ("site" %in% names(df)) {
    "site"
  } else if ("Reef_ID" %in% names(df)) {
    "Reef_ID"
  } else {
    stop("maybe_subsample_smoke(): need a site or Reef_ID column.")
  }

  sites <- sort(unique(df[[site_col]]))
  if (length(sites) > max_sites) {
    set.seed(seed)
    sites <- sort(sample(sites, max_sites))
  }
  out <- df[df[[site_col]] %in% sites, , drop = FALSE]
  region_col <- if ("region" %in% names(out)) "region" else if ("Ecoregion" %in% names(out)) "Ecoregion" else NA_character_
  n_regions <- if (is.na(region_col)) NA_integer_ else length(unique(out[[region_col]]))
  msg(
    "BETA_SMOKE subsample: %d rows | %d sites | %s regions",
    nrow(out),
    length(unique(out[[site_col]])),
    if (is.na(n_regions)) "?" else as.character(n_regions)
  )
  out
}

pick_first_existing <- function(df, candidates) {
  hits <- candidates[candidates %in% names(df)]
  if (length(hits) == 0) return(NULL)
  hits[[1]]
}

safe_mean_sd <- function(x) {
  m <- mean(x, na.rm = TRUE)
  s <- stats::sd(x, na.rm = TRUE)
  if (!is.finite(s) || s == 0) s <- 1
  list(mean = m, sd = s)
}

standardize_vars <- function(df, vars, stats_tbl = NULL) {
  if (is.null(stats_tbl)) {
    stats_tbl <- lapply(vars, function(v) safe_mean_sd(df[[v]]))
    names(stats_tbl) <- vars
  }
  out <- df
  for (v in vars) {
    out[[v]] <- (df[[v]] - stats_tbl[[v]]$mean) / stats_tbl[[v]]$sd
  }
  list(data = out, stats = stats_tbl)
}

standardize_train_test <- function(train_df, test_df, vars = FEATURE_VARS) {
  std <- standardize_vars(train_df, vars)
  test_std <- standardize_vars(test_df, vars, stats_tbl = std$stats)$data
  list(train = std$data, test = test_std, stats = std$stats)
}

make_dense_site_region <- function(df) {
  site_vals <- sort(unique(df$site))
  site_dense_map <- setNames(seq_along(site_vals), as.character(site_vals))
  df$site_dense <- as.integer(site_dense_map[as.character(df$site)])

  site_region <- df %>%
    dplyr::distinct(.data$site, .data$region) %>%
    dplyr::arrange(.data$site)

  region_per_site_n <- site_region %>%
    dplyr::count(.data$site, name = "n_regions")
  if (any(region_per_site_n$n_regions != 1L)) {
    bad_sites <- region_per_site_n$site[region_per_site_n$n_regions != 1L]
    stop(sprintf(
      "Some sites map to multiple regions. Example sites: %s",
      paste(head(bad_sites, 5), collapse = ", ")
    ))
  }

  region_vals <- sort(unique(site_region$region))
  region_dense_map <- setNames(seq_along(region_vals), as.character(region_vals))
  df$region_dense <- as.integer(region_dense_map[as.character(df$region)])

  site_level <- df %>%
    dplyr::distinct(.data$site_dense, .data$region_dense) %>%
    dplyr::arrange(.data$site_dense)

  stopifnot(all(site_level$site_dense == seq_len(nrow(site_level))))

  list(
    data = df,
    region_for_each_site = site_level$region_dense,
    Nre = length(site_vals),
    R = length(region_vals),
    site_dense_map = site_dense_map,
    region_dense_map = region_dense_map
  )
}

build_region_diversity <- function(df, R_expected) {
  reg_div <- df %>%
    dplyr::group_by(.data$region_dense) %>%
    dplyr::summarise(
      diversity = mean(.data$diversity.standardized, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    dplyr::arrange(.data$region_dense)

  if (nrow(reg_div) != R_expected) {
    stop(sprintf(
      "Region-diversity length mismatch: got %d, expected %d",
      nrow(reg_div), R_expected
    ))
  }
  if (any(!is.finite(reg_div$diversity))) {
    bad <- reg_div$region_dense[!is.finite(reg_div$diversity)]
    stop(sprintf("Non-finite diversity for regions: %s", paste(bad, collapse = ", ")))
  }
  reg_div$diversity
}

make_model_string <- function() {
  "
model{
  # Fixed effects (no intercept term in X)
  for (k in 1:K) {
    beta[k] ~ dnorm(0, 0.0001)
  }

  # Ecoregion effects (non-centered)
  for (z in 1:R) {
    eco_raw[z] ~ dnorm(0, 1)
    eco_mean[z] <- mu_global + beta_diversity * diversity[z]
    ecoregion[z] <- eco_mean[z] + sigma_ecoregion * eco_raw[z]
  }

  # Site effects nested in ecoregion (non-centered)
  for (j in 1:Nre) {
    site_raw[j] ~ dnorm(0, 1)
    a[j] <- ecoregion[region_for_each_site[j]] + sigma * site_raw[j]
  }

  # Hyperpriors
  mu_global ~ dnorm(0, 0.0001)
  beta_diversity ~ dnorm(0, 0.0001)

  num ~ dnorm(0, 0.0016)
  denom ~ dnorm(0, 1)
  sigma <- abs(num / denom)

  num_ecoregion ~ dnorm(0, 0.0016)
  denom_ecoregion ~ dnorm(0, 1)
  sigma_ecoregion <- abs(num_ecoregion / denom_ecoregion)

  numtheta ~ dnorm(0, 0.0016)
  denomtheta ~ dnorm(0, 1)
  theta <- abs(numtheta / denomtheta)

  # Likelihood
  for (i in 1:N) {
    Y[i] ~ dbeta(shape1[i], shape2[i])
    shape1[i] <- theta * pi[i]
    shape2[i] <- theta * (1 - pi[i])

    logit(pi_raw[i]) <- eta[i]
    pi[i] <- max(1.0E-6, min(0.999999, pi_raw[i]))
    eta[i] <- inprod(beta[], X[i,]) + a[re[i]]

    ExpY[i] <- pi[i]
    VarY[i] <- pi[i] * (1 - pi[i]) / (theta + 1)
    PRes[i] <- (Y[i] - ExpY[i]) / sqrt(VarY[i])

    YNew[i] ~ dbeta(shape1[i], shape2[i])
    PResNew[i] <- (YNew[i] - ExpY[i]) / sqrt(VarY[i])
    D[i] <- pow(PRes[i], 2)
    DNew[i] <- pow(PResNew[i], 2)
  }

  Fit <- sum(D[1:N])
  FitNew <- sum(DNew[1:N])
}
"
}

make_centered_glmm_inits <- function(K, Nre, seed = 20260529L) {
  # Self-contained inits for original centered beta-GLMM (monitors vector `a`).
  e <- new.env(parent = baseenv())
  e$K <- as.integer(K)
  e$Nre <- as.integer(Nre)
  e$seed <- as.integer(seed)
  e$chain_idx <- 0L

  function() {
    e$chain_idx <- e$chain_idx + 1L
    ch <- e$chain_idx
    set.seed(as.integer(e$seed + ch * 1000L))
    list(
      beta = stats::rnorm(e$K, 0, 0.1),
      beta_diversity = stats::rnorm(1, 0, 0.1),
      a = stats::rnorm(e$Nre, 0, 0.1),
      num = stats::rnorm(1, 0, 25),
      denom = stats::rnorm(1, 0, 1),
      numtheta = stats::rnorm(1, 0, 25),
      denomtheta = stats::rnorm(1, 0, 1),
      num_ecoregion = stats::rnorm(1, 0, 25),
      denom_ecoregion = stats::rnorm(1, 0, 1),
      .RNG.name = "base::Wichmann-Hill",
      .RNG.seed = as.integer(e$seed + ch * 1000L)
    )
  }
}

run_centered_jags_fit <- function(
  win.data,
  model_path,
  params,
  K,
  Nre,
  mcmc_cfg,
  init_seed = 20260529L,
  progress_bar = "text"
) {
  validate_mcmc_cfg(mcmc_cfg)
  inits <- make_centered_glmm_inits(K = K, Nre = Nre, seed = init_seed)
  if (isTRUE(mcmc_cfg$use_parallel)) {
    assign_jags_parallel_data(win.data)
  }
  jags_args <- list(
    data = win.data,
    inits = inits,
    parameters.to.save = params,
    model.file = model_path,
    n.thin = as.integer(mcmc_cfg$n_thin),
    n.chains = as.integer(mcmc_cfg$n_chains),
    n.burnin = as.integer(mcmc_cfg$n_burnin),
    n.iter = as.integer(mcmc_cfg$n_iter),
    DIC = TRUE
  )
  call_r2jags(jags_args, use_parallel = mcmc_cfg$use_parallel, progress_bar = progress_bar)
}

source_beta_model_utils <- function(data_dir = NULL) {
  if (exists("make_centered_glmm_inits", mode = "function")) {
    return(invisible(TRUE))
  }
  native_r <- Sys.getenv("BETA_NATIVE_R", unset = "")
  if (!nzchar(native_r) && !is.null(data_dir)) {
    native_r <- normalizePath(
      file.path(data_dir, "..", "..", "src", "native_r"),
      wins = FALSE,
      mustWork = FALSE
    )
  }
  utils_path <- file.path(native_r, "beta_model_reparam_utils.R")
  if (!file.exists(utils_path)) {
    stop(sprintf("Cannot find beta_model_reparam_utils.R at %s", utils_path))
  }
  source(utils_path, local = FALSE)
  invisible(TRUE)
}

make_inits <- function(seed, Nre, R, K) {
  # Self-contained init function for jags.parallel workers.
  # All state lives in `e` (parent = baseenv()); closure must not reach cfg/global.
  e <- new.env(parent = baseenv())
  e$seed <- as.integer(seed)
  e$Nre <- as.integer(Nre)
  e$R <- as.integer(R)
  e$K <- as.integer(K)
  e$chain_idx <- 0L

  function() {
    e$chain_idx <- e$chain_idx + 1L
    ch <- e$chain_idx
    set.seed(as.integer(e$seed + ch * 1000L))

    nonzero_draw <- function(sd = 1, min_abs = 0.1) {
      v <- stats::rnorm(1, 0, sd)
      if (!is.finite(v) || abs(v) < min_abs) {
        v <- ifelse(v < 0, -min_abs, min_abs)
      }
      as.numeric(v)
    }

    list(
      beta = stats::rnorm(e$K, 0, 0.05),
      beta_diversity = stats::rnorm(1, 0, 0.05),
      mu_global = stats::rnorm(1, 0, 0.05),
      eco_raw = stats::rnorm(e$R, 0, 0.1),
      site_raw = stats::rnorm(e$Nre, 0, 0.1),
      num = stats::rnorm(1, 0, 5),
      denom = nonzero_draw(sd = 1, min_abs = 0.1),
      num_ecoregion = stats::rnorm(1, 0, 5),
      denom_ecoregion = nonzero_draw(sd = 1, min_abs = 0.1),
      numtheta = stats::rnorm(1, 0, 5),
      denomtheta = nonzero_draw(sd = 1, min_abs = 0.1),
      .RNG.name = "base::Wichmann-Hill",
      .RNG.seed = as.integer(e$seed + ch * 1000L)
    )
  }
}

build_design_matrix <- function(df) {
  X <- model.matrix(
    ~ 0 + lat + Depth + Human_pop + Cyclone + SST_mean + SSTA_Mean +
      SSTA_min + SSTA_freqstdev + SSTA_dhwmax + TSA_max + TSA_freqstdev +
      Turbidity_mean + Historical_SST_max,
    data = df
  )
  if (any(!is.finite(X))) {
    stop("Non-finite values in design matrix.")
  }
  X
}

transform_beta_response <- function(cover, N, y_eps = 1e-6) {
  y <- (cover * (N - 1) + 0.5) / N
  pmax(pmin(y, 1 - y_eps), y_eps)
}

build_jags_data <- function(df, N_for_transform = NULL, y_eps = 1e-6) {
  if (is.null(N_for_transform)) N_for_transform <- nrow(df)

  dens <- make_dense_site_region(df)
  d <- dens$data
  diversity_vec <- build_region_diversity(d, dens$R)
  X <- build_design_matrix(d)
  Y <- transform_beta_response(d$Average_coral_cover, N_for_transform, y_eps)

  win.data <- list(
    Y = Y,
    N = nrow(d),
    X = X,
    K = ncol(X),
    re = d$site_dense,
    R = dens$R,
    Nre = dens$Nre,
    region_for_each_site = dens$region_for_each_site,
    diversity = diversity_vec
  )

  list(
    win.data = win.data,
    data = d,
    dense = dens,
    X = X
  )
}

load_prepared_data <- function(data_path) {
  df <- read.csv(data_path, header = TRUE, stringsAsFactors = FALSE)
  df$row_id <- seq_len(nrow(df))

  if (!"Ecoregion" %in% names(df)) {
    eco_col <- pick_first_existing(df, c("Ecoregion.x", "Ecoregion.y", "ERG"))
    if (is.null(eco_col)) stop("Could not find an ecoregion column.")
    df$Ecoregion <- df[[eco_col]]
  }

  required_cols <- c(
    "Average_coral_cover", "Latitude.Degrees", "Longitude.Degrees",
    "site", "region", "diversity.standardized", FEATURE_VARS
  )
  missing_cols <- setdiff(required_cols, names(df))
  if (length(missing_cols) > 0) {
    stop(sprintf("Missing required columns: %s", paste(missing_cols, collapse = ", ")))
  }

  df$lat <- abs(df$Latitude.Degrees)

  if (max(df$Average_coral_cover, na.rm = TRUE) > 1.5) {
    msg("Detected Average_coral_cover likely in percent scale; converting to proportion by /100.")
    df$Average_coral_cover <- df$Average_coral_cover / 100
  }

  df <- df %>%
    dplyr::filter(
      !is.na(.data$Average_coral_cover),
      .data$Average_coral_cover > 0,
      !is.na(.data$Depth),
      !is.na(.data$Human_pop),
      !is.na(.data$Cyclone),
      !is.na(.data$SST_mean),
      !is.na(.data$SSTA_Mean),
      !is.na(.data$SSTA_min),
      !is.na(.data$SSTA_freqstdev),
      !is.na(.data$SSTA_dhwmax),
      !is.na(.data$TSA_max),
      !is.na(.data$TSA_freqstdev),
      !is.na(.data$Turbidity_mean),
      !is.na(.data$Historical_SST_max),
      !is.na(.data$site),
      !is.na(.data$region),
      !is.na(.data$diversity.standardized)
    )

  df
}

run_jags_fit <- function(win.data, cfg, model_path, init_seed = NULL) {
  if (is.null(init_seed)) {
    init_seed <- cfg$seed
  }
  validate_mcmc_cfg(cfg$mcmc)

  writeLines(make_model_string(), con = model_path)

  inits <- make_inits(
    seed = init_seed,
    Nre = win.data$Nre,
    R = win.data$R,
    K = win.data$K
  )
  if (!is.function(inits)) {
    stop("make_inits() must return a function for R2jags compatibility.")
  }

  jags_args <- list(
    data = win.data,
    inits = inits,
    parameters.to.save = cfg$monitor_params,
    model.file = model_path,
    n.chains = cfg$mcmc$n_chains,
    n.burnin = cfg$mcmc$n_burnin,
    n.iter = cfg$mcmc$n_iter,
    n.thin = cfg$mcmc$n_thin,
    DIC = TRUE
  )
  if (isTRUE(cfg$use_parallel)) {
    assign_jags_parallel_data(win.data)
  }
  call_r2jags(
    jags_args,
    use_parallel = cfg$use_parallel,
    progress_bar = if (is_smoke_mode()) "none" else "text"
  )
}

summarize_convergence <- function(fit, param_rows = NULL) {
  summ <- fit$BUGSoutput$summary
  if (!is.null(param_rows)) {
    rows <- intersect(param_rows, rownames(summ))
    summ <- summ[rows, , drop = FALSE]
  }
  cols <- intersect(c("mean", "sd", "2.5%", "97.5%", "Rhat", "n.eff"), colnames(summ))
  summ[, cols, drop = FALSE]
}

predict_from_posterior <- function(fit, X_test, test_df, dense_info, y_eps = 1e-6) {
  sims <- fit$BUGSoutput$sims.list
  beta_draws <- sims$beta
  theta_draws <- sims$theta
  mu_global_draws <- sims$mu_global
  beta_diversity_draws <- sims$beta_diversity
  a_draws <- sims$a
  eco_draws <- sims$ecoregion

  n_draws <- nrow(beta_draws)
  n_test <- nrow(test_df)
  fixed_part <- beta_draws %*% t(X_test)

  test_site_dense <- as.integer(dense_info$site_dense_map[as.character(test_df$site)])
  test_region_dense <- as.integer(dense_info$region_dense_map[as.character(test_df$region)])

  hier_part <- matrix(NA_real_, nrow = n_draws, ncol = n_test)
  for (i in seq_len(n_test)) {
    s_i <- test_site_dense[i]
    r_i <- test_region_dense[i]
    d_i <- test_df$diversity.standardized[i]

    if (!is.na(s_i)) {
      hier_part[, i] <- a_draws[, s_i]
    } else if (!is.na(r_i)) {
      hier_part[, i] <- eco_draws[, r_i]
    } else if (is.finite(d_i)) {
      hier_part[, i] <- mu_global_draws + beta_diversity_draws * d_i
    } else {
      hier_part[, i] <- mu_global_draws
    }
  }

  pi_draw <- plogis(fixed_part + hier_part)
  pred_mean <- colMeans(pi_draw)
  pred_lo <- apply(pi_draw, 2, stats::quantile, probs = 0.025, na.rm = TRUE)
  pred_hi <- apply(pi_draw, 2, stats::quantile, probs = 0.975, na.rm = TRUE)

  y_obs <- transform_beta_response(test_df$Average_coral_cover, nrow(test_df), y_eps)

  rmse <- sqrt(mean((y_obs - pred_mean)^2, na.rm = TRUE))
  mae <- mean(abs(y_obs - pred_mean), na.rm = TRUE)
  coverage95 <- mean(y_obs >= pred_lo & y_obs <= pred_hi, na.rm = TRUE)

  eps <- 1e-9
  y_clamp <- pmax(pmin(y_obs, 1 - eps), eps)
  log_scores <- vapply(seq_len(n_test), function(i) {
    p_i <- pmax(pmin(pi_draw[, i], 1 - eps), eps)
    sh1 <- theta_draws * p_i
    sh2 <- theta_draws * (1 - p_i)
    dens_i <- stats::dbeta(y_clamp[i], sh1, sh2)
    log(mean(pmax(dens_i, eps)))
  }, numeric(1))

  list(
    metrics = c(rmse = rmse, mae = mae, coverage95 = coverage95, mean_log_score = mean(log_scores)),
    predictions = data.frame(
      row_id = test_df$row_id,
      y_obs_beta = y_obs,
      y_pred_mean = pred_mean,
      y_pred_lo95 = pred_lo,
      y_pred_hi95 = pred_hi,
      stringsAsFactors = FALSE
    )
  )
}

filter_model_ready_rows <- function(df) {
  if (max(df$Average_coral_cover, na.rm = TRUE) > 1.5) {
    msg("Detected Average_coral_cover likely in percent scale; converting to proportion by /100.")
    df$Average_coral_cover <- df$Average_coral_cover / 100
  }

  df %>%
    dplyr::filter(
      !is.na(.data$Average_coral_cover),
      .data$Average_coral_cover > 0,
      !is.na(.data$Depth),
      !is.na(.data$Human_pop),
      !is.na(.data$Cyclone),
      !is.na(.data$SST_mean),
      !is.na(.data$SSTA_Mean),
      !is.na(.data$SSTA_min),
      !is.na(.data$SSTA_freqstdev),
      !is.na(.data$SSTA_dhwmax),
      !is.na(.data$TSA_max),
      !is.na(.data$TSA_freqstdev),
      !is.na(.data$Turbidity_mean),
      !is.na(.data$Historical_SST_max),
      !is.na(.data$diversity.standardized)
    )
}

assign_paper_site_region_indices <- function(df) {
  # Match my_1_run_the_beta_model.Rmd after filtering:
  # data$Reef_ID <- as.factor(as.character(as.factor(data$Reef_ID)))
  # site = as.numeric(as.factor(Reef_ID)); region = as.numeric(as.factor(Ecoregion)).
  df$Reef_ID <- as.factor(as.character(as.factor(df$Reef_ID)))
  df$site <- as.integer(as.factor(df$Reef_ID))
  df$region <- as.integer(as.factor(df$Ecoregion))

  site_region <- df %>%
    dplyr::distinct(.data$site, .data$Ecoregion, .data$region)
  site_region_n <- site_region %>%
    dplyr::count(.data$site, name = "n_regions")
  if (any(site_region_n$n_regions != 1L)) {
    bad <- site_region_n$site[site_region_n$n_regions != 1L]
    stop(sprintf(
      "Paper-style indexing found sites in multiple ecoregions. Example site ids: %s",
      paste(head(bad, 5), collapse = ", ")
    ))
  }

  df
}

load_model_data_from_pipeline <- function(
  data_dir,
  diversity_lookup_path = NULL,
  shapefiles_dir = NULL,
  index_source = c("paper_factor", "data_for_maps")
) {
  index_source <- match.arg(index_source)
  if (is.null(diversity_lookup_path)) {
    diversity_lookup_path <- file.path(data_dir, "data_for_maps.csv")
  }
  if (is.null(shapefiles_dir)) {
    shapefiles_dir <- file.path(data_dir, "shapefiles")
  }

  data_path <- file.path(data_dir, "data.csv")
  if (!file.exists(data_path)) {
    stop(sprintf("Missing raw data file: %s", data_path))
  }
  if (!file.exists(diversity_lookup_path)) {
    stop(sprintf("Missing diversity lookup file: %s", diversity_lookup_path))
  }
  if (!requireNamespace("terra", quietly = TRUE)) {
    stop("Package 'terra' is required for load_model_data_from_pipeline()")
  }

  eco_shp <- file.path(shapefiles_dir, "ecoregion_exportPolygon.shp")
  if (!file.exists(eco_shp)) {
    stop(sprintf("Missing ecoregion shapefile: %s", eco_shp))
  }

  msg("Loading raw data from %s", data_path)
  df <- read.csv(data_path, header = TRUE, stringsAsFactors = FALSE)
  df$row_id <- seq_len(nrow(df))
  df$reef <- df$Reef_ID
  n0 <- nrow(df)

  df$lat <- abs(df$Latitude.Degrees)
  df$lon <- df$Longitude.Degrees
  df$Longitude <- df$lon

  ECO <- terra::vect(eco_shp)
  coral_cover_points <- terra::vect(df, geom = c("lon", "lat"), crs = "EPSG:4326")
  coral_cover_points <- terra::project(coral_cover_points, terra::crs(ECO))
  eco_extracted <- terra::extract(ECO, coral_cover_points)

  df <- as.data.frame(df)
  df$ERG <- eco_extracted$ERG
  df$Ecoregion <- eco_extracted$Ecoregion
  if ("ID" %in% names(df)) {
    df$ID <- NULL
  }

  df <- df[
    !is.na(df$Average_coral_cover) &
      df$Average_coral_cover > 0 &
      !is.na(df$SST_mean) &
      !is.na(df$SSTA_stdev) &
      !is.na(df$SSTA_freqmax) &
      !is.na(df$SSTA_freqmean) &
      !is.na(df$Turbidity_mean) &
      df$Turbidity_mean < 0.35 &
      !is.na(df$Cyclone) &
      !is.na(df$Depth) &
      !is.na(df$Historical_SST_max) &
      !is.na(df$sst_mean_rcp85_2100),
  ]
  msg("After Rmd spatial/filter steps: %d/%d rows", nrow(df), n0)

  data_for_maps <- read.csv(diversity_lookup_path, header = TRUE, stringsAsFactors = FALSE)
  trusted <- data_for_maps %>%
    dplyr::transmute(
      Reef_ID = as.character(.data$Reef_ID),
      trusted_ecoregion = as.character(.data$Ecoregion.x),
      trusted_erg = as.character(.data$ERG),
      diversity.standardized = as.numeric(.data$diversity.standardized),
      trusted_site = as.integer(.data$site),
      trusted_region = as.integer(.data$region)
    ) %>%
    dplyr::distinct(.data$Reef_ID, .keep_all = TRUE)

  df <- dplyr::left_join(df, trusted, by = "Reef_ID")
  missing_trusted <- is.na(df$trusted_ecoregion)
  if (any(missing_trusted)) {
    msg(
      "No trusted data_for_maps mapping for %d pre-final-filter row(s). Example reef_id(s): %s",
      sum(missing_trusted),
      paste(head(df$Reef_ID[missing_trusted], 5), collapse = ", ")
    )
  }

  mapped_trusted <- !missing_trusted
  disagreement <- mapped_trusted &
    !is.na(df$Ecoregion) &
    !is.na(df$trusted_ecoregion) &
    trimws(tolower(as.character(df$Ecoregion))) !=
    trimws(tolower(as.character(df$trusted_ecoregion)))
  if (any(disagreement)) {
    examples <- unique(df[disagreement, c("Reef_ID", "Ecoregion", "trusted_ecoregion")])
    msg(
      "data_for_maps.csv ecoregion mapping overrides %d row(s) from the spatial join.",
      sum(disagreement)
    )
    print(utils::head(examples, 8))
  }

  df$Ecoregion[mapped_trusted] <- df$trusted_ecoregion[mapped_trusted]
  df$ERG[mapped_trusted] <- df$trusted_erg[mapped_trusted]
  df$trusted_ecoregion <- NULL
  df$trusted_erg <- NULL

  df <- filter_model_ready_rows(df)
  if (index_source == "paper_factor") {
    df <- assign_paper_site_region_indices(df)
  } else {
    df$site <- df$trusted_site
    df$region <- df$trusted_region
  }
  df <- df[!is.na(df$site) & !is.na(df$region), , drop = FALSE]
  df$trusted_site <- NULL
  df$trusted_region <- NULL

  df <- maybe_subsample_smoke(df)
  msg(
    "Model-ready dataset: %d rows | %d sites | %d regions | index_source=%s",
    nrow(df),
    length(unique(df$site)),
    length(unique(df$region)),
    index_source
  )
  df
}

save_reparam_convergence_plots <- function(
  out,
  K,
  log_root,
  prefix = "reparam",
  mcmc_support_path = NULL
) {
  if (!dir.exists(log_root)) {
    dir.create(log_root, recursive = TRUE, showWarnings = FALSE)
  }
  if (is.null(mcmc_support_path) || !file.exists(mcmc_support_path)) {
    stop(sprintf("MCMCSupportHighstatV4.R not found at %s", mcmc_support_path))
  }
  source(mcmc_support_path)
  if (!requireNamespace("lattice", quietly = TRUE)) {
    stop("Package 'lattice' is required for convergence plots.")
  }

  param_names <- dimnames(out$sims.array)[[3]]
  beta_names <- grep("^beta\\[", param_names, value = TRUE)
  sel <- c(beta_names, "beta_diversity", "mu_global", "theta", "sigma", "sigma_ecoregion")
  sel <- intersect(sel, param_names)

  chains_plot <- MyBUGSChains(out, vars = sel)
  png(
    file.path(log_root, paste0(prefix, "_chains.png")),
    width = 9, height = 7, units = "in", res = 300
  )
  print(chains_plot)
  dev.off()

  acf_plot <- MyBUGSACF(Output = out, SelectedVar = sel)
  png(
    file.path(log_root, paste0(prefix, "_acf.png")),
    width = 9, height = 7, units = "in", res = 300
  )
  print(acf_plot)
  dev.off()

  hist_plot <- MyBUGSHist(Output = out, SelectedVar = sel)
  png(
    file.path(log_root, paste0(prefix, "_hist.png")),
    width = 9, height = 7, units = "in", res = 300
  )
  print(hist_plot)
  dev.off()
  invisible(TRUE)
}

# ---- Beta-model investigation helpers (original vs reparam) ----

ORIGINAL_CENTERED_MONITOR_PARAMS <- c(
  "beta", "beta_diversity", "a", "theta", "PRes", "Fit", "FitNew",
  "YNew", "ecoregion", "sigma", "sigma_ecoregion", "mu_global"
)

make_original_centered_model_string <- function() {
  "
    model{
    #1A. Priors
    for (k in 1:K) { beta[k]  ~ dnorm(0, 0.0001) }
    for (j in 1:Nre) {a[j] ~ dnorm(ecoregion[region_for_each_site[j]], tau)}
    # Hierarchical effects
    for(z in 1:R){
    ecoregion[z] ~ dnorm(g[z],tau_ecoregion)
    g[z] <- mu_global + beta_diversity*diversity[z]
    }
    mu_global ~ dnorm(0, 0.0001)
    beta_diversity ~ dnorm(0, 0.0001)

    #1B.
    num   ~ dnorm(0, 0.0016)
    denom ~ dnorm(0, 1)
    sigma <- abs(num / denom)

    num_ecoregion   ~ dnorm(0, 0.0016)
    denom_ecoregion ~ dnorm(0, 1)
    sigma_ecoregion <- abs(num_ecoregion / denom_ecoregion)

    #1C. half-Cauchy(25) prior tau
    tau   <- 1 / (sigma * sigma)
    numtheta   ~ dnorm(0, 0.0016)
    denomtheta ~ dnorm(0, 1)
    theta <- abs(numtheta / denomtheta)

    tau_ecoregion   <- 1 / (sigma_ecoregion * sigma_ecoregion)

#2. Likelihood
    for (i in 1:N){
      Y[i]       ~ dbeta(shape1[i], shape2[i])
      shape1[i] <- theta * prob[i]
      shape2[i] <- theta * (1 - prob[i])

      eta[i] <- inprod(beta[], X[i,]) + a[re[i]]
      logit(prob[i]) <- eta[i]

      ExpY[i] <- prob[i]
      VarY[i] <- prob[i] * (1 - prob[i])  / (theta + 1)
      PRes[i] <- (Y[i] - ExpY[i]) / sqrt(VarY[i])

      YNew[i]   ~ dbeta(shape1[i], shape2[i])
      PResNew[i] <- (YNew[i] - ExpY[i]) / sqrt(VarY[i])
      D[i]       <- pow(PRes[i], 2)
      DNew[i]    <- pow(PResNew[i], 2)
  }
    Fit         <- sum(D[1:N])
    FitNew      <- sum(DNew[1:N])
}
"
}

make_reparam_centered_model_string <- function() {
  "
model{
  for (k in 1:K) {
    beta[k] ~ dnorm(0, 0.0001)
  }

  for (j in 1:Nre) {
    a[j] ~ dnorm(ecoregion[region_for_each_site[j]], tau)
  }
  for (z in 1:R) {
    ecoregion[z] ~ dnorm(g[z], tau_ecoregion)
    g[z] <- mu_global + beta_diversity * diversity[z]
  }

  mu_global ~ dnorm(0, 0.0001)
  beta_diversity ~ dnorm(0, 0.0001)

  num ~ dnorm(0, 0.0016)
  denom ~ dnorm(0, 1)
  sigma <- abs(num / denom)
  tau <- 1 / (sigma * sigma)

  num_ecoregion ~ dnorm(0, 0.0016)
  denom_ecoregion ~ dnorm(0, 1)
  sigma_ecoregion <- abs(num_ecoregion / denom_ecoregion)
  tau_ecoregion <- 1 / (sigma_ecoregion * sigma_ecoregion)

  numtheta ~ dnorm(0, 0.0016)
  denomtheta ~ dnorm(0, 1)
  theta <- abs(numtheta / denomtheta)

  for (i in 1:N) {
    Y[i] ~ dbeta(shape1[i], shape2[i])
    shape1[i] <- theta * pi[i]
    shape2[i] <- theta * (1 - pi[i])

    logit(pi_raw[i]) <- eta[i]
    pi[i] <- max(1.0E-6, min(0.999999, pi_raw[i]))
    eta[i] <- inprod(beta[], X[i,]) + a[re[i]]

    ExpY[i] <- pi[i]
    VarY[i] <- pi[i] * (1 - pi[i]) / (theta + 1)
    PRes[i] <- (Y[i] - ExpY[i]) / sqrt(VarY[i])

    YNew[i] ~ dbeta(shape1[i], shape2[i])
    PResNew[i] <- (YNew[i] - ExpY[i]) / sqrt(VarY[i])
    D[i] <- pow(PRes[i], 2)
    DNew[i] <- pow(PResNew[i], 2)
  }

  Fit <- sum(D[1:N])
  FitNew <- sum(DNew[1:N])
}
"
}

build_original_design_matrix <- function(df, include_intercept = TRUE) {
  f <- if (include_intercept) {
    ~ lat + Depth + Human_pop + Cyclone + SST_mean + SSTA_Mean +
      SSTA_min + SSTA_freqstdev + SSTA_dhwmax + TSA_max + TSA_freqstdev +
      Turbidity_mean + Historical_SST_max
  } else {
    ~ 0 + lat + Depth + Human_pop + Cyclone + SST_mean + SSTA_Mean +
      SSTA_min + SSTA_freqstdev + SSTA_dhwmax + TSA_max + TSA_freqstdev +
      Turbidity_mean + Historical_SST_max
  }
  X <- model.matrix(f, data = df)
  if (any(!is.finite(X))) {
    stop("Non-finite values in original design matrix.")
  }
  X
}

build_diversity_vector <- function(
  df,
  ordering = c("region_dense", "region_factor", "paper_alphabetical", "erg_sorted"),
  R_expected,
  region_for_each_site = NULL
) {
  ordering <- match.arg(ordering)
  dens <- make_dense_site_region(df)
  d <- dens$data

  if (ordering == "region_dense") {
    return(build_region_diversity(d, R_expected))
  }

  if (ordering == "region_factor") {
    reg_div <- d %>%
      dplyr::distinct(.data$region, .data$diversity.standardized) %>%
      dplyr::arrange(.data$region)
    if (nrow(reg_div) != R_expected) {
      stop(sprintf(
        "region_factor diversity length %d != R_expected %d",
        nrow(reg_div), R_expected
      ))
    }
    return(reg_div$diversity.standardized)
  }

  if (ordering == "paper_alphabetical") {
    reg_div <- d %>%
      dplyr::distinct(.data$Ecoregion, .data$diversity.standardized) %>%
      dplyr::arrange(.data$Ecoregion)
    vec <- reg_div$diversity.standardized
    if (length(vec) < R_expected) {
      stop(sprintf(
        "paper_alphabetical diversity has %d regions but R_expected=%d",
        length(vec), R_expected
      ))
    }
    return(vec[seq_len(R_expected)])
  }

  if (ordering == "erg_sorted") {
    if (!"ERG" %in% names(d)) {
      stop("ERG column required for erg_sorted diversity ordering.")
    }
    reg_div <- d %>%
      dplyr::distinct(.data$ERG, .data$diversity.standardized) %>%
      dplyr::arrange(as.integer(.data$ERG))
    if (nrow(reg_div) != R_expected) {
      stop(sprintf(
        "erg_sorted diversity length %d != R_expected %d",
        nrow(reg_div), R_expected
      ))
    }
    return(reg_div$diversity.standardized)
  }

  stop("Unknown diversity ordering.")
}

resolve_region_count <- function(df, r_index = c("n_regions", "n_erg")) {
  r_index <- match.arg(r_index)
  if (r_index == "n_erg") {
    if (!"ERG" %in% names(df)) {
      stop("ERG column required when r_index='n_erg'.")
    }
    return(length(unique(df$ERG)))
  }
  length(unique(df$region))
}

build_investigation_jags_data <- function(
  df,
  include_intercept = FALSE,
  parameterization = c("noncentered", "centered"),
  diversity_ordering = "region_dense",
  r_index = "n_regions",
  y_eps = 1e-6,
  N_for_transform = NULL,
  region_encoding = c("dense", "paper_factor")
) {
  parameterization  <- match.arg(parameterization)
  region_encoding   <- match.arg(region_encoding)
  if (is.null(N_for_transform)) {
    N_for_transform <- nrow(df)
  }

  std <- standardize_vars(df, FEATURE_VARS)
  d <- std$data
  dens <- make_dense_site_region(d)
  d <- dens$data

  if (r_index == "n_erg") {
    R <- resolve_region_count(d, "n_erg")
    region_for_each_site <- dens$region_for_each_site
  } else {
    R <- dens$R
    region_for_each_site <- dens$region_for_each_site
  }

  # Optionally replicate the paper's as.factor(as.character(region)) encoding.
  # The paper passed region integers as a character factor, so JAGS received
  # lexicographic codes: "1" -> 1, "10" -> 2, "11" -> 3, ..., "2" -> 12, etc.
  # This scrambles which ecoregion prior each site actually receives.
  if (region_encoding == "paper_factor") {
    all_levels           <- sort(as.character(seq_len(dens$R)))
    region_for_each_site <- as.integer(
      factor(as.character(region_for_each_site), levels = all_levels)
    )
  }

  diversity_vec <- build_diversity_vector(
    d,
    ordering = diversity_ordering,
    R_expected = R,
    region_for_each_site = region_for_each_site
  )

  include_intercept <- isTRUE(include_intercept)
  X <- build_original_design_matrix(d, include_intercept = include_intercept)
  Y <- transform_beta_response(d$Average_coral_cover, N_for_transform, y_eps)

  win.data <- list(
    Y = Y,
    N = nrow(d),
    X = X,
    K = ncol(X),
    re = d$site_dense,
    R = R,
    Nre = dens$Nre,
    region_for_each_site = region_for_each_site,
    diversity = diversity_vec
  )

  list(
    win.data = win.data,
    data = d,
    dense = dens,
    X = X,
    spec = list(
      include_intercept = include_intercept,
      parameterization  = parameterization,
      diversity_ordering = diversity_ordering,
      r_index           = r_index,
      region_encoding   = region_encoding
    )
  )
}

diagnose_diversity_alignment <- function(df) {
  dens <- make_dense_site_region(df)
  d <- dens$data
  R_regions <- dens$R
  R_erg <- if ("ERG" %in% names(d)) length(unique(d$ERG)) else NA_integer_

  dense_div <- build_region_diversity(d, R_regions)
  region_factor_div <- build_diversity_vector(
    d, ordering = "region_factor", R_expected = R_regions
  )
  paper_div <- if (!is.na(R_erg)) {
    build_diversity_vector(
      d, ordering = "paper_alphabetical", R_expected = R_erg
    )
  } else {
    rep(NA_real_, R_regions)
  }

  region_lookup <- d %>%
    dplyr::distinct(.data$region_dense, .data$region, .data$Ecoregion, .data$ERG) %>%
    dplyr::arrange(.data$region_dense)

  out <- data.frame(
    region_dense = seq_len(R_regions),
    region = region_lookup$region,
    Ecoregion = region_lookup$Ecoregion,
    ERG = if ("ERG" %in% names(region_lookup)) region_lookup$ERG else NA,
    diversity_region_dense = dense_div,
    diversity_region_factor = region_factor_div,
    stringsAsFactors = FALSE
  )
  out$diversity_dense_vs_factor_diff <- out$diversity_region_dense - out$diversity_region_factor

  list(
    summary = out,
    n_regions = R_regions,
    n_erg = R_erg,
    n_misaligned_paper = if (!is.na(R_erg)) {
      n_cmp <- min(R_erg, R_regions)
      sum(abs(paper_div[seq_len(n_cmp)] - dense_div[seq_len(n_cmp)]) > 1e-8)
    } else {
      NA_integer_
    },
    paper_alphabetical_head = utils::head(paper_div, 10)
  )
}

extract_beta_summary <- function(
  fit,
  include_intercept = FALSE,
  drop_intercept_from_plot = TRUE
) {
  sims <- fit$BUGSoutput$sims.list
  beta_mat <- sims$beta
  beta_div <- sims$beta_diversity

  if (isTRUE(include_intercept)) {
    plot_labels <- if (drop_intercept_from_plot) COEF_LABELS else c("Intercept", COEF_LABELS)
    beta_for_plot <- if (drop_intercept_from_plot) beta_mat[, -1, drop = FALSE] else beta_mat
  } else {
    plot_labels <- COEF_LABELS
    beta_for_plot <- beta_mat
  }

  beta_df <- data.frame(
    variable = plot_labels,
    mean = colMeans(beta_for_plot),
    sd = apply(beta_for_plot, 2, sd),
    lower_2.5 = apply(beta_for_plot, 2, stats::quantile, probs = 0.025),
    upper_97.5 = apply(beta_for_plot, 2, stats::quantile, probs = 0.975),
    lower_25 = apply(beta_for_plot, 2, stats::quantile, probs = 0.25),
    upper_75 = apply(beta_for_plot, 2, stats::quantile, probs = 0.75),
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

  intercept_row <- NULL
  if (isTRUE(include_intercept)) {
    intercept_row <- data.frame(
      variable = "Intercept",
      mean = mean(beta_mat[, 1]),
      sd = sd(beta_mat[, 1]),
      lower_2.5 = stats::quantile(beta_mat[, 1], 0.025),
      upper_97.5 = stats::quantile(beta_mat[, 1], 0.975),
      lower_25 = stats::quantile(beta_mat[, 1], 0.25),
      upper_75 = stats::quantile(beta_mat[, 1], 0.75),
      stringsAsFactors = FALSE
    )
  }

  list(beta_df = beta_df, intercept = intercept_row)
}

write_beta_fit_outputs <- function(
  fit,
  pkg,
  output_dir,
  variant_label,
  include_intercept = FALSE,
  monitor_params = NULL,
  start_time = NULL,
  mcmc_cfg = list()
) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  dir.create(file.path(output_dir, "logs"), recursive = TRUE, showWarnings = FALSE)

  beta_summary <- extract_beta_summary(fit, include_intercept = include_intercept)
  write.csv(
    beta_summary$beta_df,
    file.path(output_dir, "beta_est.csv"),
    row.names = FALSE
  )
  if (!is.null(beta_summary$intercept)) {
    write.csv(
      beta_summary$intercept,
      file.path(output_dir, "intercept_beta.csv"),
      row.names = FALSE
    )
  }

  # Convergence summary for key scalar + all beta params
  K <- pkg$win.data$K
  conv_params <- c(
    paste0("beta[", seq_len(K), "]"),
    "beta_diversity", "mu_global", "theta", "sigma", "sigma_ecoregion"
  )
  if (!is.null(monitor_params)) {
    conv_params <- unique(c(conv_params, monitor_params))
  }
  conv <- summarize_convergence(fit, conv_params)
  write.csv(conv, file.path(output_dir, "convergence_diagnostics.csv"), row.names = TRUE)

  spec_path <- file.path(output_dir, "model_spec.json")
  spec_out <- c(variant = variant_label, as.list(pkg$spec))
  if (requireNamespace("jsonlite", quietly = TRUE)) {
    jsonlite::write_json(spec_out, spec_path, auto_unbox = TRUE, pretty = TRUE)
  } else {
    writeLines(
      paste(names(spec_out), spec_out, sep = ": ", collapse = "\n"),
      spec_path
    )
  }

  # Per-variant diagnostics subdirectory
  write_variant_diagnostics(
    fit = fit,
    output_dir = output_dir,
    variant_label = variant_label,
    pkg = pkg,
    include_intercept = include_intercept
  )

  # Run log
  write_run_log(
    output_dir = output_dir,
    variant_label = variant_label,
    start_time = start_time %||% Sys.time(),
    end_time = Sys.time(),
    fit = fit,
    mcmc_cfg = mcmc_cfg
  )

  beta_summary$beta_df$variant <- variant_label
  invisible(beta_summary$beta_df)
}

run_investigation_jags_fit <- function(
  pkg,
  cfg,
  output_dir,
  model_kind = c("reparam", "reparam_centered", "original_centered"),
  init_seed = NULL
) {
  model_kind <- match.arg(model_kind)
  model_path <- file.path(output_dir, "GLMM_coral_cover.txt")
  init_seed <- init_seed %||% cfg$seed

  if (model_kind == "reparam") {
    fit <- run_jags_fit(pkg$win.data, cfg, model_path, init_seed = init_seed)
    monitor_params <- cfg$monitor_params
  } else if (model_kind == "reparam_centered") {
    writeLines(make_reparam_centered_model_string(), model_path)
    monitor_params <- ORIGINAL_CENTERED_MONITOR_PARAMS
    fit <- run_centered_jags_fit(
      win.data = pkg$win.data,
      model_path = model_path,
      params = monitor_params,
      K = pkg$win.data$K,
      Nre = pkg$win.data$Nre,
      mcmc_cfg = cfg$mcmc,
      init_seed = init_seed,
      progress_bar = if (is_smoke_mode()) "none" else "text"
    )
  } else {
    writeLines(make_original_centered_model_string(), model_path)
    monitor_params <- ORIGINAL_CENTERED_MONITOR_PARAMS
    fit <- run_centered_jags_fit(
      win.data = pkg$win.data,
      model_path = model_path,
      params = monitor_params,
      K = pkg$win.data$K,
      Nre = pkg$win.data$Nre,
      mcmc_cfg = cfg$mcmc,
      init_seed = init_seed,
      progress_bar = if (is_smoke_mode()) "none" else "text"
    )
  }

  list(fit = fit, monitor_params = monitor_params)
}

plot_beta_coefficient_comparison <- function(
  beta_frames,
  output_path,
  title = "Beta coefficient comparison across model variants"
) {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("ggplot2 is required for plot_beta_coefficient_comparison().")
  }

  combined <- dplyr::bind_rows(beta_frames)
  combined$variable <- factor(
    combined$variable,
    levels = c(COEF_LABELS, "Diversity")
  )

  p <- ggplot2::ggplot(
    combined,
    ggplot2::aes(
      x = variable,
      y = mean,
      colour = variant,
      group = variant
    )
  ) +
    ggplot2::geom_hline(yintercept = 0, linetype = "dashed", colour = "gray50") +
    ggplot2::geom_errorbar(
      ggplot2::aes(ymin = lower_2.5, ymax = upper_97.5),
      width = 0.15,
      linewidth = 0.4,
      alpha = 0.85,
      position = ggplot2::position_dodge(width = 0.55)
    ) +
    ggplot2::geom_point(
      size = 2.5,
      position = ggplot2::position_dodge(width = 0.55)
    ) +
    ggplot2::coord_flip() +
    ggplot2::theme_gray(base_size = 13) +
    ggplot2::labs(
      title = title,
      x = "",
      y = expression(paste("Estimated ", gamma, " coefficients")),
      colour = "Model variant"
    ) +
    ggplot2::theme(legend.position = "bottom")

  ggplot2::ggsave(output_path, p, width = 12, height = 8, dpi = 300)
  invisible(p)
}

# ---- Per-variant diagnostics helpers ----

# Build a long-format draws data frame from BUGSoutput$sims.array.
# sims.array: [n_draws_per_chain, n_chains, n_params]
make_draws_df <- function(bugs_output, params) {
  arr <- bugs_output$sims.array
  if (is.null(arr) || length(dim(arr)) < 3L) return(NULL)
  pnames <- dimnames(arr)[[3]]
  sel    <- intersect(params, pnames)
  if (length(sel) == 0L) return(NULL)

  n_draws  <- dim(arr)[1]
  n_chains <- dim(arr)[2]

  do.call(rbind, lapply(sel, function(pn) {
    idx <- which(pnames == pn)
    data.frame(
      iteration = rep(seq_len(n_draws), times = n_chains),
      chain     = rep(paste0("chain", seq_len(n_chains)), each = n_draws),
      param     = pn,
      value     = as.vector(arr[, , idx]),
      stringsAsFactors = FALSE
    )
  }))
}

safe_ggsave <- function(p, path, width, height, dpi = 150) {
  tryCatch(
    ggplot2::ggsave(path, p, width = width, height = height, dpi = dpi, limitsize = FALSE),
    error = function(e) msg("Warning: could not save %s: %s", basename(path), conditionMessage(e))
  )
}

save_trace_png <- function(draws_df, path, variant_label, ncol = 4) {
  nparams <- length(unique(draws_df$param))
  nrows   <- ceiling(nparams / ncol)
  p <- ggplot2::ggplot(
      draws_df,
      ggplot2::aes(x = .data$iteration, y = .data$value, colour = .data$chain)
    ) +
    ggplot2::geom_line(alpha = 0.7, linewidth = 0.25) +
    ggplot2::facet_wrap(~ param, scales = "free_y", ncol = ncol) +
    ggplot2::scale_colour_brewer(palette = "Set1") +
    ggplot2::theme_bw(base_size = 9) +
    ggplot2::labs(
      title = sprintf("Trace plots: %s", variant_label),
      x = "Post-burnin iteration", y = NULL, colour = NULL
    ) +
    ggplot2::theme(
      legend.position = "bottom",
      strip.text      = ggplot2::element_text(size = 7),
      axis.text.x     = ggplot2::element_text(size = 6)
    )
  safe_ggsave(p, path, width = ncol * 3.2, height = max(3, nrows * 2.5))
}

save_density_png <- function(draws_df, path, variant_label, ncol = 4) {
  nparams <- length(unique(draws_df$param))
  nrows   <- ceiling(nparams / ncol)
  p <- ggplot2::ggplot(
      draws_df,
      ggplot2::aes(x = .data$value, fill = .data$chain, colour = .data$chain)
    ) +
    ggplot2::geom_density(alpha = 0.3) +
    ggplot2::facet_wrap(~ param, scales = "free", ncol = ncol) +
    ggplot2::scale_colour_brewer(palette = "Set1") +
    ggplot2::scale_fill_brewer(palette = "Set1") +
    ggplot2::theme_bw(base_size = 9) +
    ggplot2::labs(
      title = sprintf("Posterior densities: %s", variant_label),
      x = NULL, y = "Density", colour = NULL, fill = NULL
    ) +
    ggplot2::theme(
      legend.position = "bottom",
      strip.text      = ggplot2::element_text(size = 7)
    )
  safe_ggsave(p, path, width = ncol * 3.2, height = max(3, nrows * 2.5))
}

save_autocorr_png <- function(draws_df, path, variant_label, max_lag = 30, ncol = 4) {
  n_draws <- max(draws_df$iteration)
  if (n_draws < 4L) return(invisible(NULL))
  max_lag  <- min(max_lag, n_draws - 1L)
  ci_band  <- stats::qnorm(0.975) / sqrt(n_draws)

  acf_rows <- lapply(
    split(draws_df, list(draws_df$param, draws_df$chain), drop = TRUE),
    function(df) {
      if (nrow(df) < 5L) return(NULL)
      ac <- tryCatch(stats::acf(df$value, lag.max = max_lag, plot = FALSE), error = function(e) NULL)
      if (is.null(ac)) return(NULL)
      data.frame(
        param = df$param[1], chain = df$chain[1],
        lag   = as.integer(ac$lag),
        acf   = as.numeric(ac$acf),
        stringsAsFactors = FALSE
      )
    }
  )
  acf_df <- do.call(rbind, Filter(Negate(is.null), acf_rows))
  if (is.null(acf_df) || nrow(acf_df) == 0L) return(invisible(NULL))

  nparams <- length(unique(acf_df$param))
  nrows   <- ceiling(nparams / ncol)
  p <- ggplot2::ggplot(
      acf_df,
      ggplot2::aes(x = .data$lag, y = .data$acf, colour = .data$chain)
    ) +
    ggplot2::geom_hline(yintercept = 0, colour = "gray50") +
    ggplot2::geom_hline(
      yintercept = c(-ci_band, ci_band),
      linetype = "dashed", colour = "steelblue", alpha = 0.6
    ) +
    ggplot2::geom_segment(ggplot2::aes(xend = .data$lag, yend = 0), linewidth = 0.5, alpha = 0.8) +
    ggplot2::facet_wrap(~ param, scales = "free_y", ncol = ncol) +
    ggplot2::scale_colour_brewer(palette = "Set1") +
    ggplot2::theme_bw(base_size = 9) +
    ggplot2::labs(
      title = sprintf("Autocorrelation: %s", variant_label),
      x = "Lag", y = "ACF", colour = NULL
    ) +
    ggplot2::theme(
      legend.position = "bottom",
      strip.text      = ggplot2::element_text(size = 7)
    )
  safe_ggsave(p, path, width = ncol * 3.2, height = max(3, nrows * 2.5))
}

save_rhat_png <- function(fit, path, variant_label, params_sel) {
  summ      <- fit$BUGSoutput$summary
  available <- intersect(params_sel, rownames(summ))
  if (length(available) == 0L || !"Rhat" %in% colnames(summ)) return(invisible(NULL))
  rhat_vals <- summ[available, "Rhat"]
  rhat_df   <- data.frame(
    param = names(rhat_vals),
    rhat  = as.numeric(rhat_vals),
    stringsAsFactors = FALSE
  )
  rhat_df <- rhat_df[is.finite(rhat_df$rhat), , drop = FALSE]
  if (nrow(rhat_df) == 0L) return(invisible(NULL))

  rhat_df <- rhat_df[order(rhat_df$rhat, decreasing = TRUE), ]
  rhat_df$param  <- factor(rhat_df$param, levels = rev(rhat_df$param))
  rhat_df$status <- factor(
    ifelse(rhat_df$rhat > 1.1,  "poor (>1.10)",
    ifelse(rhat_df$rhat > 1.05, "marginal (1.05-1.10)", "good (<=1.05)")),
    levels = c("good (<=1.05)", "marginal (1.05-1.10)", "poor (>1.10)")
  )
  p <- ggplot2::ggplot(rhat_df, ggplot2::aes(x = .data$param, y = .data$rhat, fill = .data$status)) +
    ggplot2::geom_col() +
    ggplot2::scale_fill_manual(
      values = c("good (<=1.05)" = "steelblue", "marginal (1.05-1.10)" = "orange", "poor (>1.10)" = "red")
    ) +
    ggplot2::geom_hline(yintercept = 1.1,  colour = "red",    linetype = "dashed") +
    ggplot2::geom_hline(yintercept = 1.05, colour = "orange", linetype = "dashed") +
    ggplot2::coord_flip() +
    ggplot2::theme_bw(base_size = 9) +
    ggplot2::labs(
      title = sprintf("Gelman-Rubin R-hat: %s", variant_label),
      x = NULL, y = "R-hat", fill = "Convergence"
    ) +
    ggplot2::theme(axis.text.y = ggplot2::element_text(size = 7), legend.position = "bottom")
  nparams <- nrow(rhat_df)
  safe_ggsave(p, path, width = 8, height = max(4, nparams * 0.22 + 2))
}

plot_single_variant_coeff <- function(beta_df, variant_label, output_path) {
  df <- beta_df
  df$fill_color <- ifelse(df$lower_2.5 > 0, "blue",
                   ifelse(df$upper_97.5 < 0, "red", "white"))
  df <- df[order(df$mean), , drop = FALSE]
  df$variable <- factor(df$variable, levels = df$variable)

  p <- ggplot2::ggplot(df, ggplot2::aes(x = .data$variable, y = .data$mean)) +
    ggplot2::geom_errorbar(
      ggplot2::aes(ymin = .data$lower_2.5, ymax = .data$upper_97.5), width = 0
    ) +
    ggplot2::geom_errorbar(
      ggplot2::aes(ymin = .data$lower_25, ymax = .data$upper_75), width = 0, linewidth = 1.3
    ) +
    ggplot2::geom_point(pch = 21, size = 3, fill = df$fill_color, colour = "black") +
    ggplot2::coord_flip() +
    ggplot2::theme_grey(base_size = 16) +
    ggplot2::geom_hline(yintercept = 0, linetype = "dashed", colour = "gray") +
    ggplot2::labs(
      title = sprintf("Beta coefficients: %s", variant_label),
      y = expression(paste("Estimated ", gamma, " coefficients")), x = ""
    )

  grDevices::png(output_path, width = 2700, height = 2000, res = 300)
  print(p)
  grDevices::dev.off()
  invisible(p)
}

write_variant_diagnostics <- function(
  fit,
  output_dir,
  variant_label,
  pkg = NULL,
  include_intercept = FALSE
) {
  diag_dir <- file.path(output_dir, "diagnostics")
  dir.create(diag_dir, recursive = TRUE, showWarnings = FALSE)

  # Full convergence summary (all monitored params)
  summ      <- fit$BUGSoutput$summary
  conv_cols <- intersect(c("mean", "sd", "2.5%", "97.5%", "Rhat", "n.eff"), colnames(summ))
  write.csv(
    summ[, conv_cols, drop = FALSE],
    file.path(diag_dir, "convergence_full.csv"),
    row.names = TRUE
  )

  # Coefficient forest plot (paper style)
  beta_summary <- extract_beta_summary(fit, include_intercept = include_intercept)
  plot_single_variant_coeff(
    beta_df       = beta_summary$beta_df,
    variant_label = variant_label,
    output_path   = file.path(diag_dir, "coeff_forest.png")
  )

  K           <- if (!is.null(pkg)) pkg$win.data$K else 14L
  beta_params <- paste0("beta[", seq_len(K), "]")
  other_key   <- c("beta_diversity", "mu_global", "sigma", "sigma_ecoregion", "theta")
  all_key     <- c(beta_params, other_key)

  arr <- fit$BUGSoutput$sims.array
  if (!is.null(arr) && length(dim(arr)) == 3L) {
    beta_df  <- make_draws_df(fit$BUGSoutput, beta_params)
    other_df <- make_draws_df(fit$BUGSoutput, other_key)

    if (!is.null(beta_df) && nrow(beta_df) > 0L) {
      save_trace_png(beta_df,
        file.path(diag_dir, "trace_betas.png"),   variant_label, ncol = 4)
      save_density_png(beta_df,
        file.path(diag_dir, "density_betas.png"),  variant_label, ncol = 4)
      save_autocorr_png(beta_df,
        file.path(diag_dir, "autocorr_betas.png"), variant_label, ncol = 4)
    }
    if (!is.null(other_df) && nrow(other_df) > 0L) {
      save_trace_png(other_df,
        file.path(diag_dir, "trace_other.png"),    variant_label, ncol = 3)
      save_density_png(other_df,
        file.path(diag_dir, "density_other.png"),   variant_label, ncol = 3)
      save_autocorr_png(other_df,
        file.path(diag_dir, "autocorr_other.png"),  variant_label, ncol = 3)
    }
  }

  save_rhat_png(fit, file.path(diag_dir, "rhat_summary.png"), variant_label, all_key)

  invisible(diag_dir)
}

write_run_log <- function(
  output_dir,
  variant_label,
  start_time,
  end_time = NULL,
  fit = NULL,
  mcmc_cfg = list()
) {
  log_dir <- file.path(output_dir, "logs")
  dir.create(log_dir, recursive = TRUE, showWarnings = FALSE)

  end_time <- end_time %||% Sys.time()
  elapsed_min <- as.numeric(difftime(end_time, start_time, units = "mins"))

  n_chains  <- mcmc_cfg$n_chains  %||% NA_integer_
  n_iter    <- mcmc_cfg$n_iter    %||% NA_integer_
  n_burnin  <- mcmc_cfg$n_burnin  %||% NA_integer_
  n_thin    <- mcmc_cfg$n_thin    %||% NA_integer_
  post_per_chain <- if (!is.na(n_iter) && !is.na(n_burnin) && !is.na(n_thin)) {
    (as.integer(n_iter) - as.integer(n_burnin)) %/% as.integer(n_thin)
  } else NA_integer_
  total_samples <- if (!is.na(n_chains) && !is.na(post_per_chain)) {
    as.integer(n_chains) * post_per_chain
  } else NA_integer_

  lines <- c(
    paste0("variant      : ", variant_label),
    paste0("finished_at  : ", format(end_time, "%Y-%m-%d %H:%M:%S")),
    paste0("elapsed_min  : ", round(elapsed_min, 2)),
    paste0("n_chains     : ", n_chains),
    paste0("n_iter       : ", n_iter),
    paste0("n_burnin     : ", n_burnin),
    paste0("n_thin       : ", n_thin),
    paste0("post_per_chain: ", post_per_chain),
    paste0("total_samples: ", total_samples)
  )

  if (!is.null(fit)) {
    summ <- fit$BUGSoutput$summary
    rhat_vals <- if ("Rhat" %in% colnames(summ)) summ[, "Rhat"] else NULL
    neff_vals  <- if ("n.eff" %in% colnames(summ)) summ[, "n.eff"] else NULL
    if (!is.null(rhat_vals) && any(is.finite(rhat_vals))) {
      lines <- c(
        lines,
        paste0("max_rhat     : ", round(max(rhat_vals, na.rm = TRUE), 4)),
        paste0("rhat_gt_1.10 : ", sum(rhat_vals > 1.10, na.rm = TRUE)),
        paste0("rhat_gt_1.05 : ", sum(rhat_vals > 1.05, na.rm = TRUE))
      )
    }
    if (!is.null(neff_vals)) {
      lines <- c(
        lines,
        paste0("median_neff  : ", round(stats::median(neff_vals, na.rm = TRUE), 1)),
        paste0("min_neff     : ", round(min(neff_vals, na.rm = TRUE), 1))
      )
    }
  }

  writeLines(lines, file.path(log_dir, "run_log.txt"))
  invisible(file.path(log_dir, "run_log.txt"))
}

`%||%` <- function(x, y) if (is.null(x)) y else x
