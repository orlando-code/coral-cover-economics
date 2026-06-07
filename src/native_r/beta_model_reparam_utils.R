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
  inits <- make_centered_glmm_inits(K = K, Nre = Nre, seed = init_seed)
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
  if (isTRUE(mcmc_cfg$use_parallel)) {
    jags_args$envir <- .GlobalEnv
    do.call(R2jags::jags.parallel, jags_args)
  } else {
    jags_args$progress.bar <- progress_bar
    do.call(R2jags::jags, jags_args)
  }
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

  if (isTRUE(cfg$use_parallel)) {
    # jags.parallel exports win.data names to .GlobalEnv on workers only;
    # inits must be self-contained (see make_inits).
    R2jags::jags.parallel(
      data = win.data,
      inits = inits,
      parameters.to.save = cfg$monitor_params,
      model.file = model_path,
      n.chains = cfg$mcmc$n_chains,
      n.burnin = cfg$mcmc$n_burnin,
      n.iter = cfg$mcmc$n_iter,
      n.thin = cfg$mcmc$n_thin,
      DIC = TRUE,
      envir = .GlobalEnv
    )
  } else {
    R2jags::jags(
      data = win.data,
      inits = inits,
      parameters.to.save = cfg$monitor_params,
      model.file = model_path,
      n.chains = cfg$mcmc$n_chains,
      n.burnin = cfg$mcmc$n_burnin,
      n.iter = cfg$mcmc$n_iter,
      n.thin = cfg$mcmc$n_thin,
      DIC = TRUE,
      progress.bar = "text"
    )
  }
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
      !is.na(.data$site),
      !is.na(.data$region),
      !is.na(.data$diversity.standardized)
    )
}

load_model_data_from_pipeline <- function(
  data_dir,
  diversity_lookup_path = NULL,
  shapefiles_dir = NULL
) {
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
      site = as.integer(.data$site),
      region = as.integer(.data$region)
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
  df <- maybe_subsample_smoke(df)
  msg(
    "Model-ready dataset: %d rows | %d sites | %d regions",
    nrow(df),
    length(unique(df$site)),
    length(unique(df$region))
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
