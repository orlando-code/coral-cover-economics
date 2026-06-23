#!/usr/bin/env Rscript
# Investigation of site/region indexing bugs in the Sully et al. 2022 paper (https://doi.org/10.1111/gcb.16083) implementation of 1_run_the_beta_model.Rmd (https://github.com/InstituteForGlobalEcology/Present-and-future-bright-and-dark-spots-for-coral-reefs-through-climate-change/blob/main/1_run_the_beta_model.Rmd)
# where lines are referenced, e.g. L204, these refer to the line numbers in the 1_run_the_beta_model.Rmd file.
#
# File contents:
# - Dummy demo of where 1_run_the_beta_model.Rmd goes wrong
# - Actual reproduction of the indexing mixup (as faithful as possible without having access to coral_diversity_for_coral_cover.csv)
# - Report on effect of the indexing mixup on region/diversity matching
# - JAGS test: paper indexing vs corrected indexing and comparison of coefficient forest plots
#
# TL;DR: diversity values received by JAGS are associated with the samples. This is caused by two bugs:
#
# Bug 1 — Site positional indexing (Rmd L174–178, L204, JAGS L213)
#   sites_and_region_df is built with distinct(Reef_ID, Ecoregion) and never
#   sorted by site id. JAGS uses region_for_each_site[i] for site i, but the
#   vector is in distinct()-row order, not site order.
#
# Bug 2 — Lexicographic factor (Rmd L204)
#   as.factor(as.character(sites_and_region_df$region)) re-orders levels as
#   "1","10","11",...,"2" instead of numeric 1,2,3 etc. before JAGS sees them.
#
# Usage (from repo root):
#   Rscript <path_to_file>/reproduce_indexing_mixup.R
#
# Optional: SKIP_JAGS=1  (skip the MCMC comparison at the end)

library(dplyr)
library(ggplot2)
library(R2jags)

# ---------------------------------------------------------------------------
# Dummy demonstration of where 1_run_the_beta_model.Rmd goes wrong
# ---------------------------------------------------------------------------
cat("\n=== Dummy demo: 4 ecoregions, diversity = region code for simplicity ===\n")

R_levels <- 11L  # max region code in toy (83 in full data); used for L204 factor levels

region_diversity_df <- data.frame(
  Ecoregion = c("A", "B", "C", "D"),
  region = c(1L, 2L, 10L, 11L),
  diversity = c(1, 2, 10, 11),
  stringsAsFactors = FALSE
)

sites_truth <- data.frame(
  site = 1:4,
  Reef_ID = c("R1", "R2", "R3", "R4"),
  Ecoregion = region_diversity_df$Ecoregion,
  region = region_diversity_df$region,
  diversity = region_diversity_df$diversity,
  stringsAsFactors = FALSE
)

# Rmd L67: diversity <- diversity[order(diversity$Ecoregion), ]
# Rmd L205: win.data$diversity <- diversity$diversity.standardized
diversity_received <- region_diversity_df$diversity[
  order(region_diversity_df$Ecoregion)
]
N_eco <- length(diversity_received)

# Correct fix: diversity[k] = value for region integer code k (lookup rather than matching incorrect index)
diversity_by_region_code <- rep(NA_real_, R_levels)
diversity_by_region_code[region_diversity_df$region] <- region_diversity_df$diversity

cat("\nGround truth (site i -> region i -> diversity = region code):\n")
print(sites_truth)
cat("\nPaper diversity vector (L67/L205, alphabetical by ecoregion name):\n")
print(data.frame(
  position = seq_len(N_eco),
  Ecoregion = sort(region_diversity_df$Ecoregion),
  diversity = diversity_received
))

# --- Bug 1 (L174, L201, JAGS L213): row i != site i ---
# Paper never sorts sites_and_region_df by site before passing to JAGS.
sites_received_order <- sites_truth[c(3, 1, 4, 2), ]  # example of permuted rows resulting from L174: distinct(Reef_ID, Ecoregion) which is not sorted by site index

cat("\nBug 1 — region_for_each_site[i] uses row i of unsorted table:\n")
print(data.frame(
  site_i = 1:4,
  region_expected = sites_truth$region,
  region_received = sites_received_order$region,
  div_expected = sites_truth$diversity,
  div_bug1_only = diversity_by_region_code[sites_received_order$region]
))

# --- Bug 2 (L204): as.factor(as.character(region)) on top of Bug 1 ---
paper_jags_idx <- as.integer(factor(
  as.character(sites_received_order$region),
  levels = sort(as.character(seq_len(R_levels)))   # full 1..11 lexicological order, as read by JAGS in L204
))

cat("\nBug 2 — L204 remaps region codes; JAGS indexes alphabetical diversity[z] (L205/L217):\n")
print(data.frame(
  site_i = 1:4,
  region_after_b1 = sites_received_order$region,
  jags_index = paper_jags_idx,
  div_received = diversity_received[paper_jags_idx],
  div_expected = sites_truth$diversity
))

# --- what JAGS actually receives ---
idx_factor <- factor(
  as.character(sites_received_order$region),
  levels = sort(as.character(seq_len(R_levels)))
)
mod <- "
model {
  for(i in 1:N) { y[i] ~ dnorm(a[idx[i]], 1) }
  for(j in 1:K) { a[j] ~ dnorm(0, 1) }
}
"
m <- jags.model(
  textConnection(mod),
  data = list(N = 4L, K = N_eco, idx = idx_factor, y = rep(0, 4)),
  n.chains = 1, n.adapt = 0, quiet = TRUE
)
cat("\nrjags — region codes vs m$data()$idx:\n")
cat("\nSummary — diversity each site SHOULD receive vs paper pipeline (Bug 1 + Bug 2):\n")
print(data.frame(
  site_i = seq_len(nrow(sites_truth)),
  reef = sites_truth$Reef_ID,
  ecoregion = sites_truth$Ecoregion,
  diversity_should_be = sites_truth$diversity,
  diversity_paper_gets = diversity_received[paper_jags_idx],
  stringsAsFactors = FALSE
))

# ---------------------------------------------------------------------------
# Actual reproduction of the indexing mixup (as close as possible, without access to coral_diversity_for_coral_cover.csv)
# ---------------------------------------------------------------------------
cat("\n=== Actual reproduction of the indexing mixup (as close as possible, without access to coral_diversity_for_coral_cover.csv) ===\n")

repo_root <- {
  root <- Sys.getenv("REEF_COVER_ECONOMICS_ROOT", unset = NA_character_)
  if (!is.na(root) && nzchar(root)) {
    normalizePath(root)
  } else {
    script <- local({
      if (length(f <- grep("^--file=", commandArgs(FALSE), value = TRUE))) {
        return(sub("^--file=", "", f[1]))
      }
      for (i in rev(seq_len(sys.nframe()))) {
        of <- sys.frame(i)$ofile
        if (!is.null(of) && nzchar(of)) return(of)
      }
      stop("Cannot locate script path; set REEF_COVER_ECONOMICS_ROOT")
    })
    dirname(normalizePath(script))
  }
}

out_dir <- file.path(repo_root, "output", "indexing_mixup")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
maps_path <- file.path(repo_root, "data_for_maps.csv")
if (!file.exists(maps_path)) {
  stop("Expected ", maps_path, " (repo root or REEF_COVER_ECONOMICS_ROOT)")
}

# ---------------------------------------------------------------------------
# Data load (stand-in for Rmd coral-cover + diversity inputs)
# ---------------------------------------------------------------------------
# Rmd L50:  coral_cover_data <- read.csv("data.csv", ...)
# Rmd L64:  diversity <- read.csv("coral_diversity_for_coral_cover.csv", ...)
# Here we use data_for_maps.csv which already has reef, ecoregion, and
# diversity.standardized — same columns the model ultimately needs.
message("Loading ", maps_path)
raw <- read.csv(maps_path, stringsAsFactors = FALSE)

# One row per reef (maps file is already filtered).
data <- raw[, c("Reef_ID", "Ecoregion.y", "diversity.standardized")]
names(data)[2] <- "Ecoregion"
data <- unique(data)

# ---------------------------------------------------------------------------
# Chunk: "create a dataframe containing information for each site, with the ecoregion it belongs to, and the standardized diversity value"
# Rmd L170–178
# ---------------------------------------------------------------------------

data$Reef_ID <- as.factor(as.character(as.factor(data$Reef_ID)))  # Rmd L172
sites_and_region_df <- data %>% distinct(Reef_ID, Ecoregion) %>% ungroup()  # Rmd L174

# Rmd L177–178: attach diversity by ecoregion name (not shown here using data_for_maps.csv since don't have original diversity file (see below)

sites_and_region_df$site <- as.numeric(as.factor(sites_and_region_df$Reef_ID))  # Rmd L177

sites_and_region_df$region <- as.numeric(as.factor(sites_and_region_df$Ecoregion))  # Rmd L178

# ---------------------------------------------------------------------------
# Diversity vector passed to JAGS in Rmd line 205 is:
# Rmd L64–67, L166, L205: diversity$diversity.standardized
# ---------------------------------------------------------------------------
# Generated from coral_diversity_for_coral_cover.csv:
# Rmd L64: diversity<-read.csv(file=file.path(diversity_data_directory, "coral_diversity_for_coral_cover.csv"), header=TRUE, sep=",")
# Rmd L67: diversity <- diversity[order(diversity$Ecoregion), ] # this orders by ecoregion name (alphabetically)
# Rmd L166: diversity$diversity.standardized <- standardize_function(...)
# Rmd L205: win.data$diversity <- diversity$diversity.standardized
#
# In JAGS, ecoregion z uses diversity[z] (Rmd JAGS L217):
#   g[z] <- mu_global + beta_diversity * diversity[z]
# So diversity[k] must be the value for region code k.
# This is produced here, enforcing numeric region order for JAGS indexing:
region_diversity_df <- unique(data[, c("Ecoregion", "diversity.standardized")])
region_diversity_df$region <- as.numeric(as.factor(region_diversity_df$Ecoregion))
region_diversity_df <- region_diversity_df[order(region_diversity_df$region), ]

# ---------------------------------------------------------------------------
# Chunk: "define, initialize, and run the beta model..."
# Rmd L183–206
# ---------------------------------------------------------------------------

# Rmd L191: Nre = length(unique(data$site))
Nre <- length(unique(data$Reef_ID)) # number of unique sites

# Rmd L193: R = length(unique(data$ERG))
R <- nrow(region_diversity_df)

stopifnot(nrow(sites_and_region_df) == Nre) # basic sanity check that number of sites in sites_and_region_df matches number of unique sites in data

# This is the vector JAGS receives in Rmd line 205. Length Nre; element i is *supposed* to be
# the region code for site i — but the Rmd never sorts sites_and_region_df by site/Reef_ID
paper_region_vec <- as.factor(as.character(sites_and_region_df$region)) # Rmd L204
paper_region_jags <- as.integer(paper_region_vec)

# Rmd JAGS L213 (inside model string, L210–262):
#   for (i in 1:Nre) { a[i] ~ dnorm(ecoregion[region_for_each_site[i]], tau) }
# JAGS reads region_for_each_site[i] when fitting site effect a[i].
# So row i of the R vector must correspond to site id i — which the Rmd violates.

# ---------------------------------------------------------------------------
# Ground truth: what the Rmd *should* have passed
# ---------------------------------------------------------------------------
# Sort the sites_and_region_df table by site id so that row i == site i (never done in Rmd).
correct_by_site <- sites_and_region_df[order(sites_and_region_df$site), ]
correct_region <- correct_by_site$region

# ---------------------------------------------------------------------------
# Per-site comparison table
# ---------------------------------------------------------------------------
# Bug 1: row i of unsorted table
site_tbl <- data.frame(
  site = seq_len(Nre),                              # site ids 1 ... Nre (Rmd L192: re = data$site)
  reef = as.character(correct_by_site$Reef_ID),
  ecoregion = correct_by_site$Ecoregion,
  region_true = correct_region,                     # correct region for site i
  region_paper_pos = sites_and_region_df$region,    # Bug 1: row i of unsorted table
  region_paper_jags = paper_region_jags,            # Bug 1 + Bug 2: what JAGS indexes
  stringsAsFactors = FALSE
)

# Bug 2, L204: as.factor(as.character(region)) re-orders 1,2,...,10,11 region values as 1,10,11,...,2.
# diversity[z] lookup — same indexing JAGS uses via region_for_each_site[i]
site_tbl$diversity_true <- region_diversity_df$diversity.standardized[site_tbl$region_true]
site_tbl$diversity_paper_pos <- region_diversity_df$diversity.standardized[site_tbl$region_paper_pos]
site_tbl$diversity_paper <- region_diversity_df$diversity.standardized[site_tbl$region_paper_jags]
site_tbl$delta_positional <- site_tbl$diversity_paper_pos - site_tbl$diversity_true
site_tbl$delta <- site_tbl$diversity_paper - site_tbl$diversity_true

positional_mismatch <- site_tbl$region_true != site_tbl$region_paper_pos
jags_mismatch <- site_tbl$region_true != site_tbl$region_paper_jags
diversity_pos_mismatch <- abs(site_tbl$delta_positional) > 1e-12
diversity_mismatch <- abs(site_tbl$delta) > 1e-12

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
cat("\n=== Indexing mixup summary (1_run_the_beta_model.Rmd) ===\n")
cat(sprintf("Sites (reefs): %d | Ecoregions: %d\n", Nre, R))

cat("\n--- Bug 1: positional row order (Rmd L174, L204; JAGS L213) ---\n")
cat("  distinct(Reef_ID, Ecoregion) row order != sorted site id order.\n")
cat("  JAGS assumes region_for_each_site[i] is the region for site i.\n")
cat(sprintf(
  "  Wrong region (positional): %d / %d sites (%.1f%%)\n",
  sum(positional_mismatch), Nre, 100 * mean(positional_mismatch)
))
cat(sprintf(
  "  Wrong diversity (positional): %d / %d (%.1f%%)\n",
  sum(diversity_pos_mismatch), Nre, 100 * mean(diversity_pos_mismatch)
))

cat("\n--- Bug 2: lexicographic factor (Rmd L204) ---\n")
cat("  as.factor(as.character(region)) re-orders 1,2,...,10,11 as 1,10,11,...,2.\n")
cat(sprintf(
  "  Additional wrong region (after bug 1): %d / %d sites (%.1f)\n",
  sum(jags_mismatch & !positional_mismatch), Nre,
  100 * mean(jags_mismatch & !positional_mismatch)
))
cat(sprintf(
  "  Wrong region (both bugs, as in paper): %d / %d sites (%.1f%%)\n",
  sum(jags_mismatch), Nre, 100 * mean(jags_mismatch)
))
cat(sprintf(
  "  Wrong diversity (paper JAGS input): %d / %d (%.1f)\n",
  sum(diversity_mismatch), Nre, 100 * mean(diversity_mismatch)
))
cat("  i.e. slightly better than random with 83 ecoregions (1/83 ~= 98.8% chance of being wrong)",
    " due to variation in number of points per ecoregion\n")
cat(sprintf(
  "  Mean |difference in diversity| (paper vs truth): %.4f (max %.4f)\n",
  mean(abs(site_tbl$delta)), max(abs(site_tbl$delta))
))

order_diff <- !identical(
  sites_and_region_df$site,
  sort(sites_and_region_df$site)
)
cat(
  if (order_diff) "\n  sites_and_region_df row order != sorted site ids\n" else "\n  sites_and_region_df row order == sorted site ids\n"
)

cat("\n=== First mismatched sites (bug 1 alone) ===\n")
cat("  site i should get region_true[i], but Rmd row i gives region_paper_pos[i]\n")
show_pos <- site_tbl[positional_mismatch, ]
print(head(show_pos[, c(
  "site", "reef", "ecoregion",
  "region_true", "region_paper_pos",
  "diversity_true", "diversity_paper_pos"
)], 8))

cat("\n=== Correct fix (as implemented below) ===\n")
cat("  Sort by site then pass integer region codes with no character factor:\n")
cat("    region_for_each_site <- sites_and_region_df[order(sites_and_region_df$site), ]$region\n")
fixed_region <- correct_region
fixed_div <- region_diversity_df$diversity.standardized[fixed_region]
cat(sprintf(
  "  Wrong region(s): %d / %d | Wrong diversity: %d / %d\n",
  sum(fixed_region != site_tbl$region_true),
  Nre,
  sum(abs(fixed_div - site_tbl$diversity_true) > 1e-12),
  Nre
))


# ---------------------------------------------------------------------------
# JAGS test: paper indexing vs corrected indexing
# Full observation-level dataset and coefficient forest plot comparison.
# ---------------------------------------------------------------------------

# Rmd L208–261 (GLMM_coral_cover.txt) — verbatim from 1_run_the_beta_model.Rmd
PAPER_JAGS_MODEL <- "
    model{
    #1A. Priors
    for (i in 1:K) { beta[i]  ~ dnorm(0, 0.0001) }  # K coefficients
    for (i in 1:Nre) {a[i] ~ dnorm(ecoregion[region_for_each_site[i]], tau)}  # Nre sites (=2949). But region_for_each_site is not sorted with ecoregion name alphabetically, as is 'ecoregion' vector.
    # Hierarchical effects
    for(z in 1:R){ # R is total number of ecoregions
    ecoregion[z] ~ dnorm(g[z],tau_ecoregion)
    g[z] <- mu_global + beta_diversity*diversity[z]
    } # R ecoregions (=83)
    mu_global ~ dnorm(0, 0.0001) # prior for global mean
    beta_diversity ~ dnorm(0, 0.0001) #prior for the slope for diversity
    
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
    for (i in 1:N){ # N observations (=7714)        
      Y[i]       ~ dbeta(shape1[i], shape2[i])
      shape1[i] <- theta * pi[i]        #a
      shape2[i] <- theta * (1 - pi[i])  #b
      
      logit(pi[i]) <- eta[i]
      eta[i]<- inprod(beta[], X[i,]) + a[re[i]]
      

      ExpY[i] <- pi[i] 
      VarY[i] <- pi[i] * (1 - pi[i])  / (theta + 1)
      PRes[i] <- (Y[i] - ExpY[i]) / sqrt(VarY[i])

      #Discrepancy measures (used for checking overdispersion)
      YNew[i]   ~ dbeta(shape1[i], shape2[i])   #New data
      PResNew[i] <- (YNew[i] - ExpY[i]) / sqrt(VarY[i])
      D[i]       <- pow(PRes[i], 2)
      DNew[i]    <- pow(PResNew[i], 2)
  } 
    Fit         <- sum(D[1:N])
    FitNew      <- sum(DNew[1:N]) 
}
"

standardize <- function(x) (x - mean(x, na.rm = TRUE)) / sd(x, na.rm = TRUE)

COEF_LABELS <- c(
  "Intercept", "Latitude", "Depth", "Human_pop", "Cyclone", "SST_mean",
  "SSTA_Mean", "SSTA_min", "SSTA_freqstdev", "SSTA_dhwmax", "TSA_max",
  "TSA_freqstdev", "Turbidity_mean", "Historical_SST_max", "Diversity"
) # N.B. 'Intercept' leads to poor convergence across chains due to interplay with mu_global

prepare_obs_data <- function(path) {
  # Following L108-124 in 1_run_the_beta_model.Rmd
  obs <- read.csv(path, stringsAsFactors = FALSE)
  obs$Ecoregion <- obs$Ecoregion.y
  obs$Reef_ID <- as.factor(as.character(as.factor(obs$Reef_ID)))
  obs$lat <- abs(obs$Latitude.Degrees)

  obs <- obs[
    !is.na(obs$Average_coral_cover) & obs$Average_coral_cover > 0 &
      !is.na(obs$SST_mean) & !is.na(obs$Depth) & !is.na(obs$Human_pop) &
      !is.na(obs$Cyclone) & !is.na(obs$SSTA_Mean) & !is.na(obs$SSTA_min) &
      !is.na(obs$SSTA_freqstdev) & !is.na(obs$SSTA_dhwmax) &
      !is.na(obs$TSA_max) & !is.na(obs$TSA_freqstdev) &
      !is.na(obs$Turbidity_mean) & !is.na(obs$Historical_SST_max),
    ,
    drop = FALSE
  ]

  pred_cols <- c(
    "lat", "Depth", "Human_pop", "Cyclone", "SST_mean", "SSTA_Mean",
    "SSTA_min", "SSTA_freqstdev", "SSTA_dhwmax", "TSA_max", "TSA_freqstdev",
    "Turbidity_mean", "Historical_SST_max"
  )
  for (col in pred_cols) {
    obs[[col]] <- standardize(obs[[col]])
  }
  obs
}

build_win_data <- function(
  obs,
  region_for_each_site,
  diversity_vec,
  R_eco,
  Nre_sites
) {
  # Following L152-166 in 1_run_the_beta_model.Rmd
  N <- nrow(obs)
  drop_cols <- intersect(c("site", "region", "ERName"), names(obs))
  if (length(drop_cols)) {
    obs <- obs[, !names(obs) %in% drop_cols, drop = FALSE]
  }
  obs <- dplyr::left_join(obs, sites_and_region_df, by = "Reef_ID")
  if (any(is.na(obs$site))) {
    stop("Observations missing site after join with sites_and_region_df.")
  }

  pred_cols <- c(
    "lat", "Depth", "Human_pop", "Cyclone", "SST_mean", "SSTA_Mean",
    "SSTA_min", "SSTA_freqstdev", "SSTA_dhwmax", "TSA_max", "TSA_freqstdev",
    "Turbidity_mean", "Historical_SST_max"
  )
  X <- model.matrix(
    stats::as.formula(paste("~", paste(pred_cols, collapse = " + "))),
    data = obs
  )
  Y <- (obs$Average_coral_cover * (N - 1) + 0.5) / N  # Rmd L197

  # Rmd L197–205: win.data list (re = data$site, observation-level site index)
  list(
    Y = as.numeric(Y),
    N = N,
    X = X,
    K = ncol(X),
    re = obs$site,
    R = R_eco,
    Nre = Nre_sites,
    region_for_each_site = region_for_each_site,
    diversity = as.numeric(diversity_vec)
  )
}

make_paper_inits <- function(K, Nre) {
  # Following L265-277 in 1_run_the_beta_model.Rmd
  force(K); force(Nre)
  function() {
    list(
      beta = stats::rnorm(K, 0, 0.1),
      beta_diversity = stats::rnorm(1, 0, 0.1),
      a = stats::rnorm(Nre, 0, 0.1),
      num = stats::rnorm(1, 0, 25),
      denom = stats::rnorm(1, 0, 1),
      numtheta = stats::rnorm(1, 0, 25),
      denomtheta = stats::rnorm(1, 0, 1),
      num_ecoregion = stats::rnorm(1, 0, 25),
      denom_ecoregion = stats::rnorm(1, 0, 1)
    )
  }
}

extract_coefficient_df <- function(fit, label, K) {
  # approximately L305 in 1_run_the_beta_model.Rmd
  summ <- fit$BUGSoutput$summary
  beta_rows <- paste0("beta[", seq_len(K), "]")
  out <- data.frame(
    indexing = label,
    variable = COEF_LABELS[seq_len(K)],
    mean = summ[beta_rows, "mean"],
    lower = summ[beta_rows, "2.5%"],
    upper = summ[beta_rows, "97.5%"],
    stringsAsFactors = FALSE
  )
  if ("beta_diversity" %in% rownames(summ)) {
    out <- rbind(
      out,
      data.frame(
        indexing = label,
        variable = "Diversity",
        mean = summ["beta_diversity", "mean"],
        lower = summ["beta_diversity", "2.5%"],
        upper = summ["beta_diversity", "97.5%"],
        stringsAsFactors = FALSE
      )
    )
  }
  out
}

plot_indexing_forest <- function(
  coef_df,
  output_path,
  include_intercept = TRUE
) {
  # Following L320-337 but with comparison of paper and corrected indexing
  paper_label <- "Paper indexing"
  correct_label <- "Correct indexing"

  plot_df <- coef_df
  if (!include_intercept) {
    plot_df <- plot_df[plot_df$variable != "Intercept", , drop = FALSE]
  }

  paper_order <- plot_df %>%
    dplyr::filter(.data$indexing == paper_label, .data$variable != "Intercept") %>%
    dplyr::arrange(.data$mean) %>%
    dplyr::pull(.data$variable)

  y_levels <- if (include_intercept) {
    c("Intercept", paper_order)
  } else {
    paper_order
  }

  plot_df <- plot_df %>%
    dplyr::mutate(
      y_label = factor(.data$variable, levels = y_levels),
      sig = ifelse(
        .data$lower > 0, "Positive",
        ifelse(.data$upper < 0, "Negative", "Not significant")
      ),
      sig = factor(
        .data$sig,
        levels = c("Positive", "Negative", "Not significant")
      )
    )

  dodge <- ggplot2::position_dodge(width = 0.55)

  p <- ggplot2::ggplot(
    plot_df,
    ggplot2::aes(
      y = .data$y_label,
      x = .data$mean,
      xmin = .data$lower,
      xmax = .data$upper,
      colour = .data$indexing,
      fill = .data$sig
    )
  ) +
    ggplot2::geom_vline(xintercept = 0, linetype = "dashed", colour = "grey50") +
    ggplot2::geom_pointrange(
      orientation = "y",
      position = dodge,
      linewidth = 0.7,
      size = 0.8,
      pch = 21,
      stroke = 0.35
    ) +
    ggplot2::scale_y_discrete(expand = ggplot2::expansion(mult = c(0.04, 0.04))) +
    ggplot2::scale_colour_manual(
      values = stats::setNames(c("#ff5100", "#32c40a"), c(paper_label, correct_label))
    ) +
    ggplot2::scale_fill_manual(
      values = c(
        "Positive" = "blue",
        "Negative" = "red",
        "Not significant" = "white"
      ),
      name = "Significance"
    ) +
    ggplot2::theme_grey(base_size = 12) +
    ggplot2::theme(legend.position = "bottom", legend.box = "horizontal") +
    ggplot2::labs(
      title = "Indexing assessment: 2022 paper implementation vs corrected indexing",
      x = expression("Estimated "*gamma*" coefficients"),
      y = NULL,
      colour = "Indexing"
    ) +
    ggplot2::guides(
      colour = ggplot2::guide_legend(order = 1, override.aes = list(fill = NA)),
      fill = ggplot2::guide_legend(order = 2, override.aes = list(colour = "black"))
    )

  height <- if (include_intercept) 8 else 7
  ggplot2::ggsave(output_path, plot = p, width = 10, height = height, dpi = 300)
  invisible(p)
}

write_indexing_forest_plots <- function(coef_df, out_dir) {
  plot_indexing_forest(
    coef_df,
    file.path(out_dir, "indexing_comparison_coeff_forest.png"),
    include_intercept = TRUE
  )
  plot_indexing_forest(
    coef_df,
    file.path(out_dir, "indexing_comparison_coeff_forest_no_intercept.png"),
    include_intercept = FALSE
  )
}

run_jags <- function(win.data, label, model_path, mcmc) {
  # parallelised version of L282-289 in 1_run_the_beta_model.Rmd (parallelisation doesn't change results)
  message("JAGS fit: ", label, " (N=", win.data$N, ", Nre=", win.data$Nre, ")")
  jags.parallel(
    data = win.data,
    inits = make_paper_inits(win.data$K, win.data$Nre),
    parameters.to.save = c("beta", "beta_diversity"),
    model.file = model_path,
    n.chains = mcmc$n.chains,
    n.burnin = mcmc$n.burnin,
    n.iter = mcmc$n.iter,
    n.thin = mcmc$n.thin,
  )
}

# optionally skip JAGS comparison via SKIP_JAGS=1 commandline argument
if (nzchar(Sys.getenv("SKIP_JAGS", unset = ""))) {
  message("SKIP_JAGS set — skipping JAGS comparison.")
  coef_csv <- file.path(out_dir, "indexing_comparison_coefficients.csv")
  if (file.exists(coef_csv)) {
    coef_all <- read.csv(coef_csv, stringsAsFactors = FALSE)
    write_indexing_forest_plots(coef_all, out_dir)
    message("Regenerated forest plots from ", coef_csv)
  }
} else {
  cat("\n=== JAGS test (Rmd L185–290) ===\n")
  obs <- prepare_obs_data(maps_path)
  cat(sprintf("  Observations after filters: %d\n", nrow(obs)))

  diversity_vec <- region_diversity_df$diversity.standardized

  # Rmd L204: sites out of order, ecoregion codes scrambled via as.factor(as.character(sites_and_region_df$region))
  win_paper <- build_win_data(
    obs,
    region_for_each_site = as.factor(as.character(sites_and_region_df$region)),
    diversity_vec = diversity_vec,
    R_eco = R,
    Nre_sites = Nre
  )
  # Fixed: sort by site, pass integer region codes (no character factor)
  win_fixed <- build_win_data(
    obs,
    region_for_each_site = as.integer(correct_region),
    diversity_vec = diversity_vec,
    R_eco = R,
    Nre_sites = Nre
  )

  mcmc_run <- list(n.chains = 3L, n.burnin = 4000L, n.iter = 15000L, n.thin = 10L)  # N.B. these are same settings as paper (and as attached graph), but take a very long time to run
  # mcmc_run <- list(n.chains = 3L, n.burnin = 40L, n.iter = 100L, n.thin = 1L)  # use these to run faster and show the same results. N.B. when running in parallel there is no progress bar
  model_path <- file.path(out_dir, "GLMM_coral_cover.txt")
  writeLines(PAPER_JAGS_MODEL, con = model_path)

  fit_paper <- run_jags(win_paper, "Paper indexing", model_path, mcmc_run)
  fit_fixed <- run_jags(win_fixed, "Correct indexing (sorted site)", model_path, mcmc_run)

  coef_paper <- extract_coefficient_df(fit_paper, "Paper indexing", win_paper$K)
  coef_fixed <- extract_coefficient_df(fit_fixed, "Correct indexing", win_fixed$K)
  coef_all <- rbind(coef_paper, coef_fixed)

  write_indexing_forest_plots(coef_all, out_dir)
  write.csv(coef_all, file.path(out_dir, "indexing_comparison_coefficients.csv"), row.names = FALSE)

  cat("\n  Coefficient comparison written to:\n")
  cat("   ", file.path(out_dir, "indexing_comparison_coeff_forest.png"), "\n")
  cat("   ", file.path(out_dir, "indexing_comparison_coeff_forest_no_intercept.png"), "\n")
  cat("\n  beta_diversity (paper vs fixed):\n")
  print(coef_all[coef_all$variable == "Diversity", c("indexing", "mean", "lower", "upper")])
}
