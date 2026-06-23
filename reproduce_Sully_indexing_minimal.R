#!/usr/bin/env Rscript
# Minimal indexing mixup demo for the Sully et al. 2022 paper (https://doi.org/10.1111/gcb.16083) implementation of 1_run_the_beta_model.Rmd (https://github.com/InstituteForGlobalEcology/Present-and-future-bright-and-dark-spots-for-coral-reefs-through-climate-change/blob/main/1_run_the_beta_model.Rmd)
# where lines are referenced, e.g. L204, these refer to the line numbers in the 1_run_the_beta_model.Rmd file. 

library(dplyr)
library(ggplot2)
library(patchwork)

# load and prepare data produced by 1_run_the_beta_model.Rmd
raw <- read.csv("data_for_maps.csv")
sites_dat <- unique(raw[, c("Reef_ID", "Ecoregion.y", "diversity.standardized", "Latitude.Degrees", "Longitude.Degrees")])
names(sites_dat)[2] <- "Ecoregion"
sites_dat$Reef_ID <- as.factor(as.character(as.factor(sites_dat$Reef_ID)))

# Bug 1 (L176/L204): row i of unsorted table != site i
sites <- sites_dat %>% distinct(Reef_ID, Ecoregion, Latitude.Degrees, Longitude.Degrees) %>% mutate( # create a dataframe with site and region columns N.B. distinct(Reef_ID, Ecoregion) is same shape as distinct(Reef_ID) since Reef_ID is a unique location label: a string from lat/long coordinates
  site = as.numeric(factor(Reef_ID)), # assign lexicographic index to each unique Reef_ID
  region = as.numeric(factor(Ecoregion))
)
truth <- sites[order(sites$site), ] # order ascending to ensure later matching

# Bug 2 (L204): as.factor(as.character(region)) -> JAGS lexicographic index (row i, for site i) in L213
R <- max(sites$region)  # number of unique regions
lex_levels <- sort(as.character(seq_len(R)))  # lexicographic ordering of regions
paper_jags_idx <- as.integer(factor(  # assign JAGS slot index to each region
  as.character(sites$region),
  levels = lex_levels
))  # L204

# Rmd L67/L205: diversity vector sorted alphabetically by ecoregion name
region_div <- unique(sites_dat[, c("Ecoregion", "diversity.standardized")]) %>% # get diversity values for each ecoregion
  mutate(region = as.numeric(factor(Ecoregion))) %>%
  arrange(region)
div_paper_vec <- region_div$diversity.standardized[order(region_div$Ecoregion)] # L67
ecoregion_by_jags_slot <- region_div$Ecoregion[order(region_div$Ecoregion)] # get ecoregion names for each row in the JAGS slot for later comparison

site_diversity <- data.frame(
  site = truth$site,
  reef = as.character(truth$Reef_ID),
  lat = truth$Latitude.Degrees,
  lon = truth$Longitude.Degrees,
  ecoregion_correct = truth$Ecoregion,
  ecoregion_paper = ecoregion_by_jags_slot[paper_jags_idx], # order is by lexicographic sort
  diversity_correct = region_div$diversity.standardized[
    match(truth$Ecoregion, region_div$Ecoregion)
  ],  # look up correct diversity value for each ecoregion
  diversity_paper = div_paper_vec[paper_jags_idx],  # diversity value as accessed by model in L216
  stringsAsFactors = FALSE
)
site_diversity$mismatch <- abs(site_diversity$diversity_correct - site_diversity$diversity_paper) > 1e-12 # check for significant numerical mismatches

cat(sprintf(
  "Wrong ecoregion (Bug 1: row/site assignment mismatch): %d / %d (%.1f%%)\n",
  sum(truth$Ecoregion != sites$Ecoregion), nrow(truth),
  100 * mean(truth$Ecoregion != sites$Ecoregion)
))  # (L176/L204): row i of unsorted table != site i; in paper_jags_idx, this is where the lexicographic index is used to assign the JAGS slot
cat(sprintf(
  "Wrong ecoregion (Bug 1 + Bug 2: adding lexicographic ordering of biodiversity-ecoregion array): %d / %d (%.1f%%)\n",
  sum(truth$Ecoregion != site_diversity$ecoregion_paper), nrow(truth),
  100 * mean(truth$Ecoregion != site_diversity$ecoregion_paper)
))  # combined with Bug 2 (L204): as.factor(as.character(region)) -> JAGS lexicographic index (row i, for site i) in L213/L217

cat("\nFirst diversity mismatches (correct vs paper pipeline):\n")
print(head(subset(
  site_diversity,
  mismatch,
  select = c("site", "lat", "lon", "reef", "ecoregion_correct", "ecoregion_paper", "diversity_correct", "diversity_paper")
)))

out_csv <- file.path("output", "indexing_mixup", "site_diversity_paper_vs_truth.csv")
dir.create(dirname(out_csv), recursive = TRUE, showWarnings = FALSE)
write.csv(site_diversity, out_csv, row.names = FALSE)
cat("\nFull table written to:", out_csv, "\n")

# --- Spatial map illustrating combined indexing bugs ---
coords <- raw %>%
  distinct(Reef_ID, .keep_all = TRUE) %>%
  transmute(  # create new dataframe with these columns
    reef = as.character(Reef_ID),
    lat = Latitude.Degrees,
    lon = Longitude.Degrees
  )

region_map <- data.frame(
  site = truth$site,
  reef = as.character(truth$Reef_ID),
  lat = truth$Latitude.Degrees,
  lon = truth$Longitude.Degrees,
  ecoregion_correct = truth$Ecoregion,
  ecoregion_paper = ecoregion_by_jags_slot[paper_jags_idx],
  stringsAsFactors = FALSE
)

world <- ggplot2::map_data("world")
xlim <- range(region_map$lon) + c(-8, 8)
ylim <- range(region_map$lat) + c(-8, 8)

n_highlight <- 10L
top_ecoregions <- region_map %>%
  filter(ecoregion_correct != ecoregion_paper) %>%
  count(ecoregion_paper, sort = TRUE) %>%
  slice_head(n = n_highlight) %>%
  pull(ecoregion_paper)

highlight_cols <- c(
  RColorBrewer::brewer.pal(max(3L, n_highlight), "Set3")[seq_len(n_highlight)],
  "grey85"
)
names(highlight_cols) <- c(top_ecoregions, "Other")

make_ecoregion_map <- function(data, group_col, title, subtitle,
                               show_x_axis = FALSE) {
  plot_data <- data
  plot_data$map_colour <- factor(
    plot_data[[group_col]],
    levels = c(top_ecoregions, "Other")
  )
  ggplot() +
    geom_polygon( # plot world map
      data = world,
      aes(x = long, y = lat, group = group),
      fill = "#fbfbfb",
      colour = "grey75",
      linewidth = 0.15
    ) +
    geom_point( # plot non-highlighted ecoregions as light grey crosses
      data = subset(plot_data, map_colour == "Other"),
      aes(x = lon, y = lat),
      shape = 4,           # cross code
      colour = "#c5c5c5",
      size = 0.5,
      alpha = 0.7
    ) +
    geom_point( # plot highlighted (to N ecoregion examples of incorrect) ecoregions
      data = subset(plot_data, map_colour != "Other"),
      aes(x = lon, y = lat, colour = map_colour),
      size = 1,
      alpha = 1
    ) +
    coord_fixed(xlim = xlim, ylim = ylim, expand = FALSE) +
    scale_colour_manual(
      values = highlight_cols,
      breaks = top_ecoregions,
      name = "Ecoregion"
    ) +
    labs(
      title = title,
      subtitle = subtitle,
      x = if (show_x_axis) "Longitude" else NULL,
      y = "Latitude"
    ) +
    theme_minimal(base_size = 11) +
    theme(
      panel.grid.minor = element_blank(),
      legend.position = "none",
      plot.title = element_text(size = 11, margin = margin(b = 2)),
      plot.subtitle = element_text(size = 9, margin = margin(b = 4)),
      axis.title.x = if (show_x_axis) element_text() else element_blank(),
      axis.text.x = if (show_x_axis) element_text() else element_blank(),
      axis.ticks.x = if (show_x_axis) element_line() else element_blank()
    )
}

region_map <- region_map %>%
  mutate(
    map_group_wrong = if_else(
      ecoregion_correct != ecoregion_paper & ecoregion_paper %in% top_ecoregions,
      ecoregion_paper,
      "Other"
    ),
    map_group_correct = if_else(
      ecoregion_correct %in% top_ecoregions,
      ecoregion_correct,
      "Other"
    )
  )

p_wrong <- make_ecoregion_map(
  region_map,
  "map_group_wrong",
  title = "Paper pipeline: incorrectly assigned ecoregions",
  subtitle = sprintf(
    "%d / %d (%.1f%%) sites mismatched; coloured by ecoregion at JAGS slot z (L204, L213). Top %d mismatched ecoregion(s) are shown. \nGrey crosses show the rest of the data.",
    sum(region_map$ecoregion_correct != region_map$ecoregion_paper),
    nrow(region_map),
    100 * mean(region_map$ecoregion_correct != region_map$ecoregion_paper),
    n_highlight
  ),
  show_x_axis = FALSE
)

p_correct <- make_ecoregion_map(
  region_map,
  "map_group_correct",
  title = "Corrected ecoregion assignment",
  subtitle = sprintf("Same %d ecoregion(s) as above.\nGrey crosses show the rest of the data.", n_highlight),
  show_x_axis = TRUE
)

p_map <- p_wrong / p_correct +
  patchwork::plot_layout(heights = c(1, 1), guides = "collect") &
  theme(
    legend.position = "bottom",
    legend.box = "horizontal",
    legend.text = element_text(size = 7.5),
    legend.title = element_text(size = 9),
    legend.key.size = unit(0.35, "cm"),
  ) &
  guides(colour = guide_legend(nrow = 2, byrow = TRUE, title = "Ecoregion"))

out_map <- file.path("output", "indexing_mixup", "indexing_region_mismatch_map.png")
ggsave(out_map, plot = p_map, width = 9.5, height = 6.8, dpi = 300)
cat("Region mismatch map written to:", out_map, "\n")
