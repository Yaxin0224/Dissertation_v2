#!/usr/bin/env Rscript
# =============================================================================
# City Look Dissertation - RQ2 Spatial Analysis (Final Integrated Version)
# Spatial Patterns and Administrative Boundary Effects on GVI-IMD Relationship
# Date: 2025-08-10
# =============================================================================
# HOW TO RUN (example)
#   Rscript RQ2_spatial_analysis.R \
#       --csv "C:/path/to/inner_london_all_lsoas_complete.csv" \
#       --shp "C:/path/to/ESRI/LSOA_2011_London_gen_MHW.shp" \
#       --out "./output" --fig "./figures" \
#       --topn=10 --imd_q=0.80
# =============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(sf)
  library(sp)           # for GWmodel 'Spatial' coercion
  library(spdep)
  library(spatialreg)
  library(tmap)
  library(viridis)
  library(GWmodel)
  library(ggplot2)
  library(lme4)
  library(lmerTest)
  library(performance)  # r2(), icc()
  library(scales)
})

set.seed(42)
theme_set(theme_minimal(base_size = 12))

# ------------- 0. ARGUMENTS & PATHS -----------------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)
get_flag <- function(flag, default = NA_character_) {
  hit <- grep(paste0("^", flag, "="), args, value = TRUE)
  if (length(hit) == 0) return(default)
  sub(paste0("^", flag, "="), "", hit[1])
}

# Defaults (edit if you like)
default_csv <- file.path(getwd(), "output", "full_london_gvi", "inner_london_all_lsoas_complete.csv")
default_shp <- "C:/Users/z1782/OneDrive - University College London/Attachments/004/methodology/Dissertation_v2/data/raw/statistical-gis-boundaries-london/ESRI/LSOA_2011_London_gen_MHW.shp"
default_out <- file.path(getwd(), "output")
default_fig <- file.path(getwd(), "figures")

DATA_PATH      <- get_flag("--csv", default_csv)
SHAPEFILE_PATH <- get_flag("--shp", default_shp)
OUT_DIR        <- get_flag("--out", default_out)
FIG_DIR        <- get_flag("--fig", default_fig)

# New flags: Top-N per-borough list & IMD quantile threshold
N_TOPN <- suppressWarnings(as.integer(get_flag("--topn", "10")))
if (is.na(N_TOPN) || N_TOPN < 1) N_TOPN <- 10
IMD_Q  <- suppressWarnings(as.numeric(get_flag("--imd_q", "0.80")))
if (is.na(IMD_Q) || IMD_Q <= 0 || IMD_Q >= 1) IMD_Q <- 0.80

# Create output dirs if needed
if (!dir.exists(OUT_DIR))  dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)
if (!dir.exists(FIG_DIR))  dir.create(FIG_DIR, recursive = TRUE, showWarnings = FALSE)

# Normalize Windows paths (allow backslashes in user input)
DATA_PATH      <- gsub("\\\\", "/", DATA_PATH)
SHAPEFILE_PATH <- gsub("\\\\", "/", SHAPEFILE_PATH)

cat("\n========== RQ2 SPATIAL ANALYSIS: PATHS ==========\n")
cat("Data CSV   :", DATA_PATH, "\n")
cat("Shapefile  :", SHAPEFILE_PATH, "\n")
cat("Tables  -> :", OUT_DIR, "\n")
cat("Figures -> :", FIG_DIR, "\n")
cat("Top-N per Borough:", N_TOPN, " | IMD quantile:", IMD_Q, "\n")

# ------------- 1. LOAD DATA -------------------------------------------------------------------
cat("\n========== LOADING DATA ==========\n")

if (!file.exists(DATA_PATH)) {
  stop("CSV not found: ", DATA_PATH, call. = FALSE)
}
gvi_data <- readr::read_csv(DATA_PATH, show_col_types = FALSE)

if (!file.exists(SHAPEFILE_PATH)) {
  stop("Shapefile not found: ", SHAPEFILE_PATH, call. = FALSE)
} else {
  lsoa_boundaries <- st_read(SHAPEFILE_PATH, quiet = TRUE)
}

# Basic prep
gvi_data <- gvi_data %>%
  mutate(
    imd_quintile = factor(imd_quintile,
                          levels = c("Q1_Least","Q2","Q3","Q4","Q5_Most")),
    data_source = factor(data_source),
    borough     = factor(borough)
  ) %>%
  filter(!is.na(mean_gvi), !is.na(imd_score))

# ------------- 2. HARMONISE & MERGE -----------------------------------------------------------
lsoa_boundaries <- lsoa_boundaries %>%
  st_make_valid() %>%
  rename_with(tolower)

# Join (assumes gvi_data$lsoa_code matches lsoa_boundaries$lsoa11cd)
spatial_data <- lsoa_boundaries %>%
  inner_join(gvi_data, by = c("lsoa11cd" = "lsoa_code")) %>%
  st_transform(27700) # British National Grid

cat(paste("Spatial data prepared:", nrow(spatial_data), "LSOAs with geometry\n"))

# ------------- 3. SPATIAL WEIGHTS -------------------------------------------------------------
cat("\n========== SPATIAL WEIGHTS ==========\n")
# Queen contiguity
nb_queen <- poly2nb(spatial_data, queen = TRUE)
W_queen  <- nb2listw(nb_queen, style = "W", zero.policy = TRUE)

# KNN (k=8) as alternative (not used for main)
coords <- st_coordinates(st_centroid(spatial_data))
nb_knn <- knn2nb(knearneigh(coords, k = 8))
W_knn  <- nb2listw(nb_knn, style = "W", zero.policy = TRUE)

cat("--- Queen Summary ---\n")
cat("Average neighbors:", round(mean(card(nb_queen)), 2), "\n")
cat("Islands (no neighbors):", sum(card(nb_queen) == 0), "\n")

saveRDS(list(nb_queen=nb_queen, nb_knn=nb_knn, W_queen=W_queen, W_knn=W_knn),
        file.path(OUT_DIR, "spatial_weights.rds"))

# ------------- 4. GLOBAL SPATIAL AUTOCORRELATION (4.4.1) -------------------------------------
cat("\n========== GLOBAL SPATIAL AUTOCORRELATION (4.4.1) ==========\n")

moran_gvi  <- moran.test(spatial_data$mean_gvi, W_queen, zero.policy = TRUE)
moran_imd  <- moran.test(spatial_data$imd_score, W_queen, zero.policy = TRUE)

lm_basic   <- lm(mean_gvi ~ imd_score, data = st_drop_geometry(spatial_data))
moran_res  <- moran.test(residuals(lm_basic), W_queen, zero.policy = TRUE)

moran_summary <- tibble::tibble(
  Variable    = c("GVI", "IMD Score", "OLS Residuals"),
  Morans_I    = c(unname(moran_gvi$estimate["Moran I statistic"]),
                  unname(moran_imd$estimate["Moran I statistic"]),
                  unname(moran_res$estimate["Moran I statistic"])),
  Expected_I  = c(unname(moran_gvi$estimate["Expectation"]),
                  unname(moran_imd$estimate["Expectation"]),
                  unname(moran_res$estimate["Expectation"])),
  Variance    = c(unname(moran_gvi$estimate["Variance"]),
                  unname(moran_imd$estimate["Variance"]),
                  unname(moran_res$estimate["Variance"])),
  Z_score     = c(unname(moran_gvi$statistic),
                  unname(moran_imd$statistic),
                  unname(moran_res$statistic)),
  p_value     = c(moran_gvi$p.value, moran_imd$p.value, moran_res$p.value),
  Interpretation = c(
    ifelse(moran_gvi$p.value < 0.05, ifelse(moran_gvi$estimate["Moran I statistic"]>0,"Significant clustering","Significant dispersion"), "Random"),
    ifelse(moran_imd$p.value < 0.05, ifelse(moran_imd$estimate["Moran I statistic"]>0,"Significant clustering","Significant dispersion"), "Random"),
    ifelse(moran_res$p.value < 0.05, ifelse(moran_res$estimate["Moran I statistic"]>0,"Significant clustering","Significant dispersion"), "Random")
  )
)

readr::write_csv(moran_summary, file.path(OUT_DIR, "table_4_4_global_morans_i.csv"))
print(moran_summary)

# Figure 4.4a: Moran Scatterplot (GVI)
moran_plot_data <- tibble::tibble(
  gvi         = spatial_data$mean_gvi,
  lagged_gvi  = lag.listw(W_queen, spatial_data$mean_gvi, zero.policy = TRUE),
  borough     = spatial_data$borough
)

fig_4_4a <- ggplot(moran_plot_data, aes(x = gvi, y = lagged_gvi)) +
  geom_point(aes(color = borough), alpha = 0.6, size = 2) +
  geom_smooth(method = "lm", color = "red", se = TRUE, alpha = 0.2) +
  geom_hline(yintercept = mean(moran_plot_data$lagged_gvi), linetype = "dashed") +
  geom_vline(xintercept = mean(moran_plot_data$gvi), linetype = "dashed") +
  scale_color_viridis_d() +
  labs(title = "Moran's I Scatterplot for GVI",
       subtitle = paste0("Moran's I = ", round(moran_gvi$estimate["Moran I statistic"], 3),
                         ", p ", ifelse(moran_gvi$p.value < 0.001, "< 0.001", paste0("= ", signif(moran_gvi$p.value, 3)))),
       x = "GVI (%)", y = "Spatially Lagged GVI (%)",
       caption = "Quadrants: HH (UR), LL (LL), HL (LR), LH (UL)") +
  theme(plot.title = element_text(face = "bold", size = 14),
        legend.position = "none")

ggsave(file.path(FIG_DIR, "fig_4_4a_moran_scatterplot.png"),
       fig_4_4a, width = 10, height = 8, dpi = 300)

# Geary's C (supplementary)
geary_gvi <- geary.test(spatial_data$mean_gvi, W_queen, zero.policy = TRUE)
cat("\n--- Geary's C ---\n")
cat("Geary's C =", round(unname(geary_gvi$estimate[1]), 3),
    ", p =", format.pval(geary_gvi$p.value), "\n")

# ------------- 5. LISA (4.4.2) ----------------------------------------------------------------
cat("\n========== LOCAL SPATIAL AUTOCORRELATION - LISA (4.4.2) ==========\n")

lisa <- localmoran(spatial_data$mean_gvi, W_queen, zero.policy = TRUE, na.action = na.exclude)

lisa_df <- as.data.frame(lisa)
# Robustly locate the p-value column (spdep versions differ)
pname <- grep("^Pr\\(", colnames(lisa_df), value = TRUE)
if (length(pname) == 0) stop(paste0("localmoran() output has no p-value column. Columns: ", paste(colnames(lisa_df), collapse = ", ")))
spatial_data$lisa_I    <- lisa_df$Ii
spatial_data$lisa_z    <- lisa_df$Z.Ii
spatial_data$lisa_pval <- lisa_df[[pname[1]]]
# FDR-corrected p-values (recommended in literature)
spatial_data$lisa_padj <- p.adjust(spatial_data$lisa_pval, method = "fdr")
spatial_data$gvi_lag   <- lag.listw(W_queen, spatial_data$mean_gvi, zero.policy = TRUE)

gvi_mean <- mean(spatial_data$mean_gvi, na.rm = TRUE)
lag_mean <- mean(spatial_data$gvi_lag, na.rm = TRUE)

spatial_data <- spatial_data %>%
  mutate(
    lisa_cluster = case_when(
      lisa_padj > 0.05 ~ "Not Significant",
      mean_gvi > gvi_mean & gvi_lag > lag_mean ~ "High-High",
      mean_gvi < gvi_mean & gvi_lag < lag_mean ~ "Low-Low",
      mean_gvi > gvi_mean & gvi_lag < lag_mean ~ "High-Low",
      mean_gvi < gvi_mean & gvi_lag > lag_mean ~ "Low-High",
      TRUE ~ "Not Significant"
    ),
    lisa_cluster = factor(lisa_cluster,
                          levels = c("High-High","Low-Low","High-Low","Low-High","Not Significant"))
  )

lisa_summary <- spatial_data %>%
  st_drop_geometry() %>%
  group_by(lisa_cluster) %>%
  summarise(
    n = n(),
    mean_gvi = mean(mean_gvi, na.rm = TRUE),
    mean_imd = mean(imd_score, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(percentage = round(n / sum(n) * 100, 1))

readr::write_csv(lisa_summary, file.path(OUT_DIR, "lisa_cluster_summary.csv"))
print(lisa_summary)

# Identify "green inequality" hotspots (Low GVI, High IMD => Low-High)
hotspots <- spatial_data %>%
  st_drop_geometry() %>%
  filter(lisa_cluster == "Low-High") %>%
  select(lsoa11cd, borough, mean_gvi, imd_score, lisa_cluster) %>%
  arrange(mean_gvi) %>%
  mutate(rank = row_number())
readr::write_csv(hotspots, file.path(OUT_DIR, "lisa_green_inequality_hotspots.csv"))

# NEW: Top-N Low-High by Borough (policy-ready lists)
imd_thr <- quantile(spatial_data$imd_score, IMD_Q, na.rm = TRUE)  # e.g., 80th percentile (most deprived)
lh_all <- spatial_data %>% sf::st_drop_geometry() %>%
  filter(lisa_cluster == "Low-High") %>%
  mutate(
    # Priority: combine low GVI (worse is lower) and high IMD (worse is higher)
    priority = as.numeric(scale(-mean_gvi)) + as.numeric(scale(imd_score)),
    deprived = imd_score >= imd_thr
  )

lh_topN_by_borough <- lh_all %>%
  group_by(borough) %>%
  arrange(desc(priority), .by_group = TRUE) %>%
  slice_head(n = N_TOPN) %>%
  ungroup() %>%
  select(lsoa11cd, borough, mean_gvi, imd_score, lisa_I, lisa_z, lisa_pval, lisa_padj, priority, deprived)

lh_topN_deprived <- lh_topN_by_borough %>% filter(deprived)

readr::write_csv(lh_topN_by_borough, file.path(OUT_DIR, "lisa_LH_topN_by_borough.csv"))
readr::write_csv(lh_topN_deprived,    file.path(OUT_DIR, "lisa_LH_topN_by_borough_deprived.csv"))

# Maps
tmap_mode("plot")

lisa_map <- tm_shape(spatial_data) +
  tm_polygons("lisa_cluster",
              palette = c("High-High"="darkgreen",
                          "Low-Low"="darkred",
                          "High-Low"="lightgreen",
                          "Low-High"="orange",
                          "Not Significant"="gray90"),
              title = "LISA Clusters",
              border.alpha = 0.3) +
  tm_layout(title = "LISA Clusters: Green View Index",
            title.size = 1.2, title.fontface = "bold",
            legend.position = c("right","bottom"), frame = FALSE)

tmap_save(lisa_map, file.path(FIG_DIR, "fig_4_4b_lisa_clusters.png"),
          width = 10, height = 8, dpi = 300)

sig_map <- tm_shape(spatial_data) +
  tm_polygons("lisa_padj",
              style = "fixed",
              breaks = c(-Inf, 0.001, 0.01, 0.05, Inf),
              palette = c("darkred","red","orange","gray90"),
              title = "FDR-adjusted p") +
  tm_layout(title = "LISA Significance (FDR-adjusted)",
            title.size = 1.2, title.fontface = "bold",
            legend.position = c("right","bottom"), frame = FALSE)

tmap_save(sig_map, file.path(FIG_DIR, "fig_4_4c_lisa_significance.png"),
          width = 10, height = 8, dpi = 300)

# ------------- 6. BOROUGH-LEVEL ANALYSIS (4.4.3) ----------------------------------------------
cat("\n========== BOROUGH-LEVEL ANALYSIS (4.4.3) ==========\n")

# ——更稳健：分两步算，再合并——
safe_cor <- function(x, y) {
  x <- as.numeric(x); y <- as.numeric(y)
  ok <- is.finite(x) & is.finite(y)
  if (sum(ok) < 2) return(NA_real_)           # 组内样本太少
  if (sd(x[ok]) == 0 || sd(y[ok]) == 0) return(NA_real_)  # 常量列
  suppressWarnings(cor(x[ok], y[ok], use = "complete.obs"))
}

stats_df <- gvi_data %>%
  dplyr::group_by(borough) %>%
  dplyr::summarise(
    n_lsoas  = dplyr::n(),
    mean_gvi = mean(mean_gvi, na.rm = TRUE),
    sd_gvi   = sd(mean_gvi, na.rm = TRUE),
    mean_imd = mean(imd_score, na.rm = TRUE),
    sd_imd   = sd(imd_score, na.rm = TRUE),
    .groups = "drop"
  )

cor_df <- gvi_data %>%
  dplyr::group_by(borough) %>%
  dplyr::summarise(
    correlation = safe_cor(mean_gvi, imd_score),
    .groups = "drop"
  )

borough_stats <- dplyr::left_join(stats_df, cor_df, by = "borough") %>%
  dplyr::arrange(desc(mean_gvi))

readr::write_csv(borough_stats, file.path(OUT_DIR, "table_4_5_borough_statistics.csv"))
print(borough_stats)



borough_anova <- aov(mean_gvi ~ borough, data = gvi_data)
print(summary(borough_anova))

conflicted::conflict_prefer("lmer", "lmerTest")

# Multilevel models
mlm_null  <- lmer(mean_gvi ~ 1 + (1 | borough), data = gvi_data, REML = FALSE)
mlm_imd   <- lmer(mean_gvi ~ imd_score + (1 | borough), data = gvi_data, REML = FALSE)

mlm_slope <- lmerTest::lmer(
  mean_gvi ~ imd_score + (1 + imd_score | borough),
  data = gvi_data, REML = FALSE,
  control = lme4::lmerControl(optimizer = "bobyqa",
                              optCtrl = list(maxfun = 1e5))
)

model_comp <- anova(mlm_null, mlm_imd, mlm_slope)
print(model_comp)

# ICC (null model)
icc_null <- performance::icc(mlm_null)
cat("\nICC (adjusted):", round(icc_null$ICC_adjusted, 3), "\n")

# Figure 4.5a: Borough-specific regression lines
fig_4_5a <- ggplot(gvi_data, aes(x = imd_score, y = mean_gvi)) +
  geom_point(alpha = 0.3, size = 1) +
  geom_smooth(aes(group = borough, color = borough), method = "lm", se = FALSE, linewidth = 0.8) +
  geom_smooth(method = "lm", color = "black", linewidth = 1.5, linetype = "dashed", se = FALSE) +
  scale_color_viridis_d() +
  labs(title = "Borough-Specific GVI ~ IMD Relationships",
       subtitle = "Colored: Borough-specific | Black dashed: Overall",
       x = "IMD Score", y = "GVI (%)") +
  theme(plot.title = element_text(face = "bold", size = 14),
        legend.position = "none")

ggsave(file.path(FIG_DIR, "fig_4_5a_borough_regression_lines.png"),
       fig_4_5a, width = 12, height = 8, dpi = 300)

# Figure 4.5b: Borough random effects (from random slopes model)
re_mat <- ranef(mlm_slope)$borough
borough_effects <- tibble::tibble(
  borough = rownames(re_mat),
  intercept_effect = re_mat[,1],
  slope_effect     = re_mat[,2]
)

fig_4_5b <- borough_effects %>%
  tidyr::pivot_longer(cols = c(intercept_effect, slope_effect),
                      names_to = "effect_type", values_to = "effect") %>%
  dplyr::mutate(effect_type = factor(effect_type,
                                     levels = c("intercept_effect","slope_effect"),
                                     labels = c("Intercept","Slope"))) %>%
  ggplot(aes(x = reorder(borough, effect), y = effect, fill = effect_type)) +
  geom_col(position = "dodge", alpha = 0.85) +
  coord_flip() +
  scale_fill_viridis_d(begin = 0.3, end = 0.7) +
  labs(title = "Borough Random Effects", subtitle = "Deviations from overall intercept and slope",
       x = "Borough", y = "Random effect", fill = "Effect") +
  theme(plot.title = element_text(face = "bold", size = 14))


ggsave(file.path(FIG_DIR, "fig_4_5b_borough_random_effects.png"),
       fig_4_5b, width = 10, height = 8, dpi = 300)

# ------------- 7. SPATIAL REGRESSION (4.4.4) --------------------------------------------------
cat("\n========== SPATIAL REGRESSION (4.4.4) ==========\n")

# Drop geometry for OLS to avoid method dispatch confusion
df_sp <- st_drop_geometry(spatial_data)

# Baseline OLS
ols_model <- lm(mean_gvi ~ imd_score, data = df_sp)

# SAR (spatial lag) & SEM (spatial error)
sar_model <- lagsarlm(mean_gvi ~ imd_score, data = df_sp, listw = W_queen, method = "eigen", quiet = TRUE)
sem_model <- errorsarlm(mean_gvi ~ imd_score, data = df_sp, listw = W_queen, method = "eigen", quiet = TRUE)

# SDM (Durbin)
sdm_model <- lagsarlm(mean_gvi ~ imd_score, data = df_sp, listw = W_queen, type = "mixed", method = "eigen", quiet = TRUE)

# Extract p for spatial params robustly (Wald test stored in summary())
sar_sum <- summary(sar_model)
sem_sum <- summary(sem_model)
sdm_sum <- summary(sdm_model)

p_rho_sar <- tryCatch({ as.numeric(sar_sum$Wald1["Pr(>z)"][1]) }, error = function(e) NA_real_)
p_lam_sem <- tryCatch({ as.numeric(sem_sum$Wald1["Pr(>z)"][1]) }, error = function(e) NA_real_)
p_rho_sdm <- tryCatch({ as.numeric(sdm_sum$Wald1["Pr(>z)"][1]) }, error = function(e) NA_real_)

spatial_model_comp <- tibble::tibble(
  Model      = c("OLS", "Spatial Lag (SAR)", "Spatial Error (SEM)", "Spatial Durbin (SDM)"),
  AIC        = c(AIC(ols_model), AIC(sar_model), AIC(sem_model), AIC(sdm_model)),
  LogLik     = c(as.numeric(logLik(ols_model)), as.numeric(logLik(sar_model)),
                 as.numeric(logLik(sem_model)), as.numeric(logLik(sdm_model))),
  Rho_Lambda = c(NA, sar_model$rho, sem_model$lambda, sdm_model$rho),
  p_spatial  = c(NA, p_rho_sar, p_lam_sem, p_rho_sdm)
)

readr::write_csv(spatial_model_comp, file.path(OUT_DIR, "table_4_6_spatial_models.csv"))
print(spatial_model_comp)

best_spatial <- c("OLS","SAR","SEM","SDM")[which.min(spatial_model_comp$AIC)]
cat("\n--- Best (by AIC):", best_spatial, "---\n")

if (best_spatial == "SAR") {
  print(sar_sum)
  impacts_sar <- impacts(sar_model, listw = W_queen, R = 500) # direct/indirect/total
  cat("\n--- SAR Impacts (500 sims) ---\n")
  print(summary(impacts_sar, zstats = TRUE, short = TRUE))
  moran_sar_resid <- moran.test(residuals(sar_model), W_queen, zero.policy = TRUE)
  cat("\nMoran's I (SAR residuals):", round(unname(moran_sar_resid$estimate[1]), 4),
      ", p =", format.pval(moran_sar_resid$p.value), "\n")
} else if (best_spatial == "SEM") {
  print(sem_sum)
} else if (best_spatial == "SDM") {
  print(sdm_sum)
}

# ------------- 8. GEOGRAPHICALLY WEIGHTED REGRESSION (4.4.4 cont.) ----------------------------
cat("\n========== GEOGRAPHICALLY WEIGHTED REGRESSION (GWR) ==========\n")

gw_data <- as(spatial_data, "Spatial")

cat("Finding optimal adaptive bandwidth via CV ...\n")
bw_adapt <- bw.gwr(mean_gvi ~ imd_score, data = gw_data, approach = "CV",
                   kernel = "gaussian", adaptive = TRUE)

cat("Optimal bandwidth (neighbors):", round(bw_adapt, 0), "\n")

gwr_model <- gwr.basic(mean_gvi ~ imd_score, data = gw_data,
                       bw = bw_adapt, kernel = "gaussian", adaptive = TRUE)

print(gwr_model)

gwr_results <- as.data.frame(gwr_model$SDF)
spatial_data$gwr_intercept <- gwr_results$Intercept
spatial_data$gwr_slope     <- gwr_results$imd_score
spatial_data$gwr_r2        <- gwr_results$Local_R2

gwr_coef_summary <- tibble::tibble(
  Coefficient = c("Intercept","IMD Slope","Local R2"),
  Min    = c(min(spatial_data$gwr_intercept, na.rm = TRUE),
             min(spatial_data$gwr_slope, na.rm = TRUE),
             min(spatial_data$gwr_r2, na.rm = TRUE)),
  Q1     = c(quantile(spatial_data$gwr_intercept, 0.25, na.rm = TRUE),
             quantile(spatial_data$gwr_slope, 0.25, na.rm = TRUE),
             quantile(spatial_data$gwr_r2, 0.25, na.rm = TRUE)),
  Median = c(median(spatial_data$gwr_intercept, na.rm = TRUE),
             median(spatial_data$gwr_slope, na.rm = TRUE),
             median(spatial_data$gwr_r2, na.rm = TRUE)),
  Q3     = c(quantile(spatial_data$gwr_intercept, 0.75, na.rm = TRUE),
             quantile(spatial_data$gwr_slope, 0.75, na.rm = TRUE),
             quantile(spatial_data$gwr_r2, 0.75, na.rm = TRUE)),
  Max    = c(max(spatial_data$gwr_intercept, na.rm = TRUE),
             max(spatial_data$gwr_slope, na.rm = TRUE),
             max(spatial_data$gwr_r2, na.rm = TRUE))
)
print(gwr_coef_summary)
readr::write_csv(gwr_coef_summary, file.path(OUT_DIR, "gwr_coefficient_summary.csv"))

# Figure 4.6a: GWR Slope (IMD coefficient)
gwr_slope_map <- tm_shape(spatial_data) +
  tm_polygons("gwr_slope",
              style = "quantile", n = 5,
              palette = "-RdBu", midpoint = 0,
              title = "IMD Coefficient", border.alpha = 0.3) +
  tm_layout(title = "GWR: Local IMD Coefficients",
            title.size = 1.2, title.fontface = "bold",
            legend.position = c("right","bottom"), frame = FALSE)

tmap_save(gwr_slope_map, file.path(FIG_DIR, "fig_4_6a_gwr_coefficients.png"),
          width = 10, height = 8, dpi = 300)

# Figure 4.6b: Local R2
gwr_r2_map <- tm_shape(spatial_data) +
  tm_polygons("gwr_r2",
              style = "quantile", n = 5,
              palette = "YlOrRd", title = "Local R\u00B2",
              border.alpha = 0.3) +
  tm_layout(title = "GWR: Local R\u00B2",
            title.size = 1.2, title.fontface = "bold",
            legend.position = c("right","bottom"), frame = FALSE)

tmap_save(gwr_r2_map, file.path(FIG_DIR, "fig_4_6b_gwr_local_r2.png"),
          width = 10, height = 8, dpi = 300)

cat("\nSpatial variability of IMD slope (CV):",
    round(sd(spatial_data$gwr_slope, na.rm = TRUE) / mean(abs(spatial_data$gwr_slope), na.rm = TRUE), 3), "\n")
cat("Range of IMD coefficients:", round(min(spatial_data$gwr_slope, na.rm = TRUE), 3),
    "to", round(max(spatial_data$gwr_slope, na.rm = TRUE), 3), "\n")

# ------------- 9. SYNTHESIS (4.7) -------------------------------------------------------------
cat("\n========== MODEL COMPARISON SYNTHESIS (4.7) ==========\n")

# Pseudo-R2 for spatial (vs OLS) and GWR average local R2
ols_r2 <- summary(ols_model)$r.squared
sar_r2 <- tryCatch({
  1 - exp(-2 * (as.numeric(logLik(sar_model)) - as.numeric(logLik(ols_model))) / nrow(df_sp))
}, error = function(e) NA_real_)
gwr_r2_mean <- mean(spatial_data$gwr_r2, na.rm = TRUE)

# R2 for mixed model
r2_mlm <- performance::r2(mlm_slope)  # returns Marginal/Conditional
mlm_R2_marg <- as.numeric(r2_mlm$R2_marginal)
mlm_R2_cond <- as.numeric(r2_mlm$R2_conditional)

final_comparison <- tibble::tibble(
  Model = c("OLS", "Spatial Lag (SAR)", "GWR", "Multilevel (Random slopes)"),
  Global_R2 = c(ols_r2, sar_r2, gwr_r2_mean, mlm_R2_marg),
  AIC       = c(AIC(ols_model), AIC(sar_model), gwr_model$GW.diagnostic$AIC, AIC(mlm_slope)),
  Morans_I_Residuals = c(
    unname(moran.test(residuals(ols_model), W_queen, zero.policy = TRUE)$estimate[1]),
    unname(moran.test(residuals(sar_model), W_queen, zero.policy = TRUE)$estimate[1]),
    NA_real_,  # GWR has local residuals only
    NA_real_   # MLM not spatial
  ),
  Key_Finding = c(
    "Negative GVI~IMD relation; spatial autocorrelation in residuals",
    "Captures spatial spillovers; residual autocorrelation reduced",
    "Relationship varies in space (heterogeneity)",
    "Borough-level heterogeneity; random slopes improve fit"
  )
)

readr::write_csv(final_comparison, file.path(OUT_DIR, "table_4_7_final_comparison.csv"))
print(final_comparison)

# ------------- 9.5 MASTER RESULTS EXPORT (one Excel workbook) -----------------
if (!requireNamespace("writexl", quietly = TRUE)) install.packages("writexl")
library(writexl)

read_if_exists <- function(path) {
  if (file.exists(path)) {
    tryCatch(readr::read_csv(path, show_col_types = FALSE),
             error = function(e) tibble::tibble(note = paste("Failed to read:", basename(path))))
  } else {
    tibble::tibble(note = paste("Missing:", basename(path)))
  }
}

# 已在前文写出的表（直接读回）
path_tbl44  <- file.path(OUT_DIR, "table_4_4_global_morans_i.csv")
path_lisa_s <- file.path(OUT_DIR, "lisa_cluster_summary.csv")
path_hot    <- file.path(OUT_DIR, "lisa_green_inequality_hotspots.csv")
path_LH_N   <- file.path(OUT_DIR, "lisa_LH_topN_by_borough.csv")
path_LH_N_d <- file.path(OUT_DIR, "lisa_LH_topN_by_borough_deprived.csv")
path_tbl45  <- file.path(OUT_DIR, "table_4_5_borough_statistics.csv")
path_tbl46  <- file.path(OUT_DIR, "table_4_6_spatial_models.csv")
path_gwr    <- file.path(OUT_DIR, "gwr_coefficient_summary.csv")
path_tbl47  <- file.path(OUT_DIR, "table_4_7_final_comparison.csv")

# ——从内存对象补充的几个结果——
# 1) SAR 影响分解（若最佳模型为 SAR）
sar_impacts_tbl <- tryCatch({
  if (exists("impacts_sar")) {
    tibble::tibble(
      variable = names(impacts_sar$direct),
      direct   = as.numeric(impacts_sar$direct),
      indirect = as.numeric(impacts_sar$indirect),
      total    = as.numeric(impacts_sar$total)
    )
  } else {
    tibble::tibble(note = "Best model is not SAR or impacts not computed.")
  }
}, error = function(e) tibble::tibble(note = paste("SAR impacts unavailable:", e$message)))

# 2) 残差 Moran's I（OLS 与最佳空间模型）
resid_moran_tbl <- tryCatch({
  mi_ols <- moran.test(residuals(ols_model), W_queen, zero.policy = TRUE)
  if (exists("sar_model")) {
    mi_best <- moran.test(residuals(sar_model), W_queen, zero.policy = TRUE)
    tibble::tibble(
      model    = c("OLS","SAR"),
      Moran_I  = c(unname(mi_ols$estimate[1]), unname(mi_best$estimate[1])),
      p_value  = c(mi_ols$p.value, mi_best$p.value)
    )
  } else {
    tibble::tibble(
      model    = "OLS",
      Moran_I  = unname(mi_ols$estimate[1]),
      p_value  = mi_ols$p.value
    )
  }
}, error = function(e) tibble::tibble(note = paste("Residual Moran unavailable:", e$message)))

# 3) MLM 比较（把 anova 对象转成表）
mlm_comp_tbl <- tryCatch({
  if (exists("model_comp")) {
    mc <- as.data.frame(model_comp)
    tibble::as_tibble(mc, .name_repair = "unique")
  } else {
    tibble::tibble(note = "model_comp not found.")
  }
}, error = function(e) tibble::tibble(note = paste("MLM comparison unavailable:", e$message)))

# 4) Borough 随机效应（若已计算）
re_tbl <- tryCatch({
  if (exists("borough_effects")) tibble::as_tibble(borough_effects) else tibble::tibble(note = "borough_effects not found.")
}, error = function(e) tibble::tibble(note = paste("Random effects unavailable:", e$message)))

# 5) 元信息
meta_tbl <- tibble::tibble(
  timestamp = as.character(Sys.time()),
  data_path = DATA_PATH,
  shapefile = SHAPEFILE_PATH,
  N_TOPN    = N_TOPN,
  IMD_Q     = IMD_Q,
  n_obs     = nrow(spatial_data),
  best_spatial = if (exists("best_spatial")) best_spatial else NA_character_
)

# ——汇总为一个 Excel（多 Sheet）——
sheets <- list(
  "Table_4_4_Moran"          = read_if_exists(path_tbl44),
  "LISA_summary"             = read_if_exists(path_lisa_s),
  "LISA_hotspots_LH"         = read_if_exists(path_hot),
  "LISA_TopN_by_borough"     = read_if_exists(path_LH_N),
  "LISA_TopN_deprived"       = read_if_exists(path_LH_N_d),
  "Table_4_5_Borough_stats"  = read_if_exists(path_tbl45),
  "MLM_model_comp"           = mlm_comp_tbl,
  "MLM_random_effects"       = re_tbl,
  "Table_4_6_Spatial_models" = read_if_exists(path_tbl46),
  "SAR_impacts"              = sar_impacts_tbl,
  "GWR_coef_summary"         = read_if_exists(path_gwr),
  "Table_4_7_Final_compare"  = read_if_exists(path_tbl47),
  "Meta"                     = meta_tbl
)

writexl::write_xlsx(sheets, path = file.path(OUT_DIR, "RQ2_master_results.xlsx"))
cat("\nMaster workbook written to:", file.path(OUT_DIR, "RQ2_master_results.xlsx"), "\n")
# -------------------------------------------------------------------------------


# ------------- 10. OUTPUT MAPPING -------------------------------------------------------------
cat("\n========== OUTPUT MAPPING ==========\n")
cat("Section 4.4.1 - Global Spatial Autocorrelation\n")
cat("  - Table 4.4  : ", file.path(OUT_DIR, "table_4_4_global_morans_i.csv"), "\n")
cat("  - Figure 4.4a: ", file.path(FIG_DIR, "fig_4_4a_moran_scatterplot.png"), "\n")
cat("  - Geary's C  : printed above\n\n")

cat("Section 4.4.2 - LISA\n")
cat("  - LISA summary             : ", file.path(OUT_DIR, "lisa_cluster_summary.csv"), "\n")
cat("  - Hotspots (Low-High)      : ", file.path(OUT_DIR, "lisa_green_inequality_hotspots.csv"), "\n")
cat("  - Top-N LH by Borough      : ", file.path(OUT_DIR, "lisa_LH_topN_by_borough.csv"), "\n")
cat("  - Top-N LH (Deprived only) : ", file.path(OUT_DIR, "lisa_LH_topN_by_borough_deprived.csv"), "\n")
cat("  - Figure 4.4b (clusters)   : ", file.path(FIG_DIR, "fig_4_4b_lisa_clusters.png"), "\n")
cat("  - Figure 4.4c (signif, FDR): ", file.path(FIG_DIR, "fig_4_4c_lisa_significance.png"), "\n\n")

cat("Section 4.4.3 - Borough level\n")
cat("  - Table 4.5                 : ", file.path(OUT_DIR, "table_4_5_borough_statistics.csv"), "\n")
cat("  - Figure 4.5a (reg lines)   : ", file.path(FIG_DIR, "fig_4_5a_borough_regression_lines.png"), "\n")
cat("  - Figure 4.5b (rand effects): ", file.path(FIG_DIR, "fig_4_5b_borough_random_effects.png"), "\n")
cat("  - ICC (null) printed above\n\n")

cat("Section 4.4.4 - Spatial regression\n")
cat("  - Table 4.6 : ", file.path(OUT_DIR, "table_4_6_spatial_models.csv"), "\n")
cat("  - Best model & impacts printed above\n\n")

cat("Section 4.4.4 (cont.) - GWR\n")
cat("  - GWR coeff summary : ", file.path(OUT_DIR, "gwr_coefficient_summary.csv"), "\n")
cat("  - Figure 4.6a (coef): ", file.path(FIG_DIR, "fig_4_6a_gwr_coefficients.png"), "\n")
cat("  - Figure 4.6b (R2)  : ", file.path(FIG_DIR, "fig_4_6b_gwr_local_r2.png"), "\n\n")

cat("Section 4.7 - Synthesis\n")
cat("  - Table 4.7 : ", file.path(OUT_DIR, "table_4_7_final_comparison.csv"), "\n\n")

cat("========== COMPLETE ==========\n")
cat("Tables written to :", OUT_DIR, "\n")
cat("Figures written to:", FIG_DIR, "\n")

# Save workspace (optional)
save.image(file.path(OUT_DIR, "RQ2_spatial_analysis_workspace.RData"))
