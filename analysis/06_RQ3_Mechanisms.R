#!/usr/bin/env Rscript
# =============================================================================
# City Look Dissertation - RQ3 FINAL ANALYSIS
# Mechanisms and Moderators of the GVI-IMD Relationship
# Author: [Your Name]
# Date: 2025-08-11
# =============================================================================

# Load required libraries
suppressPackageStartupMessages({
  library(tidyverse)
  library(here)
  library(corrplot)
  library(interactions)  # For interaction plots
  library(mediation)     # For mediation analysis
  library(performance)   # Model diagnostics
  library(see)          # Visualization
  library(patchwork)    # Combine plots
  library(stargazer)    # Tables
  library(knitr)
  library(kableExtra)
})

cat("\n")
cat("================================================================\n")
cat("       RQ3: MECHANISMS OF THE GVI-IMD RELATIONSHIP\n")
cat("================================================================\n")

# =============================================================================
# SECTION 4.5: DATA INTEGRATION
# =============================================================================

cat("\n========== SECTION 4.5: DATA INTEGRATION ==========\n")

# Set paths
DATA_DIR <- here("data", "processed", "rq3")
OUTPUT_DIR <- here("output", "rq3")
FIG_DIR <- here("figures", "rq3")

# Create directories
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(FIG_DIR, recursive = TRUE, showWarnings = FALSE)

# ---- 4.5.1 Load all datasets ----
cat("\n--- 4.5.1: Loading Datasets ---\n")

# Main dataset
main_data <- read_csv(
  here("output", "full_london_gvi", "inner_london_all_lsoas_complete.csv"),
  show_col_types = FALSE
)
cat(sprintf("✓ Main dataset: %d LSOAs\n", nrow(main_data)))

# Urban form variables
pop_density <- read_csv(file.path(DATA_DIR, "population_density_lsoa_from_TS006.csv"), 
                        show_col_types = FALSE)
ptal_scores <- read_csv(file.path(DATA_DIR, "ptal_scores_lsoa.csv"), 
                        show_col_types = FALSE)
road_density <- read_csv(file.path(DATA_DIR, "road_network_density_lsoa_clean.csv"), 
                         show_col_types = FALSE)
building_density <- read_csv(file.path(DATA_DIR, "land_use_building_density_inner_london.csv"), 
                             show_col_types = FALSE)

cat(sprintf("✓ Population density: %d records\n", nrow(pop_density)))
cat(sprintf("✓ PTAL scores: %d records\n", nrow(ptal_scores)))
cat(sprintf("✓ Road density: %d records\n", nrow(road_density)))
cat(sprintf("✓ Building density: %d records\n", nrow(building_density)))

# ---- 4.5.2 Merge datasets ----
cat("\n--- 4.5.2: Merging Datasets ---\n")

rq3_data <- main_data %>%
  # Add population density
  left_join(pop_density, by = "lsoa_code") %>%
  # Add PTAL (select key variables)
  left_join(ptal_scores %>% 
              select(lsoa_code, ptal_score, ptal_category, accessibility_index),
            by = "lsoa_code") %>%
  # Add road density
  left_join(road_density %>%
              select(lsoa_code, road_density_km_per_sqkm),
            by = "lsoa_code") %>%
  # Add building density
  left_join(building_density, by = "lsoa_code")

cat(sprintf("Merged dataset: %d rows × %d columns\n", nrow(rq3_data), ncol(rq3_data)))

# ---- 4.5.3 Handle missing values ----
cat("\n--- 4.5.3: Handling Missing Values ---\n")

# Check missingness
missing_summary <- rq3_data %>%
  summarise(across(c(population_density, ptal_score, road_density_km_per_sqkm, 
                     building_density),
                   ~sum(is.na(.)), .names = "n_missing_{.col}")) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "n_missing") %>%
  mutate(
    variable = str_remove(variable, "n_missing_"),
    pct_missing = round(n_missing / nrow(rq3_data) * 100, 1)
  )

print(missing_summary)

# Impute missing values with borough means
borough_means <- rq3_data %>%
  group_by(borough) %>%
  summarise(across(c(population_density, building_density, road_density_km_per_sqkm),
                   ~mean(., na.rm = TRUE), .names = "borough_{.col}"))

# Apply imputation
rq3_data <- rq3_data %>%
  left_join(borough_means, by = "borough") %>%
  mutate(
    population_density = coalesce(population_density, borough_population_density),
    building_density = coalesce(building_density, borough_building_density),
    road_density_km_per_sqkm = coalesce(road_density_km_per_sqkm, borough_road_density_km_per_sqkm)
  ) %>%
  select(-starts_with("borough_"))

# Verify no missing values in key variables
n_complete <- sum(complete.cases(rq3_data %>% 
                                   select(mean_gvi, imd_score, population_density, 
                                          ptal_score, road_density_km_per_sqkm, building_density)))
cat(sprintf("Complete cases after imputation: %d (%.1f%%)\n", 
            n_complete, n_complete/nrow(rq3_data)*100))

# ---- 4.5.4 Create analysis variables ----
cat("\n--- 4.5.4: Creating Analysis Variables ---\n")

rq3_data <- rq3_data %>%
  mutate(
    # Standardized variables (z-scores)
    gvi_z = scale(mean_gvi)[,1],
    imd_z = scale(imd_score)[,1],
    pop_density_z = scale(population_density)[,1],
    ptal_z = scale(ptal_score)[,1],
    road_density_z = scale(road_density_km_per_sqkm)[,1],
    building_density_z = scale(building_density)[,1],
    
    # Log transformations for skewed variables
    log_pop_density = log(population_density + 1),
    log_building_density = log(building_density + 1),
    
    # Categorical versions for visualization
    pop_density_tertile = ntile(population_density, 3),
    pop_density_cat = factor(pop_density_tertile, 
                             labels = c("Low", "Medium", "High")),
    
    ptal_cat_simple = case_when(
      ptal_score <= 2 ~ "Poor",
      ptal_score <= 4 ~ "Moderate",  
      ptal_score > 4 ~ "Good",
      TRUE ~ "Moderate"
    ),
    
    building_density_tertile = ntile(building_density, 3),
    building_density_cat = factor(building_density_tertile,
                                  labels = c("Low", "Medium", "High"))
  )

# Save integrated dataset
write_csv(rq3_data, file.path(OUTPUT_DIR, "rq3_integrated_data.csv"))
cat(sprintf("✓ Saved integrated dataset: %d LSOAs with %d variables\n", 
            nrow(rq3_data), ncol(rq3_data)))

# =============================================================================
# SECTION 4.6: DESCRIPTIVE STATISTICS (Table 4.8)
# =============================================================================

cat("\n========== SECTION 4.6: DESCRIPTIVE STATISTICS ==========\n")

# ---- Table 4.8: Urban form variables descriptive statistics ----
desc_stats <- rq3_data %>%
  select(
    `Population Density` = population_density,
    `PTAL Score` = ptal_score,
    `Road Density` = road_density_km_per_sqkm,
    `Building Density` = building_density
  ) %>%
  pivot_longer(everything(), names_to = "Variable", values_to = "Value") %>%
  group_by(Variable) %>%
  summarise(
    N = n(),
    Mean = mean(Value, na.rm = TRUE),
    SD = sd(Value, na.rm = TRUE),
    Min = min(Value, na.rm = TRUE),
    Q25 = quantile(Value, 0.25, na.rm = TRUE),
    Median = median(Value, na.rm = TRUE),
    Q75 = quantile(Value, 0.75, na.rm = TRUE),
    Max = max(Value, na.rm = TRUE),
    .groups = 'drop'
  ) %>%
  mutate(across(where(is.numeric), ~round(., 2)))

cat("\n--- Table 4.8: Urban Form Variables Descriptive Statistics ---\n")
print(desc_stats)
write_csv(desc_stats, file.path(OUTPUT_DIR, "table_4_8_descriptive_stats.csv"))

# ---- Correlation matrix ----
cor_matrix <- rq3_data %>%
  select(GVI = mean_gvi, IMD = imd_score, 
         PopDensity = population_density, PTAL = ptal_score,
         RoadDensity = road_density_km_per_sqkm, 
         BuildingDensity = building_density) %>%
  cor(use = "pairwise.complete.obs")

cat("\n--- Correlation Matrix ---\n")
print(round(cor_matrix, 3))

# Figure 4.7a: Correlation heatmap
png(file.path(FIG_DIR, "fig_4_7a_correlation_matrix.png"), 
    width = 8, height = 8, units = "in", res = 300)
corrplot(cor_matrix, method = "color", type = "upper",
         addCoef.col = "black", tl.col = "black", tl.srt = 45,
         title = "Correlation Matrix of Key Variables", 
         mar = c(0,0,2,0))
dev.off()

# =============================================================================
# SECTION 4.7: INTERACTION EFFECTS ANALYSIS (Table 4.9)
# =============================================================================

cat("\n========== SECTION 4.7: INTERACTION EFFECTS ==========\n")

# ---- Model 1: Base model (no interactions) ----
m1_base <- lm(mean_gvi ~ imd_score, data = rq3_data)

# ---- Model 2: Main effects only ----
m2_main <- lm(mean_gvi ~ imd_score + population_density + ptal_score + 
                road_density_km_per_sqkm + building_density, 
              data = rq3_data)

# ---- Model 3: Population density interaction ----
m3_pop <- lm(mean_gvi ~ imd_score * population_density + ptal_score + 
               road_density_km_per_sqkm + building_density, 
             data = rq3_data)

# ---- Model 4: PTAL interaction ----
m4_ptal <- lm(mean_gvi ~ imd_score * ptal_score + population_density + 
                road_density_km_per_sqkm + building_density, 
              data = rq3_data)

# ---- Model 5: Building density interaction ----
m5_building <- lm(mean_gvi ~ imd_score * building_density + population_density + 
                    ptal_score + road_density_km_per_sqkm, 
                  data = rq3_data)

# ---- Model 6: All interactions ----
m6_all <- lm(mean_gvi ~ imd_score * (population_density + ptal_score + building_density) + 
               road_density_km_per_sqkm, 
             data = rq3_data)

# ---- Table 4.9: Interaction models comparison ----
model_comparison <- data.frame(
  Model = c("M1: Base", "M2: Main Effects", "M3: IMD×PopDensity", 
            "M4: IMD×PTAL", "M5: IMD×Building", "M6: All Interactions"),
  R2 = c(summary(m1_base)$r.squared, summary(m2_main)$r.squared,
         summary(m3_pop)$r.squared, summary(m4_ptal)$r.squared,
         summary(m5_building)$r.squared, summary(m6_all)$r.squared),
  Adj_R2 = c(summary(m1_base)$adj.r.squared, summary(m2_main)$adj.r.squared,
             summary(m3_pop)$adj.r.squared, summary(m4_ptal)$adj.r.squared,
             summary(m5_building)$adj.r.squared, summary(m6_all)$adj.r.squared),
  AIC = c(AIC(m1_base), AIC(m2_main), AIC(m3_pop), 
          AIC(m4_ptal), AIC(m5_building), AIC(m6_all)),
  BIC = c(BIC(m1_base), BIC(m2_main), BIC(m3_pop), 
          BIC(m4_ptal), BIC(m5_building), BIC(m6_all))
) %>%
  mutate(across(where(is.numeric), ~round(., 4)))

cat("\n--- Table 4.9: Interaction Models Comparison ---\n")
print(model_comparison)
write_csv(model_comparison, file.path(OUTPUT_DIR, "table_4_9_interaction_models.csv"))

# Identify best model
best_model_idx <- which.min(model_comparison$AIC)
cat(sprintf("\nBest model (by AIC): %s\n", model_comparison$Model[best_model_idx]))

# ---- Figure 4.7b: Interaction plots ----
library(ggplot2)

# Population density interaction
p1 <- interact_plot(m3_pop, pred = imd_score, modx = population_density,
                    modx.values = "terciles", plot.points = TRUE,
                    x.label = "IMD Score", y.label = "GVI (%)",
                    legend.main = "Population\nDensity") +
  ggtitle("A. IMD × Population Density") +
  theme_minimal()

# PTAL interaction
p2 <- interact_plot(m4_ptal, pred = imd_score, modx = ptal_score,
                    modx.values = c(1, 3, 5), plot.points = TRUE,
                    x.label = "IMD Score", y.label = "GVI (%)",
                    legend.main = "PTAL\nScore") +
  ggtitle("B. IMD × PTAL Score") +
  theme_minimal()

# Building density interaction
p3 <- interact_plot(m5_building, pred = imd_score, modx = building_density,
                    modx.values = "terciles", plot.points = TRUE,
                    x.label = "IMD Score", y.label = "GVI (%)",
                    legend.main = "Building\nDensity") +
  ggtitle("C. IMD × Building Density") +
  theme_minimal()

# Combine plots
combined_interactions <- p1 + p2 + p3 +
  plot_annotation(title = "Figure 4.7b: Interaction Effects on GVI-IMD Relationship",
                  theme = theme(plot.title = element_text(size = 14, face = "bold")))

ggsave(file.path(FIG_DIR, "fig_4_7b_interaction_plots.png"),
       combined_interactions, width = 15, height = 5, dpi = 300)

# =============================================================================
# SECTION 4.8: MEDIATION ANALYSIS (Table 4.10)
# =============================================================================

cat("\n========== SECTION 4.8: MEDIATION ANALYSIS ==========\n")

# Test if urban form variables mediate IMD → GVI relationship

# ---- Mediation 1: Population density as mediator ----
cat("\n--- Testing Population Density as Mediator ---\n")

# Step 1: IMD → Population Density
med1_a <- lm(population_density_z ~ imd_z, data = rq3_data)
# Step 2: IMD + Pop Density → GVI
med1_b <- lm(gvi_z ~ imd_z + population_density_z, data = rq3_data)
# Step 3: Total effect
med1_c <- lm(gvi_z ~ imd_z, data = rq3_data)

# Mediation test
med1 <- mediate(med1_a, med1_b, treat = "imd_z", mediator = "population_density_z",
                sims = 1000, boot = TRUE)

# ---- Mediation 2: PTAL as mediator ----
cat("\n--- Testing PTAL as Mediator ---\n")

med2_a <- lm(ptal_z ~ imd_z, data = rq3_data)
med2_b <- lm(gvi_z ~ imd_z + ptal_z, data = rq3_data)
med2 <- mediate(med2_a, med2_b, treat = "imd_z", mediator = "ptal_z",
                sims = 1000, boot = TRUE)

# ---- Mediation 3: Building density as mediator ----
cat("\n--- Testing Building Density as Mediator ---\n")

med3_a <- lm(building_density_z ~ imd_z, data = rq3_data)
med3_b <- lm(gvi_z ~ imd_z + building_density_z, data = rq3_data)
med3 <- mediate(med3_a, med3_b, treat = "imd_z", mediator = "building_density_z",
                sims = 1000, boot = TRUE)

# ---- Table 4.10: Mediation results ----
mediation_results <- data.frame(
  Mediator = c("Population Density", "PTAL Score", "Building Density"),
  ACME = c(med1$d0, med2$d0, med3$d0),
  ACME_CI_Lower = c(med1$d0.ci[1], med2$d0.ci[1], med3$d0.ci[1]),
  ACME_CI_Upper = c(med1$d0.ci[2], med2$d0.ci[2], med3$d0.ci[2]),
  ADE = c(med1$z0, med2$z0, med3$z0),
  Total_Effect = c(med1$tau.coef, med2$tau.coef, med3$tau.coef),
  Prop_Mediated = c(med1$n0, med2$n0, med3$n0),
  p_value = c(med1$d0.p, med2$d0.p, med3$d0.p)
) %>%
  mutate(across(where(is.numeric), ~round(., 4)))

cat("\n--- Table 4.10: Mediation Analysis Results ---\n")
print(mediation_results)
write_csv(mediation_results, file.path(OUTPUT_DIR, "table_4_10_mediation_results.csv"))

# =============================================================================
# SECTION 4.9: DATA BIAS ANALYSIS (Table 4.11)
# =============================================================================

cat("\n========== SECTION 4.9: DATA BIAS ANALYSIS ==========\n")

# Compare measured vs predicted GVI
bias_analysis <- rq3_data %>%
  group_by(data_source) %>%
  summarise(
    n = n(),
    mean_gvi = mean(mean_gvi),
    mean_imd = mean(imd_score),
    mean_pop_density = mean(population_density, na.rm = TRUE),
    mean_ptal = mean(ptal_score, na.rm = TRUE),
    .groups = 'drop'
  )

cat("\n--- Measured vs Predicted Comparison ---\n")
print(bias_analysis)

# T-test for differences
t_test_gvi <- t.test(mean_gvi ~ data_source, data = rq3_data)
cat(sprintf("\nGVI difference: t = %.3f, p = %.4f\n", 
            t_test_gvi$statistic, t_test_gvi$p.value))

# Check if image count affects relationship
cor_images_imd <- cor.test(rq3_data$n_images, rq3_data$imd_score)
cat(sprintf("Correlation between image count and IMD: r = %.3f, p = %.4f\n",
            cor_images_imd$estimate, cor_images_imd$p.value))

# =============================================================================
# SECTION 4.10: FINAL INTEGRATED MODEL
# =============================================================================

cat("\n========== SECTION 4.10: FINAL INTEGRATED MODEL ==========\n")

# Best model based on previous analyses
final_model <- lm(mean_gvi ~ imd_score + population_density + ptal_score + 
                    building_density + road_density_km_per_sqkm +
                    imd_score:population_density + imd_score:ptal_score +
                    borough, 
                  data = rq3_data)

cat("\n--- Final Model Summary ---\n")
print(summary(final_model))

# Model diagnostics
cat("\n--- Model Diagnostics ---\n")
check_model_results <- check_model(final_model, check = c("vif", "normality", "linearity"))

# Save model results
final_model_summary <- broom::tidy(final_model) %>%
  mutate(across(where(is.numeric), ~round(., 4)))
write_csv(final_model_summary, file.path(OUTPUT_DIR, "final_model_coefficients.csv"))

# =============================================================================
# SUMMARY OUTPUT MAPPING
# =============================================================================

cat("\n========== OUTPUT MAPPING FOR DISSERTATION ==========\n")
cat("\nSECTION 4.5 - Data Integration:")
cat("\n  - Integrated dataset: output/rq3/rq3_integrated_data.csv")

cat("\n\nSECTION 4.6 - Descriptive Statistics:")
cat("\n  - Table 4.8: table_4_8_descriptive_stats.csv")
cat("\n  - Figure 4.7a: fig_4_7a_correlation_matrix.png")

cat("\n\nSECTION 4.7 - Interaction Effects:")
cat("\n  - Table 4.9: table_4_9_interaction_models.csv")
cat("\n  - Figure 4.7b: fig_4_7b_interaction_plots.png")

cat("\n\nSECTION 4.8 - Mediation Analysis:")
cat("\n  - Table 4.10: table_4_10_mediation_results.csv")

cat("\n\nSECTION 4.9 - Data Bias:")
cat("\n  - Bias analysis included in output")

cat("\n\nSECTION 4.10 - Final Model:")
cat("\n  - Model coefficients: final_model_coefficients.csv")

cat("\n\n========== RQ3 ANALYSIS COMPLETE ==========\n")

# Save workspace
save.image(file.path(OUTPUT_DIR, "rq3_analysis_workspace.RData"))
cat("Workspace saved to: output/rq3/rq3_analysis_workspace.RData\n")