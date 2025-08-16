#!/usr/bin/env Rscript
# =============================================================================
# City Look Dissertation - RQ1 Analysis (UPDATED for ggplot2 3.4+)
# Descriptive and Correlation Analysis for GVI-IMD Relationship
# Author: [Your Name]
# Date: 2025
# =============================================================================

# Load required libraries
library(tidyverse)
library(here)
library(sf)
library(ggplot2)
library(viridis)
library(corrplot)
library(car)
library(moments)
library(broom)
library(stargazer)
library(gridExtra)
library(knitr)
library(kableExtra)
library(psych)
library(mgcv)
library(segmented)
library(ggpubr)
library(scales)

# Set theme for all plots
theme_set(theme_minimal(base_size = 12))

# =============================================================================
# 1. DATA PREPARATION
# =============================================================================

cat("\n========== LOADING DATA ==========\n")

# ---- Read data (update path to your actual location) ----
DATA_PATH <- here("output", "full_london_gvi", "inner_london_all_lsoas_complete.csv")
gvi_data  <- readr::read_csv(DATA_PATH)

# Data cleaning and preparation
gvi_data <- gvi_data %>%
  mutate(
    # Ensure proper data types
    imd_quintile = factor(imd_quintile, 
                          levels = c("Q1_Least", "Q2", "Q3", "Q4", "Q5_Most"),
                          labels = c("Q1\n(Least)", "Q2", "Q3", "Q4", "Q5\n(Most)")),
    data_source = factor(data_source),
    borough = factor(borough),
    
    # Create additional variables for analysis
    imd_rank = rank(imd_score) / n(),  # Ranked IMD for inequality measures
    gvi_rank = rank(mean_gvi) / n()    # Ranked GVI for inequality measures
  ) %>%
  filter(!is.na(mean_gvi))  # Remove any missing values

cat(paste("Dataset loaded:", nrow(gvi_data), "LSOAs\n"))
cat(paste("Measured LSOAs:", sum(gvi_data$data_source == "measured"), "\n"))
cat(paste("Predicted LSOAs:", sum(gvi_data$data_source == "predicted"), "\n"))

# Ensure output dirs exist
if (!dir.exists(here("figures"))) dir.create(here("figures"), recursive = TRUE)
if (!dir.exists(here("output")))  dir.create(here("output"),  recursive = TRUE)

# =============================================================================
# 2. DESCRIPTIVE ANALYSIS (Section 4.2)
# =============================================================================

cat("\n========== DESCRIPTIVE ANALYSIS (Section 4.2) ==========\n")

# -----------------------------------------------------------------------------
# 2.1 Overall GVI Distribution (Section 4.2.1)
# -----------------------------------------------------------------------------

# Calculate comprehensive descriptive statistics
desc_stats_overall <- gvi_data %>%
  summarise(
    n = n(),
    mean = mean(mean_gvi, na.rm = TRUE),
    sd = sd(mean_gvi, na.rm = TRUE),
    median = median(mean_gvi, na.rm = TRUE),
    min = min(mean_gvi, na.rm = TRUE),
    max = max(mean_gvi, na.rm = TRUE),
    q25 = quantile(mean_gvi, 0.25, na.rm = TRUE),
    q75 = quantile(mean_gvi, 0.75, na.rm = TRUE),
    iqr = IQR(mean_gvi, na.rm = TRUE),
    skewness = skewness(mean_gvi, na.rm = TRUE),
    kurtosis = kurtosis(mean_gvi, na.rm = TRUE),
    se = sd(mean_gvi, na.rm = TRUE) / sqrt(n())
  )

# Normality test
shapiro_test <- shapiro.test(sample(gvi_data$mean_gvi, 
                                    min(5000, nrow(gvi_data))))

cat("\n--- Overall GVI Statistics ---\n")
print(desc_stats_overall)
cat(paste("\nShapiro-Wilk test: W =", round(shapiro_test$statistic, 4), 
          ", p =", format.pval(shapiro_test$p.value), "\n"))

# Figure 4.1a: Histogram with density overlay (UPDATED syntax)
fig_4_1a <- ggplot(gvi_data, aes(x = mean_gvi)) +
  geom_histogram(aes(y = after_stat(density)), bins = 50, 
                 fill = "forestgreen", alpha = 0.7, color = "white") +
  geom_density(color = "darkgreen", linewidth = 1) +
  geom_vline(xintercept = desc_stats_overall$mean, 
             color = "red", linetype = "dashed", linewidth = 1) +
  geom_vline(xintercept = desc_stats_overall$median, 
             color = "blue", linetype = "dashed", linewidth = 1) +
  labs(
    title = "Distribution of Green View Index across Inner London LSOAs",
    subtitle = paste("Mean =", round(desc_stats_overall$mean, 2), "% (red)",
                     "| Median =", round(desc_stats_overall$median, 2), "% (blue)"),
    x = "Green View Index (%)",
    y = "Density",
    caption = paste("n =", nrow(gvi_data), "LSOAs")
  ) +
  theme(plot.title = element_text(face = "bold", size = 14))

# Save figure
ggsave(here("figures", "fig_4_1a_gvi_distribution.png"), 
       fig_4_1a, width = 10, height = 6, dpi = 300)

# -----------------------------------------------------------------------------
# 2.2 GVI by IMD Quintiles (Section 4.2.2)
# -----------------------------------------------------------------------------

# Calculate statistics by IMD quintile
desc_stats_quintile <- gvi_data %>%
  group_by(imd_quintile) %>%
  summarise(
    n = n(),
    mean = mean(mean_gvi, na.rm = TRUE),
    sd = sd(mean_gvi, na.rm = TRUE),
    median = median(mean_gvi, na.rm = TRUE),
    min = min(mean_gvi, na.rm = TRUE),
    max = max(mean_gvi, na.rm = TRUE),
    q25 = quantile(mean_gvi, 0.25, na.rm = TRUE),
    q75 = quantile(mean_gvi, 0.75, na.rm = TRUE),
    se = sd(mean_gvi, na.rm = TRUE) / sqrt(n()),
    ci_lower = mean - 1.96 * se,
    ci_upper = mean + 1.96 * se,
    .groups = 'drop'
  )

# Table 4.1: Descriptive Statistics by IMD Quintile
table_4_1 <- desc_stats_quintile %>%
  dplyr::mutate(across(where(is.numeric), ~round(., 2))) %>%
  dplyr::mutate(
    `Mean (SD)` = paste0(mean, " (", sd, ")"),
    `Median [IQR]` = paste0(median, " [", q25, "-", q75, "]"),
    `Range` = paste0(min, "-", max)
  ) %>%
  dplyr::select(
    `IMD Quintile` = imd_quintile,
    N = n,
    `Mean (SD)`,
    `Median [IQR]`,
    `Range`
  )

# Save table
write_csv(table_4_1, here("output", "table_4_1_descriptive_stats.csv"))

cat("\n--- Table 4.1: GVI Statistics by IMD Quintile ---\n")
print(kable(table_4_1, format = "simple"))

# ANOVA test for differences between quintiles
anova_result <- aov(mean_gvi ~ imd_quintile, data = gvi_data)
anova_summary <- summary(anova_result)

# Effect size (eta-squared)
eta_squared <- anova_summary[[1]]$`Sum Sq`[1] / 
  sum(anova_summary[[1]]$`Sum Sq`)

# Post-hoc test (Tukey HSD)
tukey_result <- TukeyHSD(anova_result)

cat("\n--- ANOVA Results ---\n")
print(anova_summary)
cat(paste("\nEffect size (η²) =", round(eta_squared, 3), "\n"))

# Figure 4.2: Violin plots by IMD quintile
fig_4_2 <- ggplot(gvi_data, aes(x = imd_quintile, y = mean_gvi, 
                                fill = imd_quintile)) +
  geom_violin(alpha = 0.7, scale = "width") +
  geom_boxplot(width = 0.2, fill = "white", alpha = 0.8, 
               outlier.size = 1, outlier.alpha = 0.5) +
  geom_point(data = desc_stats_quintile, 
             aes(x = imd_quintile, y = mean),
             color = "red", size = 3, shape = 18) +
  scale_fill_viridis_d(option = "viridis", begin = 0.2, end = 0.8) +
  labs(
    title = "Green View Index Distribution by IMD Quintile",
    subtitle = paste("ANOVA: F =", round(anova_summary[[1]]$`F value`[1], 2),
                     ", p <", ifelse(anova_summary[[1]]$`Pr(>F)`[1] < 0.001, 
                                     "0.001", 
                                     round(anova_summary[[1]]$`Pr(>F)`[1], 3)),
                     ", η² =", round(eta_squared, 3)),
    x = "IMD Quintile",
    y = "Green View Index (%)",
    caption = "Red diamonds indicate mean values"
  ) +
  theme(legend.position = "none",
        plot.title = element_text(face = "bold", size = 14))

ggsave(here("figures", "fig_4_2_gvi_by_quintile.png"), 
       fig_4_2, width = 10, height = 6, dpi = 300)

# Kruskal-Wallis test (non-parametric alternative)
kruskal_result <- kruskal.test(mean_gvi ~ imd_quintile, data = gvi_data)

cat("\n--- Kruskal-Wallis Test (non-parametric) ---\n")
cat(paste("Chi-squared =", round(kruskal_result$statistic, 2),
          ", df =", kruskal_result$parameter,
          ", p =", format.pval(kruskal_result$p.value), "\n"))

# -----------------------------------------------------------------------------
# 2.3 Borough-level Statistics (Additional for Section 4.2.2)
# -----------------------------------------------------------------------------

desc_stats_borough <- gvi_data %>%
  group_by(borough) %>%
  summarise(
    n = n(),
    mean_gvi = mean(mean_gvi, na.rm = TRUE),
    sd_gvi = sd(mean_gvi, na.rm = TRUE),
    mean_imd = mean(imd_score, na.rm = TRUE),
    .groups = 'drop'
  ) %>%
  arrange(desc(mean_gvi))

cat("\n--- Top 5 Boroughs by Mean GVI ---\n")
print(head(desc_stats_borough, 5))

# =============================================================================
# 3. CORRELATION ANALYSIS (Section 4.3)
# =============================================================================

cat("\n========== CORRELATION ANALYSIS (Section 4.3) ==========\n")

# -----------------------------------------------------------------------------
# 3.1 Bivariate Correlations (Section 4.3.1)
# -----------------------------------------------------------------------------

# Calculate multiple correlation coefficients
cor_pearson <- cor.test(gvi_data$mean_gvi, gvi_data$imd_score, 
                        method = "pearson")
cor_spearman <- cor.test(gvi_data$mean_gvi, gvi_data$imd_score, 
                         method = "spearman")
cor_kendall <- cor.test(gvi_data$mean_gvi, gvi_data$imd_score, 
                        method = "kendall")

# Create correlation summary table
cor_summary <- data.frame(
  Method = c("Pearson", "Spearman", "Kendall"),
  Coefficient = c(cor_pearson$estimate, cor_spearman$estimate, 
                  cor_kendall$estimate),
  CI_Lower = c(cor_pearson$conf.int[1], NA, NA),
  CI_Upper = c(cor_pearson$conf.int[2], NA, NA),
  t_statistic = c(cor_pearson$statistic, NA, NA),
  p_value = c(cor_pearson$p.value, cor_spearman$p.value, 
              cor_kendall$p.value),
  Interpretation = c(
    ifelse(abs(cor_pearson$estimate) < 0.3, "Small",
           ifelse(abs(cor_pearson$estimate) < 0.5, "Medium", "Large")),
    ifelse(abs(cor_spearman$estimate) < 0.3, "Small",
           ifelse(abs(cor_spearman$estimate) < 0.5, "Medium", "Large")),
    ifelse(abs(cor_kendall$estimate) < 0.3, "Small",
           ifelse(abs(cor_kendall$estimate) < 0.5, "Medium", "Large"))
  )
)

cat("\n--- Table 4.2a: Correlation Coefficients ---\n")
print(cor_summary)

# Save correlation table
write_csv(cor_summary, here("output", "table_4_2a_correlations.csv"))

# Figure 4.3a: Basic scatterplot with regression line
fig_4_3a <- ggplot(gvi_data, aes(x = imd_score, y = mean_gvi)) +
  geom_point(aes(color = imd_quintile), alpha = 0.6, size = 2) +
  geom_smooth(method = "lm", color = "red", se = TRUE, alpha = 0.2) +
  scale_color_viridis_d(option = "viridis", begin = 0.2, end = 0.8,
                        name = "IMD Quintile") +
  labs(
    title = "Relationship between Green View Index and Deprivation",
    subtitle = paste("Pearson r =", round(cor_pearson$estimate, 3),
                     ", p <", ifelse(cor_pearson$p.value < 0.001, "0.001",
                                     round(cor_pearson$p.value, 3))),
    x = "IMD Score (higher = more deprived)",
    y = "Green View Index (%)",
    caption = paste("n =", nrow(gvi_data), "LSOAs")
  ) +
  theme(plot.title = element_text(face = "bold", size = 14),
        legend.position = "right")

ggsave(here("figures", "fig_4_3a_scatter_basic.png"), 
       fig_4_3a, width = 10, height = 6, dpi = 300)

# -----------------------------------------------------------------------------
# 3.2 Non-linear Relationship Assessment (Section 4.3.2)
# -----------------------------------------------------------------------------

# Linear model (baseline)
lm_linear <- lm(mean_gvi ~ imd_score, data = gvi_data)

# Quadratic model
lm_quad <- lm(mean_gvi ~ poly(imd_score, 2), data = gvi_data)

# Cubic model
lm_cubic <- lm(mean_gvi ~ poly(imd_score, 3), data = gvi_data)

# GAM model for flexible non-linear relationship
gam_model <- gam(mean_gvi ~ s(imd_score), data = gvi_data)

# Model comparison
model_comparison <- data.frame(
  Model = c("Linear", "Quadratic", "Cubic", "GAM"),
  R2 = c(summary(lm_linear)$r.squared,
         summary(lm_quad)$r.squared,
         summary(lm_cubic)$r.squared,
         summary(gam_model)$r.sq),
  Adj_R2 = c(summary(lm_linear)$adj.r.squared,
             summary(lm_quad)$adj.r.squared,
             summary(lm_cubic)$adj.r.squared,
             summary(gam_model)$r.sq),
  AIC = c(AIC(lm_linear), AIC(lm_quad), AIC(lm_cubic), AIC(gam_model)),
  BIC = c(BIC(lm_linear), BIC(lm_quad), BIC(lm_cubic), BIC(gam_model))
)

cat("\n--- Non-linear Model Comparison ---\n")
print(model_comparison)

# ANOVA to test if polynomial terms improve fit
anova_poly <- anova(lm_linear, lm_quad, lm_cubic)

cat("\n--- Polynomial Model Comparison (ANOVA) ---\n")
print(anova_poly)

# Figure 4.3b: Non-linear relationships visualization (UPDATED linewidth)
# Create predictions for plotting
pred_data <- data.frame(
  imd_score = seq(min(gvi_data$imd_score), max(gvi_data$imd_score), 
                  length.out = 200)
)
pred_data$linear <- predict(lm_linear, pred_data)
pred_data$quadratic <- predict(lm_quad, pred_data)
pred_data$gam <- predict(gam_model, pred_data)

fig_4_3b <- ggplot(gvi_data, aes(x = imd_score, y = mean_gvi)) +
  geom_point(alpha = 0.3, color = "gray50") +
  geom_line(data = pred_data, aes(y = linear), 
            color = "red", linewidth = 1, linetype = "solid") +
  geom_line(data = pred_data, aes(y = quadratic), 
            color = "blue", linewidth = 1, linetype = "dashed") +
  geom_line(data = pred_data, aes(y = gam), 
            color = "green", linewidth = 1, linetype = "dotted") +
  labs(
    title = "Linear vs Non-linear Relationships",
    subtitle = "Red: Linear | Blue: Quadratic | Green: GAM",
    x = "IMD Score",
    y = "Green View Index (%)"
  ) +
  theme(plot.title = element_text(face = "bold", size = 14))

ggsave(here("figures", "fig_4_3b_nonlinear_comparison.png"), 
       fig_4_3b, width = 10, height = 6, dpi = 300)

# -----------------------------------------------------------------------------
# 3.3 Regression Models (Section 4.3.3)
# -----------------------------------------------------------------------------

# Model 1: Base model
m1 <- lm(mean_gvi ~ imd_score, data = gvi_data)

# Model 2: Control for data source
m2 <- lm(mean_gvi ~ imd_score + data_source, data = gvi_data)

# Model 3: Control for number of images
m3 <- lm(mean_gvi ~ imd_score + n_images, data = gvi_data)

# Model 4: Full model with borough fixed effects
m4 <- lm(mean_gvi ~ imd_score + data_source + n_images + borough, 
         data = gvi_data)

# Model 5: Interaction with data source
m5 <- lm(mean_gvi ~ imd_score * data_source, data = gvi_data)

# Create regression table
reg_table <- data.frame(
  Model = c("M1: Base", "M2: +Source", "M3: +Images", 
            "M4: Full", "M5: Interaction"),
  Beta_IMD = c(coef(m1)["imd_score"],
               coef(m2)["imd_score"],
               coef(m3)["imd_score"],
               coef(m4)["imd_score"],
               coef(m5)["imd_score"]),
  SE_IMD = c(summary(m1)$coefficients["imd_score", "Std. Error"],
             summary(m2)$coefficients["imd_score", "Std. Error"],
             summary(m3)$coefficients["imd_score", "Std. Error"],
             summary(m4)$coefficients["imd_score", "Std. Error"],
             summary(m5)$coefficients["imd_score", "Std. Error"]),
  R2 = c(summary(m1)$r.squared,
         summary(m2)$r.squared,
         summary(m3)$r.squared,
         summary(m4)$r.squared,
         summary(m5)$r.squared),
  Adj_R2 = c(summary(m1)$adj.r.squared,
             summary(m2)$adj.r.squared,
             summary(m3)$adj.r.squared,
             summary(m4)$adj.r.squared,
             summary(m5)$adj.r.squared),
  F_stat = c(summary(m1)$fstatistic[1],
             summary(m2)$fstatistic[1],
             summary(m3)$fstatistic[1],
             summary(m4)$fstatistic[1],
             summary(m5)$fstatistic[1]),
  AIC = c(AIC(m1), AIC(m2), AIC(m3), AIC(m4), AIC(m5))
)

cat("\n--- Table 4.3: Regression Model Comparison ---\n")
print(reg_table)

# Save regression table
write_csv(reg_table, here("output", "table_4_3_regression_models.csv"))

# Detailed output for best model
cat("\n--- Detailed Results for Model 4 (Full Model) ---\n")
print(summary(m4))

# Residual diagnostics for best model
par(mfrow = c(2, 2))
plot(m4)
dev.off()

# VIF for multicollinearity check (Model 4)
vif_values <- car::vif(m3)  # Use m3 as m4 has too many factor levels
cat("\n--- Variance Inflation Factors (Model 3) ---\n")
print(vif_values)

# -----------------------------------------------------------------------------
# 3.4 Partial Correlations (Additional Analysis)
# -----------------------------------------------------------------------------

# Partial correlation controlling for borough
partial_cor <- partial.r(gvi_data[, c("mean_gvi", "imd_score", "n_images")],
                         c(1, 2), 3)

cat("\n--- Partial Correlation (controlling for n_images) ---\n")
cat(paste("r =", round(partial_cor, 3), "\n"))

# -----------------------------------------------------------------------------
# 3.5 By-Group Correlations (Section 4.3.3 supplement)
# -----------------------------------------------------------------------------

# Correlation by data source
cor_by_source <- gvi_data %>%
  group_by(data_source) %>%
  summarise(
    n = n(),
    correlation = cor(mean_gvi, imd_score),
    p_value = cor.test(mean_gvi, imd_score)$p.value,
    .groups = 'drop'
  )

cat("\n--- Correlation by Data Source ---\n")
print(cor_by_source)

# Correlation by borough
cor_by_borough <- gvi_data %>%
  group_by(borough) %>%
  summarise(
    n = n(),
    correlation = cor(mean_gvi, imd_score),
    p_value = cor.test(mean_gvi, imd_score)$p.value,
    .groups = 'drop'
  ) %>%
  arrange(correlation)

cat("\n--- Correlation by Borough (sorted) ---\n")
print(cor_by_borough)

# Figure 4.3c: Borough-specific correlations (UPDATED linewidth)
fig_4_3c <- ggplot(cor_by_borough, aes(x = reorder(borough, correlation), 
                                       y = correlation)) +
  geom_col(aes(fill = correlation), show.legend = FALSE) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  geom_hline(yintercept = cor_pearson$estimate, 
             linetype = "dashed", color = "red", linewidth = 1) +
  scale_fill_gradient2(low = "red", mid = "white", high = "blue", 
                       midpoint = 0) +
  coord_flip() +
  labs(
    title = "GVI-IMD Correlation by Borough",
    subtitle = paste("Red line indicates overall correlation (r =", 
                     round(cor_pearson$estimate, 3), ")"),
    x = "Borough",
    y = "Correlation Coefficient"
  ) +
  theme(plot.title = element_text(face = "bold", size = 14))

ggsave(here("figures", "fig_4_3c_borough_correlations.png"), 
       fig_4_3c, width = 10, height = 8, dpi = 300)

# =============================================================================
# 4. QUINTILE COMPARISON (Section 4.2.2 supplement)
# =============================================================================

# Pairwise comparisons between quintiles
quintile_pairs <- combn(levels(gvi_data$imd_quintile), 2, simplify = FALSE)

pairwise_tests <- purrr::map_df(quintile_pairs, function(pair) {
  data1 <- gvi_data$mean_gvi[gvi_data$imd_quintile == pair[1]]
  data2 <- gvi_data$mean_gvi[gvi_data$imd_quintile == pair[2]]
  
  t_test <- t.test(data1, data2)
  
  data.frame(
    Comparison = paste(pair[1], "vs", pair[2]),
    Mean_Diff = mean(data1) - mean(data2),
    t_statistic = t_test$statistic,
    p_value = t_test$p.value,
    CI_lower = t_test$conf.int[1],
    CI_upper = t_test$conf.int[2]
  )
})

# Adjust p-values for multiple comparisons (Bonferroni)
pairwise_tests$p_adjusted <- p.adjust(pairwise_tests$p_value, 
                                      method = "bonferroni")

cat("\n--- Pairwise Comparisons (Bonferroni adjusted) ---\n")
print(pairwise_tests %>% 
        dplyr::mutate(across(where(is.numeric), ~round(., 4))) %>%
        dplyr::filter(p_adjusted < 0.05))

# =============================================================================
# 5. SUMMARY OUTPUT MAPPING
# =============================================================================

cat("\n========== OUTPUT MAPPING FOR DISSERTATION ==========\n")
cat("\nSECTION 4.2.1 - Overall GVI Distribution:\n- desc_stats_overall: Overall descriptive statistics\n- Figure 4.1a: Histogram with density overlay (fig_4_1a_gvi_distribution.png)\n- Shapiro-Wilk normality test results\n\nSECTION 4.2.2 - GVI by IMD Quintiles:\n- Table 4.1: Descriptive statistics by quintile (table_4_1_descriptive_stats.csv)\n- Figure 4.2: Violin plots by IMD quintile (fig_4_2_gvi_by_quintile.png)\n- ANOVA results with effect size\n- Tukey HSD post-hoc test\n- Kruskal-Wallis non-parametric test\n\nSECTION 4.3.1 - Bivariate Correlations:\n- Table 4.2a: Correlation coefficients (table_4_2a_correlations.csv)\n- Figure 4.3a: Basic scatterplot (fig_4_3a_scatter_basic.png)\n- Pearson, Spearman, and Kendall correlations\n\nSECTION 4.3.2 - Non-linear Relationships:\n- Model comparison table (Linear vs Polynomial vs GAM)\n- Figure 4.3b: Non-linear relationships (fig_4_3b_nonlinear_comparison.png)\n- ANOVA for polynomial terms\n\nSECTION 4.3.3 - Regression Models:\n- Table 4.3: Regression model comparison (table_4_3_regression_models.csv)\n- Detailed Model 4 output\n- VIF for multicollinearity\n- Residual diagnostic plots\n\nSUPPLEMENTARY ANALYSES:\n- Borough-level statistics\n- Figure 4.3c: Borough correlations (fig_4_3c_borough_correlations.png)\n- Correlation by data source\n- Pairwise quintile comparisons\n")

cat("\n========== ANALYSIS COMPLETE ==========\n")
cat("All figures saved to: figures/\n")
cat("All tables saved to: output/\n")

# Save workspace for future reference
save.image(here("output", "RQ1_analysis_workspace.RData"))
