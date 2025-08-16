#!/usr/bin/env Rscript
# =============================================================================
# Figure 4.8: Results Synthesis Dashboard (4-panel)
# Comprehensive visualization of key findings from all RQs
# =============================================================================

# Load required libraries
library(tidyverse)
library(patchwork)
library(scales)
library(here)

# Set paths
DATA_PATH <- here("output", "full_london_gvi", "inner_london_all_lsoas_complete.csv")
OUTPUT_DIR <- here("figures")
FIG_DIR <- here("figures")

# Load data
gvi_data <- read_csv(DATA_PATH, show_col_types = FALSE)

cat("=== Creating Figure 4.8: Results Synthesis Dashboard ===\n")

# =============================================================================
# PANEL A: Main Correlation with Quintile Highlights
# =============================================================================

# Create quintile colors
quintile_colors <- c("#2166ac", "#4393c3", "#92c5de", "#f4a582", "#d6604d")

panel_a <- ggplot(gvi_data, aes(x = imd_score, y = mean_gvi, color = imd_quintile)) +
  geom_point(alpha = 0.6, size = 0.8) +
  geom_smooth(method = "lm", se = TRUE, color = "black", linewidth = 1.2) +
  scale_color_manual(values = quintile_colors, name = "IMD Quintile") +
  labs(
    title = "A. Primary Relationship",
    subtitle = "r = -0.084, p < 0.001",
    x = "Index of Multiple Deprivation",
    y = "Green View Index (%)"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 11),
    plot.subtitle = element_text(size = 9),
    legend.position = "bottom",
    legend.title = element_text(size = 8),
    legend.text = element_text(size = 7),
    axis.text = element_text(size = 8),
    axis.title = element_text(size = 9)
  ) +
  guides(color = guide_legend(override.aes = list(size = 2)))

# =============================================================================
# PANEL B: Borough Heterogeneity
# =============================================================================

# Calculate borough correlations
borough_cors <- gvi_data %>%
  group_by(borough) %>%
  summarise(
    correlation = cor(mean_gvi, imd_score, use = "complete.obs"),
    mean_gvi = mean(mean_gvi, na.rm = TRUE),
    .groups = 'drop'
  ) %>%
  arrange(correlation)

panel_b <- ggplot(borough_cors, aes(x = reorder(borough, correlation), y = correlation)) +
  geom_col(aes(fill = correlation), show.legend = FALSE) +
  geom_hline(yintercept = -0.084, linetype = "dashed", color = "red", linewidth = 1) +
  scale_fill_gradient2(low = "#d73027", mid = "white", high = "#1a9850", midpoint = 0) +
  coord_flip() +
  labs(
    title = "B. Borough Heterogeneity",
    subtitle = "Red line = Overall correlation",
    x = "Borough",
    y = "GVI-IMD Correlation"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 11),
    plot.subtitle = element_text(size = 9),
    axis.text.y = element_text(size = 7),
    axis.text.x = element_text(size = 8),
    axis.title = element_text(size = 9)
  )

# =============================================================================
# PANEL C: Model Performance Comparison
# =============================================================================

# Create model comparison data
model_data <- data.frame(
  Model = c("OLS", "Spatial Lag", "GWR", "Multilevel"),
  R2 = c(0.007, 0.426, 0.317, 0.007),
  Type = c("Conventional", "Spatial", "Spatial", "Hierarchical")
)

panel_c <- ggplot(model_data, aes(x = reorder(Model, R2), y = R2, fill = Type)) +
  geom_col(alpha = 0.8) +
  scale_fill_manual(values = c("Conventional" = "#969696", 
                               "Spatial" = "#2166ac", 
                               "Hierarchical" = "#762a83")) +
  scale_y_continuous(labels = percent_format(), limits = c(0, 0.45)) +
  coord_flip() +
  labs(
    title = "C. Model Performance",
    subtitle = "Explained variance (R²)",
    x = "Model Type",
    y = "R² (% variance explained)",
    fill = "Approach"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 11),
    plot.subtitle = element_text(size = 9),
    legend.position = "bottom",
    legend.title = element_text(size = 8),
    legend.text = element_text(size = 7),
    axis.text = element_text(size = 8),
    axis.title = element_text(size = 9)
  )

# =============================================================================
# PANEL D: Mediation Effects
# =============================================================================

# Mediation results data
mediation_data <- data.frame(
  Mediator = c("Population\nDensity", "PTAL\nScore", "Building\nDensity"),
  Prop_Mediated = c(76.7, 13.7, -68.0),
  Significant = c(TRUE, FALSE, TRUE),
  Direction = c("Negative", "Negative", "Suppression")
)

panel_d <- ggplot(mediation_data, aes(x = reorder(Mediator, abs(Prop_Mediated)), 
                                      y = Prop_Mediated, 
                                      fill = Direction)) +
  geom_col(alpha = 0.8) +
  geom_hline(yintercept = 0, linetype = "solid", color = "black", linewidth = 0.5) +
  scale_fill_manual(values = c("Negative" = "#d73027", 
                               "Suppression" = "#762a83")) +
  labs(
    title = "D. Mediation Effects",
    subtitle = "Proportion of IMD→GVI mediated",
    x = "Urban Form Mediator",
    y = "Proportion Mediated (%)",
    fill = "Effect Type"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 11),
    plot.subtitle = element_text(size = 9),
    legend.position = "bottom",
    legend.title = element_text(size = 8),
    legend.text = element_text(size = 7),
    axis.text = element_text(size = 8),
    axis.title = element_text(size = 9)
  )

# =============================================================================
# COMBINE PANELS
# =============================================================================

# Create 2x2 dashboard
dashboard <- (panel_a + panel_b) / (panel_c + panel_d) +
  plot_annotation(
    title = "Results Synthesis: Environmental Justice in Inner London",
    subtitle = "Multi-scale analysis of Green View Index and socioeconomic deprivation patterns",
    caption = "Source: Mapillary street-view imagery (2020-2024) | n = 1,731 LSOAs",
    theme = theme(
      plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
      plot.subtitle = element_text(size = 12, hjust = 0.5),
      plot.caption = element_text(size = 9, hjust = 0.5)
    )
  )

# Save dashboard
output_file <- file.path(FIG_DIR, "fig_4_8_results_synthesis_dashboard.png")
ggsave(
  filename = output_file,
  plot = dashboard,
  width = 14,
  height = 10,
  dpi = 300,
  bg = "white"
)

cat("✓ Figure 4.8 saved to:", output_file, "\n")

# =============================================================================
# CREATE ALTERNATIVE COMPACT VERSION
# =============================================================================

# More compact version for space-constrained documents
dashboard_compact <- (panel_a + panel_b) / (panel_c + panel_d) +
  plot_annotation(
    title = "Environmental Justice Results Synthesis",
    caption = "Source: n = 1,731 Inner London LSOAs",
    theme = theme(
      plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
      plot.caption = element_text(size = 8, hjust = 0.5)
    )
  )

compact_file <- file.path(FIG_DIR, "fig_4_8_results_synthesis_compact.png")
ggsave(
  filename = compact_file,
  plot = dashboard_compact,
  width = 12,
  height = 8,
  dpi = 300,
  bg = "white"
)

cat("✓ Compact version saved to:", compact_file, "\n")

# =============================================================================
# SUMMARY STATISTICS FOR TEXT
# =============================================================================

cat("\n=== KEY FINDINGS SUMMARY ===\n")
cat("Primary Correlation: r = -0.084, p < 0.001\n")
cat("Borough Correlation Range:", round(min(borough_cors$correlation), 3), "to", 
    round(max(borough_cors$correlation), 3), "\n")
cat("Best Spatial Model: SAR (R² = 0.426)\n")
cat("Strongest Mediator: Population Density (76.7%)\n")
cat("Significant Suppression: Building Density (-68.0%)\n")

cat("\n=== Figure 4.8 Creation Complete ===\n")
cat("Files created:\n")
cat("  - fig_4_8_results_synthesis_dashboard.png (full version)\n")