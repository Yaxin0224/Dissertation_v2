#!/usr/bin/env Rscript
# =============================================================================
# City Look Dissertation - PTAL Data Cleaning
# Clean and prepare PTAL (Public Transport Accessibility Level) scores
# for Inner London LSOAs
# Author: [Your Name]
# Date: 2025
# =============================================================================

# Load required libraries
suppressPackageStartupMessages({
  library(tidyverse)
  library(here)
  library(readr)
})

cat("\n========== PTAL DATA CLEANING ==========\n")
cat("Processing Public Transport Accessibility Level scores for Inner London\n")

# =============================================================================
# 1. LOAD DATA
# =============================================================================

cat("\n--- Step 1: Loading Data ---\n")

# Load Inner London LSOA list (your main dataset)
inner_london <- read_csv(
  here("output", "full_london_gvi", "inner_london_all_lsoas_complete.csv"),
  show_col_types = FALSE
)
cat(paste("Inner London LSOAs loaded:", nrow(inner_london), "\n"))

# Get unique boroughs
inner_boroughs <- unique(inner_london$borough)
cat(paste("Inner London Boroughs:", paste(inner_boroughs, collapse = ", "), "\n\n"))

# Load PTAL data
ptal_raw <- read_csv(
  here("data", "raw", "rq3", "ptal_scores.csv"),
  show_col_types = FALSE
)
cat(paste("PTAL records loaded:", nrow(ptal_raw), "\n"))
cat(paste("Columns:", paste(names(ptal_raw), collapse = ", "), "\n\n"))

# =============================================================================
# 2. INITIAL DATA INSPECTION
# =============================================================================

cat("--- Step 2: Data Inspection ---\n")

# Check column names and rename for consistency
ptal_data <- ptal_raw %>%
  rename(
    lsoa_code = LSOA21CD,
    lsoa_name = LSOA21NM,
    mean_ai = mean_AI,      # AI = Accessibility Index
    median_ai = MEDIAN_AI,
    min_ai = MIN_AI,
    max_ai = MAX_AI,
    mean_ptal = MEAN_PTAL_  # PTAL category
  )

# Check PTAL categories
ptal_categories <- unique(ptal_data$mean_ptal)
cat("PTAL Categories found:\n")
print(sort(ptal_categories))

# Convert PTAL categories to numeric scores if needed
# PTAL scale: 0, 1a, 1b, 2, 3, 4, 5, 6a, 6b
ptal_to_numeric <- function(ptal_cat) {
  case_when(
    ptal_cat == "0" ~ 0,
    ptal_cat == "1a" ~ 1.0,
    ptal_cat == "1b" ~ 1.5,
    ptal_cat == "2" ~ 2,
    ptal_cat == "3" ~ 3,
    ptal_cat == "4" ~ 4,
    ptal_cat == "5" ~ 5,
    ptal_cat == "6a" ~ 6.0,
    ptal_cat == "6b" ~ 6.5,
    TRUE ~ NA_real_
  )
}

# Add numeric PTAL score
ptal_data <- ptal_data %>%
  mutate(
    ptal_score = ptal_to_numeric(mean_ptal),
    # Create simplified categories
    ptal_category = case_when(
      ptal_score <= 1.5 ~ "Very Poor",
      ptal_score <= 2 ~ "Poor",
      ptal_score <= 3 ~ "Moderate",
      ptal_score <= 4 ~ "Good",
      ptal_score <= 5 ~ "Very Good",
      ptal_score > 5 ~ "Excellent",
      TRUE ~ NA_character_
    )
  )

# Summary statistics before filtering
cat("\n--- PTAL Statistics (All London) ---\n")
ptal_summary_all <- ptal_data %>%
  summarise(
    n = n(),
    mean_ai = mean(mean_ai, na.rm = TRUE),
    median_ai = median(median_ai, na.rm = TRUE),
    mean_ptal_score = mean(ptal_score, na.rm = TRUE),
    median_ptal_score = median(ptal_score, na.rm = TRUE)
  )
print(ptal_summary_all)

# =============================================================================
# 3. FILTER FOR INNER LONDON
# =============================================================================

cat("\n--- Step 3: Filtering for Inner London ---\n")

# Method 1: Filter by LSOA codes
ptal_inner_london <- ptal_data %>%
  filter(lsoa_code %in% inner_london$lsoa_code)

cat(paste("LSOAs matched by code:", nrow(ptal_inner_london), "\n"))

# Method 2: Also check by borough name (as backup)
if (nrow(ptal_inner_london) < nrow(inner_london)) {
  # Check which boroughs are in the PTAL data
  ptal_boroughs <- unique(ptal_data$borough)
  cat("\nBoroughs in PTAL data:\n")
  print(sort(ptal_boroughs))
  
  # Try matching by borough
  ptal_by_borough <- ptal_data %>%
    filter(borough %in% inner_boroughs)
  
  cat(paste("\nLSOAs in Inner London boroughs:", nrow(ptal_by_borough), "\n"))
  
  # Combine both methods
  ptal_inner_london <- ptal_data %>%
    filter(lsoa_code %in% inner_london$lsoa_code | 
             borough %in% inner_boroughs) %>%
    filter(lsoa_code %in% inner_london$lsoa_code)  # Final filter by code
}

# =============================================================================
# 4. CHECK COVERAGE AND HANDLE MISSING
# =============================================================================

cat("\n--- Step 4: Coverage Analysis ---\n")

# Check coverage
coverage <- nrow(ptal_inner_london) / nrow(inner_london) * 100
cat(paste("Coverage:", nrow(ptal_inner_london), "/", nrow(inner_london), 
          "(", round(coverage, 1), "%)\n"))

# Identify missing LSOAs
missing_lsoas <- setdiff(inner_london$lsoa_code, ptal_inner_london$lsoa_code)
n_missing <- length(missing_lsoas)

if (n_missing > 0) {
  cat(paste("\nMissing LSOAs:", n_missing, "\n"))
  
  # Check which boroughs have missing data
  missing_by_borough <- inner_london %>%
    filter(lsoa_code %in% missing_lsoas) %>%
    count(borough) %>%
    arrange(desc(n))
  
  cat("\nMissing LSOAs by borough:\n")
  print(missing_by_borough)
  
  # Impute missing values using borough averages
  borough_avg_ptal <- ptal_inner_london %>%
    group_by(borough) %>%
    summarise(
      borough_mean_ai = mean(mean_ai, na.rm = TRUE),
      borough_median_ai = mean(median_ai, na.rm = TRUE),
      borough_ptal_score = mean(ptal_score, na.rm = TRUE),
      .groups = 'drop'
    )
  
  # If any borough has no data, use overall Inner London average
  overall_avg <- ptal_inner_london %>%
    summarise(
      overall_mean_ai = mean(mean_ai, na.rm = TRUE),
      overall_ptal_score = mean(ptal_score, na.rm = TRUE)
    )
  
  # Create records for missing LSOAs
  missing_data <- inner_london %>%
    filter(lsoa_code %in% missing_lsoas) %>%
    select(lsoa_code, lsoa_name, borough) %>%
    left_join(borough_avg_ptal, by = "borough") %>%
    mutate(
      mean_ai = coalesce(borough_mean_ai, overall_avg$overall_mean_ai),
      median_ai = coalesce(borough_median_ai, overall_avg$overall_mean_ai),
      ptal_score = coalesce(borough_ptal_score, overall_avg$overall_ptal_score),
      min_ai = NA_real_,
      max_ai = NA_real_,
      mean_ptal = NA_character_,
      ptal_category = case_when(
        ptal_score <= 1.5 ~ "Very Poor",
        ptal_score <= 2 ~ "Poor",
        ptal_score <= 3 ~ "Moderate",
        ptal_score <= 4 ~ "Good",
        ptal_score <= 5 ~ "Very Good",
        ptal_score > 5 ~ "Excellent",
        TRUE ~ NA_character_
      ),
      data_source = "imputed"
    ) %>%
    select(-starts_with("borough_"), -starts_with("overall_"))
  
  # Add data source flag to original data
  ptal_inner_london <- ptal_inner_london %>%
    mutate(data_source = "measured")
  
  # Combine
  ptal_complete <- bind_rows(ptal_inner_london, missing_data) %>%
    arrange(lsoa_code)
  
} else {
  cat("✓ All Inner London LSOAs have PTAL data!\n")
  ptal_complete <- ptal_inner_london %>%
    mutate(data_source = "measured")
}

# =============================================================================
# 5. CLEAN AND STANDARDIZE
# =============================================================================

cat("\n--- Step 5: Cleaning and Standardizing ---\n")

# Select and rename key variables
ptal_final <- ptal_complete %>%
  select(
    lsoa_code,
    lsoa_name,
    ptal_score,           # Main numeric score (0-6.5)
    ptal_category,        # Simplified category
    ptal_original = mean_ptal,  # Original category (1a, 1b, etc.)
    accessibility_index = mean_ai,  # Accessibility Index
    ai_median = median_ai,
    ai_min = min_ai,
    ai_max = max_ai,
    data_source
  ) %>%
  # Add standardized scores
  mutate(
    ptal_score_z = scale(ptal_score)[,1],
    ai_z = scale(accessibility_index)[,1]
  )

# =============================================================================
# 6. SUMMARY STATISTICS
# =============================================================================

cat("\n--- Step 6: Summary Statistics ---\n")

# Overall statistics
summary_stats <- ptal_final %>%
  summarise(
    n = n(),
    n_measured = sum(data_source == "measured"),
    n_imputed = sum(data_source == "imputed"),
    mean_ptal = mean(ptal_score, na.rm = TRUE),
    median_ptal = median(ptal_score, na.rm = TRUE),
    sd_ptal = sd(ptal_score, na.rm = TRUE),
    min_ptal = min(ptal_score, na.rm = TRUE),
    max_ptal = max(ptal_score, na.rm = TRUE),
    mean_ai = mean(accessibility_index, na.rm = TRUE),
    median_ai = median(accessibility_index, na.rm = TRUE)
  )

cat("\nOverall Statistics:\n")
print(summary_stats)

# Category distribution
cat("\nPTAL Category Distribution:\n")
category_dist <- ptal_final %>%
  count(ptal_category) %>%
  mutate(percentage = round(n / sum(n) * 100, 1)) %>%
  arrange(desc(n))
print(category_dist)

# Borough statistics
borough_stats <- inner_london %>%
  left_join(ptal_final, by = "lsoa_code") %>%
  group_by(borough) %>%
  summarise(
    n_lsoas = n(),
    mean_ptal = mean(ptal_score, na.rm = TRUE),
    median_ptal = median(ptal_score, na.rm = TRUE),
    mean_ai = mean(accessibility_index, na.rm = TRUE),
    .groups = 'drop'
  ) %>%
  arrange(desc(mean_ptal))

cat("\nBorough Rankings by PTAL:\n")
print(borough_stats)

# =============================================================================
# 7. SAVE CLEANED DATA
# =============================================================================

cat("\n--- Step 7: Saving Cleaned Data ---\n")

# Create output directory
output_dir <- here("data", "processed", "rq3")
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# Save main PTAL file
output_file <- file.path(output_dir, "ptal_scores_lsoa.csv")
write_csv(ptal_final, output_file)
cat(paste("Main file saved to:", output_file, "\n"))

# Save detailed version with borough and IMD info
detailed <- inner_london %>%
  select(lsoa_code, borough, imd_score, imd_quintile) %>%
  left_join(ptal_final, by = "lsoa_code")

detailed_file <- file.path(output_dir, "ptal_scores_lsoa_detailed.csv")
write_csv(detailed, detailed_file)
cat(paste("Detailed file saved to:", detailed_file, "\n"))

# =============================================================================
# 8. VISUALIZATION
# =============================================================================

library(ggplot2)

# Create visualization directory
plot_dir <- here("figures", "rq3")
if (!dir.exists(plot_dir)) dir.create(plot_dir, recursive = TRUE)

# 1. PTAL Score Distribution
p1 <- ggplot(ptal_final, aes(x = ptal_score)) +
  geom_histogram(aes(fill = ptal_category), bins = 20, alpha = 0.8) +
  geom_vline(aes(xintercept = mean(ptal_score)), 
             color = "red", linetype = "dashed", size = 1) +
  scale_fill_viridis_d() +
  labs(
    title = "PTAL Score Distribution - Inner London LSOAs",
    subtitle = paste("Mean:", round(mean(ptal_final$ptal_score), 2)),
    x = "PTAL Score (0-6.5)",
    y = "Number of LSOAs",
    fill = "Category"
  ) +
  theme_minimal() +
  theme(legend.position = "bottom")

ggsave(
  file.path(plot_dir, "ptal_distribution.png"),
  p1, width = 10, height = 6, dpi = 300
)

# 2. PTAL by Borough
p2 <- detailed %>%
  ggplot(aes(x = reorder(borough, ptal_score), y = ptal_score)) +
  geom_boxplot(aes(fill = borough), alpha = 0.7, show.legend = FALSE) +
  scale_fill_viridis_d() +
  coord_flip() +
  labs(
    title = "PTAL Scores by Borough",
    subtitle = "Public Transport Accessibility Level",
    x = "Borough",
    y = "PTAL Score"
  ) +
  theme_minimal()

ggsave(
  file.path(plot_dir, "ptal_by_borough.png"),
  p2, width = 10, height = 8, dpi = 300
)

# 3. PTAL vs IMD Scatter
p3 <- detailed %>%
  ggplot(aes(x = imd_score, y = ptal_score)) +
  geom_point(aes(color = ptal_category), alpha = 0.6) +
  geom_smooth(method = "lm", se = TRUE, color = "red") +
  scale_color_viridis_d() +
  labs(
    title = "PTAL Score vs IMD Score",
    subtitle = "Relationship between transport accessibility and deprivation",
    x = "IMD Score (higher = more deprived)",
    y = "PTAL Score (higher = better transport)",
    color = "PTAL Category"
  ) +
  theme_minimal() +
  theme(legend.position = "right")

ggsave(
  file.path(plot_dir, "ptal_vs_imd.png"),
  p3, width = 10, height = 6, dpi = 300
)

# Calculate correlation
cor_ptal_imd <- cor.test(detailed$imd_score, detailed$ptal_score)
cat(paste("\nCorrelation between PTAL and IMD: r =", 
          round(cor_ptal_imd$estimate, 3), 
          ", p =", format.pval(cor_ptal_imd$p.value), "\n"))

# =============================================================================
# 9. FINAL SUMMARY REPORT
# =============================================================================

cat("\n========== CLEANING COMPLETE ==========\n")

report <- list(
  total_lsoas = nrow(ptal_final),
  measured = sum(ptal_final$data_source == "measured"),
  imputed = sum(ptal_final$data_source == "imputed"),
  coverage_pct = round(sum(ptal_final$data_source == "measured") / nrow(ptal_final) * 100, 1),
  mean_ptal = round(mean(ptal_final$ptal_score), 2),
  median_ptal = round(median(ptal_final$ptal_score), 2),
  correlation_with_imd = round(cor_ptal_imd$estimate, 3)
)

cat("\nFinal Summary:\n")
cat(paste("✓ Total LSOAs processed:", report$total_lsoas, "\n"))
cat(paste("✓ Coverage:", report$coverage_pct, "%\n"))
cat(paste("✓ Mean PTAL score:", report$mean_ptal, "\n"))
cat(paste("✓ Correlation with IMD:", report$correlation_with_imd, "\n"))

if (report$imputed > 0) {
  cat(paste("\n⚠ Note:", report$imputed, "LSOAs used imputed values\n"))
}

cat("\nOutput files:\n")
cat(paste("- Main data:", output_file, "\n"))
cat(paste("- Detailed data:", detailed_file, "\n"))
cat(paste("- Visualizations: figures/rq3/\n"))

cat("\nNext steps:\n")
cat("1. Review the visualizations\n")
cat("2. Collect additional RQ3 variables (building density, etc.)\n")
cat("3. Run the main RQ3 analysis\n")

# Save summary report
report_file <- file.path(output_dir, "ptal_cleaning_report.txt")
sink(report_file)
cat("PTAL Data Cleaning Report\n")
cat("=========================\n")
cat(paste("Generated:", Sys.time(), "\n\n"))
print(report)
sink()

cat(paste("\nReport saved to:", report_file, "\n"))