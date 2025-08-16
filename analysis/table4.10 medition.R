#!/usr/bin/env Rscript
# =============================================================================
# Mediation Analysis for RQ3: Table 4.10 Generation
# Based on project knowledge: 06.1_section4.8.R
# =============================================================================

# Load required libraries
suppressPackageStartupMessages({
  library(tidyverse)
  library(here)
  library(mediation)  # For causal mediation analysis
  library(broom)      # For tidy model outputs
})

cat("\n========== MEDIATION ANALYSIS FOR TABLE 4.10 ==========\n")

# =============================================================================
# 1. LOAD AND PREPARE DATA
# =============================================================================

# Set paths
DATA_PATH <- here("output", "rq3", "rq3_integrated_data_with_zscores.csv")
OUTPUT_DIR <- here("output", "rq3")
FIG_DIR <- here("figures", "rq3")

# Create directories
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(FIG_DIR, recursive = TRUE, showWarnings = FALSE)

# Load integrated dataset with z-scores
if (!file.exists(DATA_PATH)) {
  stop("Integrated dataset not found. Please run the main RQ3 analysis first.")
}

rq3_data <- read_csv(DATA_PATH, show_col_types = FALSE)
cat("✓ Loaded RQ3 integrated dataset:", nrow(rq3_data), "observations\n")

# Check required variables
required_vars <- c("gvi_z", "imd_z", "population_density_z", "ptal_z", "building_density_z")
missing_vars <- setdiff(required_vars, names(rq3_data))

if (length(missing_vars) > 0) {
  cat("⚠ Missing z-score variables:", paste(missing_vars, collapse = ", "), "\n")
  cat("Creating missing z-scores...\n")
  
  # Create missing z-scores if needed
  if ("population_density_z" %in% missing_vars) {
    rq3_data$population_density_z <- scale(rq3_data$population_density)[, 1]
  }
  if ("ptal_z" %in% missing_vars) {
    rq3_data$ptal_z <- scale(rq3_data$ptal_score)[, 1]
  }
  if ("building_density_z" %in% missing_vars) {
    rq3_data$building_density_z <- scale(rq3_data$building_density)[, 1]
  }
  if ("gvi_z" %in% missing_vars) {
    rq3_data$gvi_z <- scale(rq3_data$mean_gvi)[, 1]
  }
  if ("imd_z" %in% missing_vars) {
    rq3_data$imd_z <- scale(rq3_data$imd_score)[, 1]
  }
}

# Remove rows with missing values for mediation analysis
rq3_data <- rq3_data %>% 
  filter(!is.na(gvi_z), !is.na(imd_z), 
         !is.na(population_density_z), !is.na(ptal_z), !is.na(building_density_z))

cat("✓ Final analysis dataset:", nrow(rq3_data), "complete observations\n")

# =============================================================================
# 2. MEDIATION ANALYSIS
# =============================================================================

cat("\n========== RUNNING MEDIATION ANALYSES ==========\n")

# Initialize results storage
mediation_results_list <- list()

# Set seed for reproducibility
set.seed(12345)

# ---- Mediation 1: Population Density as mediator ----
cat("\n--- Testing Population Density as Mediator ---\n")

tryCatch({
  # Step 1: IMD → Population Density (a path)
  med1_a <- lm(population_density_z ~ imd_z, data = rq3_data)
  a_coef <- coef(med1_a)["imd_z"]
  a_p <- summary(med1_a)$coefficients["imd_z", "Pr(>|t|)"]
  
  cat(sprintf("  Step 1 (IMD → Pop Density): β = %.3f, p = %.4f\n", a_coef, a_p))
  
  # Step 2: IMD + Population Density → GVI (b and c' paths)
  med1_b <- lm(gvi_z ~ imd_z + population_density_z, data = rq3_data)
  b_coef <- coef(med1_b)["population_density_z"]
  b_p <- summary(med1_b)$coefficients["population_density_z", "Pr(>|t|)"]
  c_prime <- coef(med1_b)["imd_z"]
  
  cat(sprintf("  Step 2 - Pop Density effect (b): β = %.3f, p = %.4f\n", b_coef, b_p))
  cat(sprintf("  Step 2 - Direct IMD effect (c'): β = %.3f\n", c_prime))
  
  # Step 3: Total effect (c path)
  med1_c <- lm(gvi_z ~ imd_z, data = rq3_data)
  c_total <- coef(med1_c)["imd_z"]
  cat(sprintf("  Total effect (c): β = %.3f\n", c_total))
  
  # Formal mediation test
  cat("  Running formal mediation test (1000 simulations)...\n")
  med1 <- mediate(med1_a, med1_b, treat = "imd_z", mediator = "population_density_z",
                  sims = 1000, boot = TRUE)
  
  # Store results
  mediation_results_list[["Population Density"]] <- list(
    ACME = med1$d0,
    ACME_CI = med1$d0.ci,
    ADE = med1$z0,
    Total = med1$tau.coef,
    Prop_Mediated = med1$n0,
    p_value = med1$d0.p
  )
  
  cat(sprintf("  ✓ ACME (Indirect effect): %.4f [%.4f, %.4f], p = %.4f\n",
              med1$d0, med1$d0.ci[1], med1$d0.ci[2], med1$d0.p))
  cat(sprintf("  ✓ Proportion mediated: %.1f%%\n", med1$n0 * 100))
  
}, error = function(e) {
  cat(paste("  ✗ Error in Population Density mediation:", e$message, "\n"))
})

# ---- Mediation 2: PTAL as mediator ----
cat("\n--- Testing PTAL as Mediator ---\n")

tryCatch({
  med2_a <- lm(ptal_z ~ imd_z, data = rq3_data)
  a_coef <- coef(med2_a)["imd_z"]
  a_p <- summary(med2_a)$coefficients["imd_z", "Pr(>|t|)"]
  
  cat(sprintf("  Step 1 (IMD → PTAL): β = %.3f, p = %.4f\n", a_coef, a_p))
  
  med2_b <- lm(gvi_z ~ imd_z + ptal_z, data = rq3_data)
  b_coef <- coef(med2_b)["ptal_z"]
  b_p <- summary(med2_b)$coefficients["ptal_z", "Pr(>|t|)"]
  
  cat(sprintf("  Step 2 - PTAL effect (b): β = %.3f, p = %.4f\n", b_coef, b_p))
  
  cat("  Running formal mediation test...\n")
  med2 <- mediate(med2_a, med2_b, treat = "imd_z", mediator = "ptal_z",
                  sims = 1000, boot = TRUE)
  
  mediation_results_list[["PTAL Score"]] <- list(
    ACME = med2$d0,
    ACME_CI = med2$d0.ci,
    ADE = med2$z0,
    Total = med2$tau.coef,
    Prop_Mediated = med2$n0,
    p_value = med2$d0.p
  )
  
  cat(sprintf("  ✓ ACME: %.4f [%.4f, %.4f], p = %.4f\n",
              med2$d0, med2$d0.ci[1], med2$d0.ci[2], med2$d0.p))
  cat(sprintf("  ✓ Proportion mediated: %.1f%%\n", med2$n0 * 100))
  
}, error = function(e) {
  cat(paste("  ✗ Error in PTAL mediation:", e$message, "\n"))
})

# ---- Mediation 3: Building Density as mediator ----
cat("\n--- Testing Building Density as Mediator ---\n")

tryCatch({
  med3_a <- lm(building_density_z ~ imd_z, data = rq3_data)
  a_coef <- coef(med3_a)["imd_z"]
  a_p <- summary(med3_a)$coefficients["imd_z", "Pr(>|t|)"]
  
  cat(sprintf("  Step 1 (IMD → Building): β = %.3f, p = %.4f\n", a_coef, a_p))
  
  med3_b <- lm(gvi_z ~ imd_z + building_density_z, data = rq3_data)
  b_coef <- coef(med3_b)["building_density_z"]
  b_p <- summary(med3_b)$coefficients["building_density_z", "Pr(>|t|)"]
  
  cat(sprintf("  Step 2 - Building effect (b): β = %.3f, p = %.4f\n", b_coef, b_p))
  
  cat("  Running formal mediation test...\n")
  med3 <- mediate(med3_a, med3_b, treat = "imd_z", mediator = "building_density_z",
                  sims = 1000, boot = TRUE)
  
  mediation_results_list[["Building Density"]] <- list(
    ACME = med3$d0,
    ACME_CI = med3$d0.ci,
    ADE = med3$z0,
    Total = med3$tau.coef,
    Prop_Mediated = med3$n0,
    p_value = med3$d0.p
  )
  
  cat(sprintf("  ✓ ACME: %.4f [%.4f, %.4f], p = %.4f\n",
              med3$d0, med3$d0.ci[1], med3$d0.ci[2], med3$d0.p))
  cat(sprintf("  ✓ Proportion mediated: %.1f%%\n", med3$n0 * 100))
  
}, error = function(e) {
  cat(paste("  ✗ Error in Building Density mediation:", e$message, "\n"))
})

# =============================================================================
# 3. CREATE TABLE 4.10
# =============================================================================

if (length(mediation_results_list) > 0) {
  cat("\n========== CREATING TABLE 4.10: MEDIATION RESULTS ==========\n")
  
  # Convert results to dataframe
  mediation_table <- map_df(names(mediation_results_list), function(mediator) {
    res <- mediation_results_list[[mediator]]
    data.frame(
      Mediator = mediator,
      ACME = res$ACME,
      ACME_Lower = res$ACME_CI[1],
      ACME_Upper = res$ACME_CI[2],
      ADE = res$ADE,
      Total_Effect = res$Total,
      Prop_Mediated = res$Prop_Mediated,
      p_value = res$p_value
    )
  }) %>%
    mutate(
      # Format confidence intervals
      CI_95 = paste0("[", sprintf("%.3f", ACME_Lower), ", ", sprintf("%.3f", ACME_Upper), "]"),
      # Calculate percentage mediated
      Prop_Mediated_Pct = paste0(sprintf("%.1f", Prop_Mediated * 100), "%"),
      # Add significance stars
      Significance = case_when(
        p_value < 0.001 ~ "***",
        p_value < 0.01 ~ "**",
        p_value < 0.05 ~ "*",
        p_value < 0.1 ~ "†",
        TRUE ~ ""
      )
    ) %>%
    # Round numeric columns
    mutate(across(c(ACME, ADE, Total_Effect), ~round(., 4))) %>%
    mutate(p_value = round(p_value, 4))
  
  # Display table
  cat("\n--- Table 4.10: Mediation Analysis Results ---\n")
  print(mediation_table)
  
  # Save full table
  write_csv(mediation_table, file.path(OUTPUT_DIR, "table_4_10_mediation_results.csv"))
  cat(sprintf("\n✓ Full table saved to: %s\n", file.path(OUTPUT_DIR, "table_4_10_mediation_results.csv")))
  
  # Create publication-ready version
  pub_table <- mediation_table %>%
    select(Mediator, ACME, CI_95, ADE, Total_Effect, Prop_Mediated_Pct, p_value, Significance) %>%
    rename(
      "Pathway" = "Mediator",
      "ACME" = "ACME",
      "95% CI" = "CI_95", 
      "ADE" = "ADE",
      "Total Effect" = "Total_Effect",
      "Proportion Mediated" = "Prop_Mediated_Pct",
      "p-value" = "p_value",
      "Sig." = "Significance"
    )
  
  write_csv(pub_table, file.path(OUTPUT_DIR, "table_4_10_publication_ready.csv"))
  cat(sprintf("✓ Publication table saved to: %s\n", file.path(OUTPUT_DIR, "table_4_10_publication_ready.csv")))
  
  # =============================================================================
  # 4. INTERPRETATION AND SUMMARY
  # =============================================================================
  
  cat("\n========== MEDIATION ANALYSIS INTERPRETATION ==========\n")
  
  # Identify significant mediators
  sig_mediators <- mediation_table %>% filter(p_value < 0.05)
  
  if (nrow(sig_mediators) > 0) {
    cat("✓ Significant mediators (p < 0.05):\n")
    for (i in 1:nrow(sig_mediators)) {
      cat(sprintf("  • %s: %.1f%% of IMD→GVI effect mediated (ACME = %.4f, p = %.4f)\n", 
                  sig_mediators$Mediator[i], 
                  sig_mediators$Prop_Mediated[i] * 100,
                  sig_mediators$ACME[i],
                  sig_mediators$p_value[i]))
    }
  } else {
    cat("⚠ No significant mediators found at p < 0.05 level\n")
  }
  
  # Check for suppression effects
  suppression <- mediation_table %>% 
    filter((ACME > 0 & ADE < 0) | (ACME < 0 & ADE > 0))
  
  if (nrow(suppression) > 0) {
    cat("\n⚠ Suppression effects detected for:\n")
    for (i in 1:nrow(suppression)) {
      cat(sprintf("  • %s: ACME and ADE have opposite signs\n", suppression$Mediator[i]))
    }
  }
  
  # Calculate total mediation
  total_mediation <- sum(abs(mediation_table$Prop_Mediated[mediation_table$p_value < 0.05]), na.rm = TRUE)
  cat(sprintf("\n✓ Total proportion mediated by significant pathways: %.1f%%\n", total_mediation * 100))
  cat(sprintf("✓ Direct effects account for: %.1f%% of total relationship\n", (1 - total_mediation) * 100))
  
} else {
  cat("\n⚠ No mediation analyses completed successfully\n")
}

cat("\n========== MEDIATION ANALYSIS COMPLETE ==========\n")
cat("Files created:\n")
cat("  - table_4_10_mediation_results.csv (full results)\n")
cat("  - table_4_10_publication_ready.csv (formatted for dissertation)\n")
cat("\nNext steps: Include Table 4.10 in Section 4.5 of Results chapter\n")