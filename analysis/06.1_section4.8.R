#!/usr/bin/env Rscript
# =============================================================================
# RQ3 MEDIATION ANALYSIS FIX
# Standalone script to run mediation analysis
# =============================================================================

library(tidyverse)
library(here)
library(mediation)

cat("\n========== FIXING MEDIATION ANALYSIS ==========\n")

# Load the integrated data
OUTPUT_DIR <- here("output", "rq3")
rq3_data <- read_csv(file.path(OUTPUT_DIR, "rq3_integrated_data.csv"), show_col_types = FALSE)

cat(paste("Data loaded:", nrow(rq3_data), "LSOAs\n"))

# Check if standardized variables exist
cat("\n--- Checking for standardized variables ---\n")
if(!"population_density_z" %in% names(rq3_data)) {
  cat("Standardized variables not found. Creating them now...\n")
  
  # Create standardized variables (z-scores)
  # First check which variables exist
  cat("\nAvailable variables:\n")
  urban_vars <- c("population_density", "ptal_score", "road_density_km_per_sqkm", "building_density")
  for(var in urban_vars) {
    if(var %in% names(rq3_data)) {
      cat(paste("  ✓", var, "\n"))
    } else {
      cat(paste("  ✗", var, "NOT FOUND\n"))
    }
  }
  
  # Create z-scores only for existing variables
  rq3_data <- rq3_data %>%
    mutate(
      gvi_z = scale(mean_gvi)[,1],
      imd_z = scale(imd_score)[,1]
    )
  
  # Add z-scores for urban form variables if they exist
  if("population_density" %in% names(rq3_data)) {
    rq3_data$population_density_z <- scale(rq3_data$population_density)[,1]
  }
  
  if("ptal_score" %in% names(rq3_data)) {
    rq3_data$ptal_z <- scale(rq3_data$ptal_score)[,1]
  }
  
  if("road_density_km_per_sqkm" %in% names(rq3_data)) {
    rq3_data$road_density_z <- scale(rq3_data$road_density_km_per_sqkm)[,1]
  }
  
  if("building_density" %in% names(rq3_data)) {
    rq3_data$building_density_z <- scale(rq3_data$building_density)[,1]
  }
  
  cat("✓ Standardized variables created\n")
} else {
  cat("✓ Standardized variables already exist\n")
}

# Verify all needed variables
cat("\n--- Verifying required variables ---\n")
required_vars <- c("gvi_z", "imd_z", "population_density_z", "ptal_z", "building_density_z")
for(var in required_vars) {
  if(var %in% names(rq3_data)) {
    cat(paste("✓", var, "found\n"))
  } else {
    cat(paste("⚠", var, "missing - will skip this analysis\n"))
  }
}

# =============================================================================
# MEDIATION ANALYSIS
# =============================================================================

cat("\n========== SECTION 4.8: MEDIATION ANALYSIS ==========\n")

# Initialize results list
mediation_results_list <- list()

# ---- Mediation 1: Population density as mediator ----
if("population_density_z" %in% names(rq3_data)) {
  cat("\n--- Testing Population Density as Mediator ---\n")
  
  tryCatch({
    # Step 1: IMD → Population Density
    med1_a <- lm(population_density_z ~ imd_z, data = rq3_data)
    cat("  Step 1 (IMD → Pop Density): ")
    cat(sprintf("β = %.3f, p = %.4f\n", 
                coef(med1_a)["imd_z"], 
                summary(med1_a)$coefficients["imd_z", "Pr(>|t|)"]))
    
    # Step 2: IMD + Pop Density → GVI
    med1_b <- lm(gvi_z ~ imd_z + population_density_z, data = rq3_data)
    cat("  Step 2 (IMD + Pop Density → GVI):\n")
    cat(sprintf("    IMD effect: β = %.3f, p = %.4f\n", 
                coef(med1_b)["imd_z"],
                summary(med1_b)$coefficients["imd_z", "Pr(>|t|)"]))
    cat(sprintf("    Pop Density effect: β = %.3f, p = %.4f\n", 
                coef(med1_b)["population_density_z"],
                summary(med1_b)$coefficients["population_density_z", "Pr(>|t|)"]))
    
    # Step 3: Total effect
    med1_c <- lm(gvi_z ~ imd_z, data = rq3_data)
    cat(sprintf("  Total effect: β = %.3f, p = %.4f\n", 
                coef(med1_c)["imd_z"],
                summary(med1_c)$coefficients["imd_z", "Pr(>|t|)"]))
    
    # Mediation test
    cat("  Running mediation test (1000 simulations)...\n")
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
    
    cat(sprintf("  ACME (Indirect effect): %.4f [%.4f, %.4f], p = %.4f\n",
                med1$d0, med1$d0.ci[1], med1$d0.ci[2], med1$d0.p))
    cat(sprintf("  Proportion mediated: %.1f%%\n", med1$n0 * 100))
    
  }, error = function(e) {
    cat(paste("  Error:", e$message, "\n"))
  })
}

# ---- Mediation 2: PTAL as mediator ----
if("ptal_z" %in% names(rq3_data)) {
  cat("\n--- Testing PTAL as Mediator ---\n")
  
  tryCatch({
    med2_a <- lm(ptal_z ~ imd_z, data = rq3_data)
    cat(sprintf("  Step 1 (IMD → PTAL): β = %.3f, p = %.4f\n", 
                coef(med2_a)["imd_z"], 
                summary(med2_a)$coefficients["imd_z", "Pr(>|t|)"]))
    
    med2_b <- lm(gvi_z ~ imd_z + ptal_z, data = rq3_data)
    cat(sprintf("  Step 2 - PTAL effect: β = %.3f, p = %.4f\n", 
                coef(med2_b)["ptal_z"],
                summary(med2_b)$coefficients["ptal_z", "Pr(>|t|)"]))
    
    cat("  Running mediation test...\n")
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
    
    cat(sprintf("  ACME: %.4f [%.4f, %.4f], p = %.4f\n",
                med2$d0, med2$d0.ci[1], med2$d0.ci[2], med2$d0.p))
    cat(sprintf("  Proportion mediated: %.1f%%\n", med2$n0 * 100))
    
  }, error = function(e) {
    cat(paste("  Error:", e$message, "\n"))
  })
}

# ---- Mediation 3: Building density as mediator ----
if("building_density_z" %in% names(rq3_data)) {
  cat("\n--- Testing Building Density as Mediator ---\n")
  
  tryCatch({
    med3_a <- lm(building_density_z ~ imd_z, data = rq3_data)
    cat(sprintf("  Step 1 (IMD → Building): β = %.3f, p = %.4f\n", 
                coef(med3_a)["imd_z"], 
                summary(med3_a)$coefficients["imd_z", "Pr(>|t|)"]))
    
    med3_b <- lm(gvi_z ~ imd_z + building_density_z, data = rq3_data)
    cat(sprintf("  Step 2 - Building effect: β = %.3f, p = %.4f\n", 
                coef(med3_b)["building_density_z"],
                summary(med3_b)$coefficients["building_density_z", "Pr(>|t|)"]))
    
    cat("  Running mediation test...\n")
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
    
    cat(sprintf("  ACME: %.4f [%.4f, %.4f], p = %.4f\n",
                med3$d0, med3$d0.ci[1], med3$d0.ci[2], med3$d0.p))
    cat(sprintf("  Proportion mediated: %.1f%%\n", med3$n0 * 100))
    
  }, error = function(e) {
    cat(paste("  Error:", e$message, "\n"))
  })
}

# ---- Mediation 4: Road density as mediator (if available) ----
if("road_density_z" %in% names(rq3_data)) {
  cat("\n--- Testing Road Density as Mediator ---\n")
  
  tryCatch({
    med4_a <- lm(road_density_z ~ imd_z, data = rq3_data)
    cat(sprintf("  Step 1 (IMD → Road): β = %.3f, p = %.4f\n", 
                coef(med4_a)["imd_z"], 
                summary(med4_a)$coefficients["imd_z", "Pr(>|t|)"]))
    
    med4_b <- lm(gvi_z ~ imd_z + road_density_z, data = rq3_data)
    cat(sprintf("  Step 2 - Road effect: β = %.3f, p = %.4f\n", 
                coef(med4_b)["road_density_z"],
                summary(med4_b)$coefficients["road_density_z", "Pr(>|t|)"]))
    
    cat("  Running mediation test...\n")
    med4 <- mediate(med4_a, med4_b, treat = "imd_z", mediator = "road_density_z",
                    sims = 1000, boot = TRUE)
    
    mediation_results_list[["Road Density"]] <- list(
      ACME = med4$d0,
      ACME_CI = med4$d0.ci,
      ADE = med4$z0,
      Total = med4$tau.coef,
      Prop_Mediated = med4$n0,
      p_value = med4$d0.p
    )
    
    cat(sprintf("  ACME: %.4f [%.4f, %.4f], p = %.4f\n",
                med4$d0, med4$d0.ci[1], med4$d0.ci[2], med4$d0.p))
    cat(sprintf("  Proportion mediated: %.1f%%\n", med4$n0 * 100))
    
  }, error = function(e) {
    cat(paste("  Error:", e$message, "\n"))
  })
}

# =============================================================================
# CREATE SUMMARY TABLE
# =============================================================================

if(length(mediation_results_list) > 0) {
  cat("\n========== TABLE 4.10: MEDIATION RESULTS SUMMARY ==========\n")
  
  # Convert list to dataframe
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
      Significance = case_when(
        p_value < 0.001 ~ "***",
        p_value < 0.01 ~ "**",
        p_value < 0.05 ~ "*",
        p_value < 0.1 ~ "†",
        TRUE ~ ""
      ),
      Prop_Mediated_Pct = paste0(round(Prop_Mediated * 100, 1), "%")
    ) %>%
    mutate(across(c(ACME, ACME_Lower, ACME_Upper, ADE, Total_Effect), ~round(., 4))) %>%
    mutate(p_value = round(p_value, 4))
  
  print(mediation_table)
  
  # Save table
  write_csv(mediation_table, file.path(OUTPUT_DIR, "table_4_10_mediation_results_fixed.csv"))
  cat("\n✓ Table saved to: output/rq3/table_4_10_mediation_results_fixed.csv\n")
  
  # Interpretation
  cat("\n--- INTERPRETATION ---\n")
  sig_mediators <- mediation_table %>% filter(p_value < 0.05)
  
  if(nrow(sig_mediators) > 0) {
    cat("Significant mediators (p < 0.05):\n")
    for(i in 1:nrow(sig_mediators)) {
      cat(sprintf("  • %s: %.1f%% of IMD effect mediated\n", 
                  sig_mediators$Mediator[i], 
                  sig_mediators$Prop_Mediated[i] * 100))
    }
  } else {
    cat("No significant mediators found at p < 0.05 level\n")
  }
  
  # Check for suppression effects (opposite signs)
  suppression <- mediation_table %>% 
    filter((ACME > 0 & ADE < 0) | (ACME < 0 & ADE > 0))
  
  if(nrow(suppression) > 0) {
    cat("\n⚠ Suppression effects detected for:\n")
    print(suppression$Mediator)
  }
  
} else {
  cat("\n⚠ No mediation analyses completed successfully\n")
}

# =============================================================================
# VISUALIZATION OF MEDIATION PATHS
# =============================================================================

if(length(mediation_results_list) > 0) {
  cat("\n--- Creating mediation path diagram ---\n")
  
  library(ggplot2)
  
  # Create a simple bar plot of mediation effects
  p_mediation <- mediation_table %>%
    mutate(Mediator = factor(Mediator, levels = Mediator[order(abs(ACME), decreasing = TRUE)])) %>%
    ggplot(aes(x = Mediator, y = ACME)) +
    geom_col(aes(fill = p_value < 0.05), show.legend = FALSE) +
    geom_errorbar(aes(ymin = ACME_Lower, ymax = ACME_Upper), width = 0.2) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
    scale_fill_manual(values = c("TRUE" = "steelblue", "FALSE" = "gray70")) +
    coord_flip() +
    labs(
      title = "Mediation Effects: Urban Form Variables on IMD→GVI Relationship",
      subtitle = "Average Causal Mediation Effects (ACME) with 95% CI",
      x = "Mediator Variable",
      y = "Indirect Effect (ACME)",
      caption = "Blue bars indicate significant mediation (p < 0.05)"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(face = "bold", size = 14),
      plot.subtitle = element_text(size = 10)
    )
  
  ggsave(file.path(here("figures", "rq3"), "fig_4_8_mediation_effects.png"),
         p_mediation, width = 10, height = 6, dpi = 300)
  cat("✓ Figure saved to: figures/rq3/fig_4_8_mediation_effects.png\n")
}

cat("\n========== MEDIATION ANALYSIS COMPLETE ==========\n")

# Save updated dataset with z-scores
write_csv(rq3_data, file.path(OUTPUT_DIR, "rq3_integrated_data_with_zscores.csv"))
cat("✓ Updated dataset saved with z-scores\n")