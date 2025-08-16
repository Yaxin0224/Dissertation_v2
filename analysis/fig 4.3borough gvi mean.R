#!/usr/bin/env Rscript
# =============================================================================
# Figure 4.2c: Borough-level GVI Means Choropleth Map
# =============================================================================

# Load required libraries
library(tidyverse)
library(sf)
library(here)
library(scales)
library(RColorBrewer)

# Set paths (based on project structure from knowledge)
DATA_PATH <- here("output", "full_london_gvi", "inner_london_all_lsoas_complete.csv")
SHAPEFILE_PATH <- here(
  "data", "raw", "statistical-gis-boundaries-london", "ESRI",
  "LSOA_2011_London_gen_MHW.shp"
)
OUTPUT_DIR <- here("figures")
FIG_DIR <- here("figures")

# Create output directory
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

cat("=== Creating Figure 4.2c: Borough-level GVI Means Map ===\n")

# Load data
cat("Loading data...\n")
gvi_data <- read_csv(DATA_PATH, show_col_types = FALSE)
lsoa_boundaries <- st_read(SHAPEFILE_PATH, quiet = TRUE)

# Calculate borough-level statistics
borough_stats <- gvi_data %>%
  group_by(borough) %>%
  summarise(
    n_lsoas = n(),
    mean_gvi = mean(mean_gvi, na.rm = TRUE),
    median_gvi = median(mean_gvi, na.rm = TRUE),
    sd_gvi = sd(mean_gvi, na.rm = TRUE),
    min_gvi = min(mean_gvi, na.rm = TRUE),
    max_gvi = max(mean_gvi, na.rm = TRUE),
    .groups = 'drop'
  ) %>%
  arrange(desc(mean_gvi))

# Print borough statistics
cat("\nBorough-level GVI Statistics:\n")
print(borough_stats)

# Create borough boundaries by dissolving LSOA boundaries
cat("Creating borough boundaries...\n")

# First, ensure we have the right columns and valid geometries
lsoa_boundaries <- lsoa_boundaries %>%
  st_make_valid() %>%
  rename_with(tolower) %>%
  filter(lsoa11cd %in% gvi_data$lsoa_code)

# Add borough information to boundaries
lsoa_with_borough <- lsoa_boundaries %>%
  left_join(gvi_data %>% select(lsoa_code, borough), 
            by = c("lsoa11cd" = "lsoa_code"))

# Dissolve by borough to create borough boundaries
borough_boundaries <- lsoa_with_borough %>%
  group_by(borough) %>%
  summarise(geometry = st_union(geometry), .groups = 'drop') %>%
  st_make_valid()

# Join with borough statistics
borough_map_data <- borough_boundaries %>%
  left_join(borough_stats, by = "borough") %>%
  filter(!is.na(mean_gvi))

# Transform to appropriate projection for mapping
borough_map_data <- borough_map_data %>%
  st_transform(crs = 4326)  # WGS84 for better display

# Create color palette
# Using a green-yellow-red palette to represent GVI levels
n_colors <- 7
colors <- RColorBrewer::brewer.pal(n_colors, "RdYlGn")

# Create the map
cat("Creating choropleth map...\n")

# Define breaks for classification
breaks <- seq(
  from = floor(min(borough_map_data$mean_gvi)), 
  to = ceiling(max(borough_map_data$mean_gvi)), 
  length.out = n_colors + 1
)

# Create the plot
p <- ggplot(borough_map_data) +
  geom_sf(aes(fill = mean_gvi), color = "white", size = 0.5) +
  scale_fill_gradientn(
    name = "Mean GVI (%)",
    colors = colors,
    breaks = breaks,
    labels = scales::number_format(accuracy = 0.1, suffix = "%"),
    limits = c(min(breaks), max(breaks)),
    guide = guide_colorbar(
      title.position = "top",
      title.hjust = 0.5,
      barwidth = 15,
      barheight = 1
    )
  ) +
  
  # Add borough labels
  geom_sf_text(
    aes(label = paste0(borough, "\n", round(mean_gvi, 1), "%")),
    size = 2.8,
    fontface = "bold",
    color = "black"
  ) +
  
  # Theme and styling
  theme_void() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5, size = 11, color = "gray30"),
    legend.position = "bottom",
    legend.title = element_text(size = 10),
    legend.text = element_text(size = 9),
    panel.background = element_rect(fill = "white", color = NA),
    plot.background = element_rect(fill = "white", color = NA),
    plot.margin = margin(20, 20, 20, 20)
  ) +
  
  labs(
    title = "Borough-level Green View Index Means",
    subtitle = paste0("Inner London, ", nrow(borough_map_data), " Boroughs | Range: ",
                      round(min(borough_map_data$mean_gvi), 1), "% - ",
                      round(max(borough_map_data$mean_gvi), 1), "%"),
    caption = "Source: Mapillary street-view imagery (2020-2024) | GVI computed via SegFormer"
  )

# Save the plot
output_file <- file.path(FIG_DIR, "fig_4_2c_borough_gvi_means.png")
ggsave(
  filename = output_file,
  plot = p,
  width = 10,
  height = 8,
  dpi = 300,
  bg = "white"
)

cat("✓ Figure saved to:", output_file, "\n")

# Also save borough statistics table
borough_table_file <- file.path(OUTPUT_DIR, "borough_gvi_statistics.csv")
write_csv(borough_stats, borough_table_file)
cat("✓ Borough statistics saved to:", borough_table_file, "\n")

# Print summary statistics
cat("\n=== Summary Statistics ===\n")
cat("Highest GVI Borough:", borough_stats$borough[1], "(", round(borough_stats$mean_gvi[1], 1), "%)\n")
cat("Lowest GVI Borough:", borough_stats$borough[nrow(borough_stats)], "(", round(borough_stats$mean_gvi[nrow(borough_stats)], 1), "%)\n")
cat("Range:", round(max(borough_stats$mean_gvi) - min(borough_stats$mean_gvi), 1), "percentage points\n")
cat("Coefficient of Variation:", round(sd(borough_stats$mean_gvi) / mean(borough_stats$mean_gvi) * 100, 1), "%\n")

cat("\n=== Borough Ranking (Highest to Lowest GVI) ===\n")
for(i in 1:nrow(borough_stats)) {
  cat(sprintf("%2d. %-25s %5.1f%% (n=%d LSOAs)\n", 
              i, 
              borough_stats$borough[i], 
              borough_stats$mean_gvi[i],
              borough_stats$n_lsoas[i]))
}

cat("\n=== Figure 4.2c Creation Complete ===\n")