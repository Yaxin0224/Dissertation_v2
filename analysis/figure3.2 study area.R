#!/usr/bin/env Rscript
# =============================================================================
# Figure 3.2: Study Area Map for Methodology Chapter
# Shows all 1,731 Inner London LSOAs with IMD quintiles and data coverage
# =============================================================================

# Load required libraries ------------------------------------------------------
suppressPackageStartupMessages({
  library(tidyverse)
  library(sf)
  library(tmap)
  library(viridis)
  library(ggplot2)
  library(cowplot)
  library(here)
  library(scales)
})

# Global theme -----------------------------------------------------------------
theme_set(theme_minimal(base_size = 12))
tmap_mode("plot")

# =============================================================================
# 1. LOAD DATA
# =============================================================================

cat("\n========== LOADING DATA FOR STUDY AREA MAP ==========\n")

# File paths - 根据你的项目结构调整
DATA_PATH <- here("output", "full_london_gvi", "inner_london_all_lsoas_complete.csv")
SHAPEFILE_PATH <- here(
  "data", "raw", "statistical-gis-boundaries-london", "ESRI",
  "LSOA_2011_London_gen_MHW.shp"
)

# Load GVI data ----------------------------------------------------------------
gvi_data <- read_csv(DATA_PATH, show_col_types = FALSE) |>
  mutate(
    imd_quintile = factor(
      imd_quintile,
      levels = c("Q1_Least", "Q2", "Q3", "Q4", "Q5_Most"),
      labels = c(
        "Q1 (Least Deprived)", "Q2", "Q3", "Q4", "Q5 (Most Deprived)"
      )
    ),
    data_coverage = factor(
      data_source,
      levels = c("measured", "predicted"),
      labels = c("Direct Measurement", "Imputed")
    )
  )

# Load LSOA boundaries ---------------------------------------------------------
lsoa_boundaries <- st_read(SHAPEFILE_PATH, quiet = TRUE) |>
  st_transform(27700) |>  # British National Grid
  rename_with(tolower)

# Join spatial and attribute data ---------------------------------------------
spatial_data <- lsoa_boundaries |>
  inner_join(gvi_data, by = c("lsoa11cd" = "lsoa_code")) |>
  st_make_valid()

# Borough boundaries for reference --------------------------------------------
borough_boundaries <- spatial_data |>
  group_by(borough) |>
  summarise(geometry = st_union(geometry), .groups = "drop") |>
  st_make_valid()

cat(paste("✓ Loaded", nrow(spatial_data), "LSOAs with spatial data\n"))
cat(paste("✓ Direct measurement:", sum(gvi_data$data_source == "measured"), "LSOAs\n"))
cat(paste("✓ Imputed:", sum(gvi_data$data_source == "predicted"), "LSOAs\n"))

# =============================================================================
# 2. CREATE MAIN MAP - IMD QUINTILES
# =============================================================================

cat("\n========== CREATING STUDY AREA MAP ==========\n")

# Color palette ----------------------------------------------------------------
imd_colors <- c(
  "Q1 (Least Deprived)" = "#2E7D32",
  "Q2" = "#66BB6A",
  "Q3" = "#FFF176",
  "Q4" = "#FF9800",
  "Q5 (Most Deprived)" = "#C62828"
)

# Main map ---------------------------------------------------------------------
main_map <- ggplot() +
  # LSOAs by IMD quintile
  geom_sf(
    data = spatial_data,
    aes(fill = imd_quintile),
    color = NA,
    size = 0.05
  ) +
  # Borough outlines
  geom_sf(
    data = borough_boundaries,
    fill = NA,
    color = "grey30",
    size = 0.5,
    linetype = "solid"
  ) +
  # Fill scale & legend (move to bottom-right, inside plot)
  scale_fill_manual(
    values = imd_colors,
    name = "IMD Quintile",
    guide = guide_legend(
      title.position = "top",
      title.hjust = 0.5,
      label.position = "right",
      keyheight = unit(1, "lines"),
      keywidth = unit(1.5, "lines"),
      ncol = 1
    )
  ) +
  # City of London note
  annotate(
    "text",
    x = 532750, y = 181330,
    label = "City of\nLondon\n(excluded)",
    size = 3, color = "grey40", fontface = "italic", lineheight = 0.9
  ) +
  # Annotations
  labs(
    title = "Study Area: Inner London LSOAs by Deprivation Level",
    subtitle = paste(
      "1,731 Lower Layer Super Output Areas (LSOAs) classified by Index of Multiple Deprivation 2019",
      "City of London excluded due to its unique administrative and demographic characteristics",
      sep = "\n"
    ),
    caption = "Source: ONS LSOA boundaries 2011, MHCLG IMD 2019\nProjection: British National Grid (EPSG:27700)"
  ) +
  # Theme
  theme_void() +
  theme(
    plot.title = element_text(size = 14, face = "bold", hjust = 0),
    plot.subtitle = element_text(size = 10, hjust = 0, margin = margin(t = 5, b = 10)),
    plot.caption = element_text(size = 8, hjust = 0, color = "grey50"),
    # 关键：图例移动到最右下角，并锚定右下角，避免遮挡地图主体
    legend.position = c(0.98, 0.03),
    legend.justification = c(1, 0),
    legend.background = element_rect(fill = "white", color = NA),
    legend.title = element_text(size = 10, face = "bold"),
    legend.text = element_text(size = 9),
    plot.margin = margin(10, 10, 10, 10)
  )

# =============================================================================
# 3. CREATE INSET MAP - DATA COVERAGE
# =============================================================================

# Imputed LSOAs ----------------------------------------------------------------
imputed_lsoas <- spatial_data |>
  filter(data_source == "predicted")

# Inset map (will be placed at top-right) -------------------------------------
inset_map <- ggplot() +
  geom_sf(
    data = spatial_data,
    fill = "grey90",
    color = "grey95",
    size = 0.05
  ) +
  geom_sf(
    data = imputed_lsoas,
    fill = "#E91E63",
    color = NA,
    alpha = 0.8
  ) +
  geom_sf(
    data = borough_boundaries,
    fill = NA,
    color = "grey50",
    size = 0.3
  ) +
  labs(
    title = "Data Coverage",
    subtitle = paste0(nrow(imputed_lsoas), " LSOAs (4%) required imputation")
  ) +
  theme_void() +
  theme(
    plot.title = element_text(size = 10, face = "bold"),
    plot.subtitle = element_text(size = 8, color = "grey50"),
    plot.margin = margin(5, 5, 5, 5),
    # 给 inset 一个白底 + 细边，便于与主图分离
    plot.background = element_rect(fill = "white", color = "grey60", linewidth = 0.5)
  )

# =============================================================================
# 4. (可选) 统计表构建，但不加入最终图像
# =============================================================================

# 如需保留统计结果供后续表格/报告使用，可计算但不绘制到最终成图
stats_summary <- gvi_data |>
  group_by(imd_quintile) |>
  summarise(
    n = n(),
    mean_gvi = mean(mean_gvi, na.rm = TRUE),
    .groups = "drop"
  )

# =============================================================================
# 5. COMBINE ELEMENTS
# =============================================================================

# 关键调整：
# - inset_map 放到右上角（x ~ 0.70-0.72, y ~ 0.72-0.74 之间），
#   width/height 稍小，避免与标题/图例互相遮挡
# - 不再 draw stats_plot（LSOA distribution table 被移除）
final_figure <- ggdraw() +
  draw_plot(main_map, x = 0, y = 0, width = 1, height = 1) +
  # Data coverage inset：右上角
  draw_plot(
    inset_map,
    x = 0.65,   # 水平位置（越大越靠右）
    y = 0.68,   # 垂直位置（越大越靠上）
    width = 0.32,
    height = 0.30
  )

# =============================================================================
# 6. SAVE FIGURE
# =============================================================================

output_dir <- here("figures", "methodology")
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

ggsave(
  filename = file.path(output_dir, "figure_3_2_study_area_map.png"),
  plot = final_figure,
  width = 12,
  height = 10,
  dpi = 300,
  bg = "white"
)

ggsave(
  filename = file.path(output_dir, "figure_3_2_study_area_map.pdf"),
  plot = final_figure,
  width = 12,
  height = 10,
  device = "pdf"
)

cat("\n✓ Study area map saved to:", file.path(output_dir, "figure_3_2_study_area_map.png"), "\n")

# =============================================================================
# 7. ALTERNATIVE: INTERACTIVE MAP USING TMAP
# =============================================================================

tmap_mode("view")

interactive_map <- tm_shape(spatial_data) +
  tm_polygons(
    col = "imd_quintile",
    palette = imd_colors,
    title = "IMD Quintile",
    popup.vars = c(
      "LSOA Name" = "lsoa_name",
      "Borough" = "borough",
      "IMD Score" = "imd_score",
      "Mean GVI (%)" = "mean_gvi",
      "Data Source" = "data_source"
    ),
    id = "lsoa11cd",
    alpha = 0.8
  ) +
  tm_shape(borough_boundaries) +
  tm_borders(col = "black", lwd = 2) +
  tm_basemap(server = "OpenStreetMap") +
  tm_layout(
    title = "Interactive Study Area Map",
    title.position = c("left", "top")
  )

tmap_save(
  tm = interactive_map,
  filename = file.path(output_dir, "figure_3_2_study_area_interactive.html")
)

cat("✓ Interactive map saved to:", file.path(output_dir, "figure_3_2_study_area_interactive.html"), "\n")

# =============================================================================
# 8. PRINT SUMMARY STATISTICS
# =============================================================================

cat("\n========== STUDY AREA SUMMARY ==========\n")
cat("Total LSOAs:", nrow(spatial_data), "\n")
cat("Total area:", round(sum(st_area(spatial_data)) / 1e6, 1), "km²\n")
cat("Number of boroughs:", n_distinct(spatial_data$borough), "\n")
cat("\nIMD Quintile Distribution:\n")
print(table(spatial_data$imd_quintile))
cat("\nData Coverage:\n")
print(table(spatial_data$data_coverage))
cat("\nBorough-level Statistics:\n")
spatial_data |>
  st_drop_geometry() |>
  group_by(borough) |>
  summarise(
    n_lsoas = n(),
    mean_imd = round(mean(imd_score), 1),
    mean_gvi = round(mean(mean_gvi), 1),
    pct_imputed = round(100 * sum(data_source == "predicted") / n(), 1),
    .groups = "drop"
  ) |>
  arrange(desc(mean_imd)) |>
  print(n = 14)

cat("\n========== FIGURE GENERATION COMPLETE ==========\n")
