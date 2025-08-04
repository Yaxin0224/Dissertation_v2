#!/usr/bin/env Rscript
# =============================================================================
# Script: 02_streetview_download_v3.R
# Project: City Look - Urban Character Assessment
# Description: Downloads street view images from Mapillary API for selected LSOAs
# Note: Based on successful test results
# =============================================================================

# Load required libraries
library(tidyverse)
library(sf)
library(httr)
library(jsonlite)
library(here)
library(lubridate)
library(logger)

# Source configuration file
source(here("R", "config.R"))

# =============================================================================
# 1. SETUP AND INITIALIZATION
# =============================================================================

# Create necessary directories
dir.create(here("logs"), showWarnings = FALSE, recursive = TRUE)
dir.create(here("data", "raw", "mapillary_images"), showWarnings = FALSE, recursive = TRUE)
dir.create(here("reports"), showWarnings = FALSE, recursive = TRUE)
dir.create(here("figures"), showWarnings = FALSE, recursive = TRUE)

# Set up logging
log_appender(appender_file(here("logs", paste0("streetview_download_", 
                                               format(Sys.Date(), "%Y%m%d"), ".log"))))
log_threshold(INFO)
log_info("Starting street view download process")

# Mapillary API configuration
MAPILLARY_TOKEN <- "MLY|9922859457805691|cef02444f32c339cf09761b104ca4bb5"
MAPILLARY_BASE_URL <- "https://graph.mapillary.com"
IMAGES_PER_LSOA_TARGET <- 80
IMAGES_PER_LSOA_MIN <- 60

# =============================================================================
# 2. LOAD DATA
# =============================================================================

log_info("Loading selected LSOAs data")

# Load selected LSOAs
selected_lsoas <- read_csv(here("data", "processed", "selected_lsoas.csv"), 
                           show_col_types = FALSE)
log_info(paste("Loaded", nrow(selected_lsoas), "LSOAs"))

# =============================================================================
# 3. API FUNCTIONS (Based on successful test)
# =============================================================================

#' Search for Mapillary images within a bounding box
search_mapillary_images <- function(bbox, access_token, limit = 500) {
  
  base_url <- "https://graph.mapillary.com/images"
  
  # Parameters that work based on our test
  params <- list(
    access_token = access_token,
    fields = "id,captured_at,compass_angle,geometry,height,width",
    bbox = paste(bbox, collapse = ","),
    limit = limit
  )
  
  response <- GET(base_url, query = params)
  
  if(status_code(response) != 200) {
    log_error(paste("API request failed with status:", status_code(response)))
    return(NULL)
  }
  
  # Parse response
  data <- content(response, "parsed")
  
  if(is.null(data$data) || length(data$data) == 0) {
    return(NULL)
  }
  
  # Convert to data frame
  images_list <- list()
  valid_count <- 0
  
  for(i in 1:length(data$data)) {
    img <- data$data[[i]]
    
    # Extract coordinates
    if(!is.null(img$geometry) && !is.null(img$geometry$coordinates)) {
      coords <- img$geometry$coordinates
      lon <- as.numeric(coords[1])
      lat <- as.numeric(coords[2])
    } else {
      next
    }
    
    # Handle timestamp
    captured_timestamp <- as.numeric(img$captured_at)
    if(!is.na(captured_timestamp)) {
      captured_date <- as.POSIXct(captured_timestamp/1000, origin="1970-01-01", tz="UTC")
      captured_year <- as.numeric(format(captured_date, "%Y"))
    } else {
      captured_date <- NA
      captured_year <- NA
    }
    
    valid_count <- valid_count + 1
    
    images_list[[valid_count]] <- data.frame(
      id = img$id,
      longitude = lon,
      latitude = lat,
      captured_at = captured_timestamp,
      captured_date = captured_date,
      captured_year = captured_year,
      compass_angle = as.numeric(img$compass_angle %||% NA),
      height = as.integer(img$height %||% NA),
      width = as.integer(img$width %||% NA),
      stringsAsFactors = FALSE
    )
  }
  
  # Combine results
  if(valid_count > 0) {
    images_df <- bind_rows(images_list)
    return(images_df)
  } else {
    return(NULL)
  }
}

#' Download a single image from Mapillary
download_mapillary_image <- function(image_id, output_path, access_token) {
  
  # Step 1: Get image URL
  meta_url <- sprintf("https://graph.mapillary.com/%s?access_token=%s&fields=thumb_2048_url",
                      image_id, access_token)
  
  meta_response <- GET(meta_url)
  
  if(status_code(meta_response) != 200) {
    return(FALSE)
  }
  
  meta_data <- content(meta_response, "parsed")
  
  if(is.null(meta_data$thumb_2048_url)) {
    return(FALSE)
  }
  
  # Step 2: Download image
  img_response <- GET(meta_data$thumb_2048_url, 
                      write_disk(output_path, overwrite = TRUE))
  
  return(status_code(img_response) == 200 && file.exists(output_path))
}

#' Select images with good spatial distribution
spatial_selection <- function(images_df, target_count) {
  
  if(nrow(images_df) <= target_count) {
    return(images_df)
  }
  
  # Sort by year (newest first) and quality
  images_df <- images_df %>%
    arrange(desc(captured_year), desc(width))
  
  # Simple selection: take newest images with minimum spacing
  selected <- images_df[1, ]
  
  for(i in 2:nrow(images_df)) {
    if(nrow(selected) >= target_count) break
    
    # Check minimum distance to already selected images
    current_img <- images_df[i, ]
    min_dist <- min(sqrt((selected$longitude - current_img$longitude)^2 + 
                           (selected$latitude - current_img$latitude)^2))
    
    # Accept if far enough (about 10-20 meters in degrees)
    if(min_dist > 0.0001) {
      selected <- rbind(selected, current_img)
    }
  }
  
  # If still not enough, add more without distance constraint
  if(nrow(selected) < target_count) {
    remaining <- images_df[!images_df$id %in% selected$id, ]
    needed <- target_count - nrow(selected)
    if(nrow(remaining) >= needed) {
      selected <- rbind(selected, remaining[1:needed, ])
    } else {
      selected <- rbind(selected, remaining)
    }
  }
  
  return(selected)
}

# =============================================================================
# 4. MAIN DOWNLOAD PROCESS
# =============================================================================

# Initialize results tracking
download_results <- tibble()
all_image_metadata <- tibble()

# Process each LSOA
cat("\n")
for (i in 1:nrow(selected_lsoas)) {
  
  lsoa <- selected_lsoas[i, ]
  
  cat(paste0("\n", strrep("=", 60), "\n"))
  cat(paste0("[", i, "/", nrow(selected_lsoas), "] Processing: ", 
             lsoa$lsoa_code, " - ", lsoa$lsoa_name, "\n"))
  cat(paste0("Borough: ", lsoa$borough, " | IMD Quintile: ", lsoa$inner_imd_quintile, "\n"))
  cat(strrep("-", 60), "\n")
  
  log_info(paste("Processing LSOA:", lsoa$lsoa_code, "in", lsoa$borough))
  
  # Create LSOA output directory
  lsoa_dir <- here("data", "raw", "mapillary_images", lsoa$lsoa_code)
  dir.create(lsoa_dir, showWarnings = FALSE, recursive = TRUE)
  
  # Load borough shapefile
  shp_path <- here("data", "raw", "LB_LSOA2021_shp", "LB_shp", 
                   paste0(lsoa$borough, ".shp"))
  
  if(!file.exists(shp_path)) {
    log_error(paste("Shapefile not found for borough:", lsoa$borough))
    next
  }
  
  borough_boundaries <- st_read(shp_path, quiet = TRUE)
  
  # Get LSOA boundary
  lsoa_boundary <- borough_boundaries %>%
    filter(lsoa21cd == lsoa$lsoa_code) %>%
    st_transform(4326)
  
  if (nrow(lsoa_boundary) == 0) {
    log_error(paste("No boundary found for LSOA:", lsoa$lsoa_code))
    next
  }
  
  # Get bounding box
  bbox <- st_bbox(lsoa_boundary)
  
  # Search for images
  cat("Searching for images...")
  available_images <- search_mapillary_images(
    c(bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]),
    MAPILLARY_TOKEN,
    limit = 500
  )
  
  # If not enough images, try expanded area
  if(is.null(available_images) || nrow(available_images) < IMAGES_PER_LSOA_TARGET) {
    cat(" expanding search area...")
    
    # Expand by 20%
    expansion <- 0.2
    center_x <- (bbox["xmin"] + bbox["xmax"]) / 2
    center_y <- (bbox["ymin"] + bbox["ymax"]) / 2
    width <- bbox["xmax"] - bbox["xmin"]
    height <- bbox["ymax"] - bbox["ymin"]
    
    expanded_bbox <- c(
      center_x - width * (0.5 + expansion/2),
      center_y - height * (0.5 + expansion/2),
      center_x + width * (0.5 + expansion/2),
      center_y + height * (0.5 + expansion/2)
    )
    
    expanded_images <- search_mapillary_images(expanded_bbox, MAPILLARY_TOKEN, limit = 500)
    
    if(!is.null(expanded_images)) {
      if(is.null(available_images)) {
        available_images <- expanded_images
      } else {
        available_images <- bind_rows(available_images, expanded_images) %>%
          distinct(id, .keep_all = TRUE)
      }
    }
  }
  
  if(is.null(available_images) || nrow(available_images) == 0) {
    cat(" No images found!\n")
    log_warn(paste("No images found for LSOA:", lsoa$lsoa_code))
    
    download_results <- bind_rows(download_results, 
                                  tibble(
                                    lsoa_code = lsoa$lsoa_code,
                                    borough = lsoa$borough,
                                    imd_quintile = lsoa$inner_imd_quintile,
                                    images_available = 0,
                                    images_downloaded = 0,
                                    status = "no_images_found"
                                  ))
    next
  }
  
  cat(paste0("\nFound ", nrow(available_images), " images\n"))
  log_info(paste("Found", nrow(available_images), "images for LSOA:", lsoa$lsoa_code))
  
  # Show year distribution
  year_dist <- available_images %>%
    group_by(captured_year) %>%
    summarise(count = n(), .groups = 'drop') %>%
    arrange(desc(captured_year))
  
  cat("\nYear distribution:\n")
  for(j in 1:min(5, nrow(year_dist))) {
    cat(paste0("  ", year_dist$captured_year[j], ": ", year_dist$count[j], " images\n"))
  }
  
  # Select images with spatial distribution
  selected_images <- spatial_selection(available_images, IMAGES_PER_LSOA_TARGET)
  
  cat(paste0("\nSelected ", nrow(selected_images), " images for download\n"))
  log_info(paste("Selected", nrow(selected_images), "images for download"))
  
  # Download images
  download_count <- 0
  cat("\nDownloading: ")
  
  for (j in 1:nrow(selected_images)) {
    if(j %% 10 == 0) cat(paste0(j, "..."))
    
    image <- selected_images[j, ]
    image_filename <- paste0(image$id, ".jpg")
    image_path <- file.path(lsoa_dir, image_filename)
    
    # Skip if already exists
    if(file.exists(image_path) && file.info(image_path)$size > 10000) {
      download_count <- download_count + 1
      next
    }
    
    # Download image
    success <- download_mapillary_image(image$id, image_path, MAPILLARY_TOKEN)
    
    if (success) {
      download_count <- download_count + 1
      
      # Add metadata
      all_image_metadata <- bind_rows(
        all_image_metadata,
        image %>%
          mutate(
            lsoa_code = lsoa$lsoa_code,
            borough = lsoa$borough,
            imd_quintile = lsoa$inner_imd_quintile,
            filename = image_filename,
            download_timestamp = Sys.time()
          )
      )
    }
    
    # Rate limiting
    Sys.sleep(0.1)
  }
  
  cat(paste0("\nDownloaded ", download_count, " images successfully\n"))
  
  # Save LSOA metadata
  lsoa_metadata <- selected_images %>%
    mutate(
      lsoa_code = lsoa$lsoa_code,
      lsoa_name = lsoa$lsoa_name,
      borough = lsoa$borough,
      downloaded = file.exists(file.path(lsoa_dir, paste0(id, ".jpg")))
    )
  
  write_csv(lsoa_metadata, file.path(lsoa_dir, "metadata.csv"))
  
  # Record results
  download_results <- bind_rows(
    download_results,
    tibble(
      lsoa_code = lsoa$lsoa_code,
      borough = lsoa$borough,
      imd_quintile = lsoa$inner_imd_quintile,
      images_available = nrow(available_images),
      images_selected = nrow(selected_images),
      images_downloaded = download_count,
      status = ifelse(download_count >= IMAGES_PER_LSOA_MIN, "success", "insufficient_images")
    )
  )
  
  log_info(paste("Downloaded", download_count, "images for LSOA:", lsoa$lsoa_code))
}

# =============================================================================
# 5. SAVE METADATA AND REPORTS
# =============================================================================

cat(paste0("\n", strrep("=", 60), "\n"))
cat("FINALIZING RESULTS\n")
cat(strrep("=", 60), "\n")

log_info("Saving metadata and generating reports")

# Save complete metadata
if(nrow(all_image_metadata) > 0) {
  metadata_path <- here("data", "processed", "streetview_metadata.csv")
  write_csv(all_image_metadata, metadata_path)
  log_info(paste("Saved image metadata to:", metadata_path))
}

# Save download summary
summary_path <- here("data", "processed", "download_summary.csv")
write_csv(download_results, summary_path)
log_info(paste("Saved download summary to:", summary_path))

# =============================================================================
# 6. GENERATE REPORT
# =============================================================================

# Summary statistics
total_downloaded <- sum(download_results$images_downloaded)
successful_lsoas <- sum(download_results$status == "success")
failed_lsoas <- sum(download_results$status != "success")

# Generate text report
report_lines <- c(
  "MAPILLARY STREET VIEW DOWNLOAD REPORT",
  "=====================================",
  paste("Generated:", Sys.time()),
  "",
  "OVERALL SUMMARY",
  "---------------",
  paste("Total LSOAs processed:", nrow(download_results)),
  paste("Successful LSOAs (>=", IMAGES_PER_LSOA_MIN, "images):", successful_lsoas),
  paste("Failed LSOAs:", failed_lsoas),
  paste("Total images downloaded:", total_downloaded),
  paste("Average images per LSOA:", round(mean(download_results$images_downloaded), 1)),
  "",
  "BY IMD QUINTILE",
  "---------------"
)

# Summary by IMD quintile
imd_summary <- download_results %>%
  group_by(imd_quintile) %>%
  summarise(
    n_lsoas = n(),
    total_images = sum(images_downloaded),
    avg_images = round(mean(images_downloaded), 1),
    success_rate = round(sum(status == "success") / n() * 100, 1),
    .groups = 'drop'
  )

for (i in 1:nrow(imd_summary)) {
  report_lines <- c(report_lines,
                    paste0("Quintile ", imd_summary$imd_quintile[i], ": ",
                           imd_summary$total_images[i], " images from ",
                           imd_summary$n_lsoas[i], " LSOAs (avg: ",
                           imd_summary$avg_images[i], ", success rate: ",
                           imd_summary$success_rate[i], "%)")
  )
}

report_lines <- c(report_lines, "", "BY BOROUGH", "----------")

# Summary by borough
borough_summary <- download_results %>%
  group_by(borough) %>%
  summarise(
    n_lsoas = n(),
    total_images = sum(images_downloaded),
    avg_images = round(mean(images_downloaded), 1),
    .groups = 'drop'
  ) %>%
  arrange(desc(total_images))

for (i in 1:nrow(borough_summary)) {
  report_lines <- c(report_lines,
                    paste0(borough_summary$borough[i], ": ",
                           borough_summary$total_images[i], " images from ",
                           borough_summary$n_lsoas[i], " LSOAs (avg: ",
                           borough_summary$avg_images[i], ")")
  )
}

# Failed LSOAs
if (failed_lsoas > 0) {
  report_lines <- c(report_lines, "", "FAILED/INSUFFICIENT LSOAS", "-------------------------")
  
  failed_details <- download_results %>%
    filter(status != "success")
  
  for (i in 1:nrow(failed_details)) {
    report_lines <- c(report_lines,
                      paste0(failed_details$lsoa_code[i], " (",
                             failed_details$borough[i], "): ",
                             failed_details$images_downloaded[i], " images - ",
                             failed_details$status[i])
    )
  }
}

# Save report
report_path <- here("reports", paste0("download_report_", format(Sys.Date(), "%Y%m%d"), ".txt"))
writeLines(report_lines, report_path)
log_info(paste("Saved download report to:", report_path))

# Print report
cat("\n")
cat(paste(report_lines, collapse = "\n"))
cat("\n")

# =============================================================================
# 7. CREATE VISUALIZATIONS
# =============================================================================

library(ggplot2)

# Plot 1: Images by IMD quintile
p1 <- download_results %>%
  ggplot(aes(x = factor(imd_quintile), y = images_downloaded, fill = factor(imd_quintile))) +
  geom_boxplot(alpha = 0.7) +
  geom_point(position = position_jitter(width = 0.2), alpha = 0.5) +
  geom_hline(yintercept = IMAGES_PER_LSOA_MIN, linetype = "dashed", color = "red") +
  geom_hline(yintercept = IMAGES_PER_LSOA_TARGET, linetype = "dashed", color = "blue") +
  scale_fill_viridis_d(name = "IMD Quintile") +
  labs(
    title = "Street View Images Downloaded by IMD Quintile",
    x = "IMD Quintile (1 = Least Deprived)",
    y = "Number of Images Downloaded",
    caption = paste("Red line =", IMAGES_PER_LSOA_MIN, "(minimum) | Blue line =", IMAGES_PER_LSOA_TARGET, "(target)")
  ) +
  theme_minimal() +
  theme(legend.position = "none")

ggsave(here("figures", paste0("download_summary_", format(Sys.Date(), "%Y%m%d"), ".png")),
       p1, width = 10, height = 6, dpi = 300)

# Plot 2: Success rate by borough
p2 <- download_results %>%
  mutate(success_binary = ifelse(status == "success", 1, 0)) %>%
  group_by(borough) %>%
  summarise(
    success_rate = mean(success_binary) * 100,
    avg_images = mean(images_downloaded),
    .groups = 'drop'
  ) %>%
  ggplot(aes(x = reorder(borough, success_rate), y = success_rate)) +
  geom_col(aes(fill = avg_images), alpha = 0.8) +
  geom_text(aes(label = paste0(round(success_rate), "%")), 
            hjust = -0.2, size = 3) +
  scale_fill_viridis_c(name = "Avg Images") +
  coord_flip() +
  labs(
    title = "Download Success Rate by Borough",
    x = "Borough",
    y = "Success Rate (%)"
  ) +
  theme_minimal()

ggsave(here("figures", paste0("borough_success_", format(Sys.Date(), "%Y%m%d"), ".png")),
       p2, width = 10, height = 8, dpi = 300)

log_info("Download process completed successfully")

cat(paste0("\n", strrep("=", 60), "\n"))
cat("DOWNLOAD COMPLETE!\n")
cat(strrep("=", 60), "\n")
cat(paste("\nTotal time:", round(difftime(Sys.time(), start_time, units = "mins"), 1), "minutes\n"))
cat("\nNext steps:\n")
cat("1. Check download_summary.csv for detailed results\n")
cat("2. Review images in data/raw/mapillary_images/\n")
cat("3. Proceed with image analysis (GVI calculation)\n")

# =============================================================================
# END OF SCRIPT
# =============================================================================