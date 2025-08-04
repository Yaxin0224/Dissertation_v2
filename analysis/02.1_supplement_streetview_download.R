#!/usr/bin/env Rscript
# =============================================================================
# Script: 02.1_supplement_streetview_download.R
# Project: City Look - Urban Character Assessment
# Description: Supplements existing images by downloading 20 more per LSOA
# =============================================================================

library(tidyverse)
library(sf)
library(httr)
library(jsonlite)
library(here)
library(logger)

# Source configuration
source(here("R", "config.R"))

# Setup logging
log_appender(appender_file(here("logs", paste0("supplement_download_", 
                                               format(Sys.Date(), "%Y%m%d"), ".log"))))
log_threshold(INFO)
log_info("Starting supplementary image download")

# Constants
MAPILLARY_TOKEN <- "MLY|9922859457805691|cef02444f32c339cf09761b104ca4bb5"
SUPPLEMENT_TARGET <- 20  # Additional images to download
TOTAL_TARGET <- 100     # Total images per LSOA

# Load data
selected_lsoas <- read_csv(here("data", "processed", "selected_lsoas.csv"), 
                           show_col_types = FALSE)

# API functions (same as before)
search_mapillary_images <- function(bbox, access_token, limit = 500) {
  base_url <- "https://graph.mapillary.com/images"
  
  params <- list(
    access_token = access_token,
    fields = "id,captured_at,compass_angle,geometry,height,width",
    bbox = paste(bbox, collapse = ","),
    limit = limit
  )
  
  response <- GET(base_url, query = params)
  
  if(status_code(response) != 200) {
    return(NULL)
  }
  
  data <- content(response, "parsed")
  
  if(is.null(data$data) || length(data$data) == 0) {
    return(NULL)
  }
  
  images_list <- list()
  valid_count <- 0
  
  for(i in 1:length(data$data)) {
    img <- data$data[[i]]
    
    if(!is.null(img$geometry) && !is.null(img$geometry$coordinates)) {
      coords <- img$geometry$coordinates
      lon <- as.numeric(coords[1])
      lat <- as.numeric(coords[2])
    } else {
      next
    }
    
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
  
  if(valid_count > 0) {
    return(bind_rows(images_list))
  } else {
    return(NULL)
  }
}

download_mapillary_image <- function(image_id, output_path, access_token) {
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
  
  img_response <- GET(meta_data$thumb_2048_url, 
                      write_disk(output_path, overwrite = TRUE))
  
  return(status_code(img_response) == 200 && file.exists(output_path))
}

# Initialize tracking
supplement_results <- tibble()
all_new_metadata <- tibble()

cat("\n========================================\n")
cat("SUPPLEMENTARY DOWNLOAD - Adding 20 images per LSOA\n")
cat("========================================\n\n")

# Process each LSOA
for (i in 1:nrow(selected_lsoas)) {
  lsoa <- selected_lsoas[i, ]
  
  cat(paste0("\n[", i, "/", nrow(selected_lsoas), "] Processing: ", 
             lsoa$lsoa_code, " - ", lsoa$lsoa_name, "\n"))
  
  lsoa_dir <- here("data", "raw", "mapillary_images", lsoa$lsoa_code)
  
  # Check existing images
  existing_files <- list.files(lsoa_dir, pattern = "\\.jpg$", full.names = FALSE)
  existing_ids <- gsub("\\.jpg$", "", existing_files)
  existing_count <- length(existing_ids)
  
  cat(paste0("Existing images: ", existing_count, "\n"))
  
  # Read existing metadata if available
  metadata_file <- file.path(lsoa_dir, "metadata.csv")
  if(file.exists(metadata_file)) {
    existing_metadata <- read_csv(metadata_file, show_col_types = FALSE)
  } else {
    cat("Warning: No metadata file found\n")
    next
  }
  
  # Load boundaries
  shp_path <- here("data", "raw", "LB_LSOA2021_shp", "LB_shp", 
                   paste0(lsoa$borough, ".shp"))
  
  if(!file.exists(shp_path)) {
    next
  }
  
  borough_boundaries <- st_read(shp_path, quiet = TRUE)
  lsoa_boundary <- borough_boundaries %>%
    filter(lsoa21cd == lsoa$lsoa_code) %>%
    st_transform(4326)
  
  if (nrow(lsoa_boundary) == 0) {
    next
  }
  
  bbox <- st_bbox(lsoa_boundary)
  
  # Search for all available images
  cat("Searching for additional images...")
  available_images <- search_mapillary_images(
    c(bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]),
    MAPILLARY_TOKEN,
    limit = 500
  )
  
  # If not enough, expand search
  if(is.null(available_images) || nrow(available_images) < (existing_count + SUPPLEMENT_TARGET)) {
    cat(" expanding search area...")
    
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
  
  if(is.null(available_images)) {
    cat(" No images found!\n")
    next
  }
  
  # Filter out already downloaded images
  new_images <- available_images %>%
    filter(!id %in% existing_ids) %>%
    arrange(desc(captured_year), desc(width))
  
  cat(paste0("\nTotal available: ", nrow(available_images), 
             " | Already downloaded: ", existing_count,
             " | New candidates: ", nrow(new_images), "\n"))
  
  if(nrow(new_images) == 0) {
    cat("No new images available\n")
    supplement_results <- bind_rows(supplement_results,
                                    tibble(
                                      lsoa_code = lsoa$lsoa_code,
                                      existing_count = existing_count,
                                      new_available = 0,
                                      downloaded = 0,
                                      total_count = existing_count
                                    ))
    next
  }
  
  # Select additional images with spatial distribution
  # Simple approach: take the newest ones that aren't too close to existing ones
  selected_new <- new_images[1:min(SUPPLEMENT_TARGET, nrow(new_images)), ]
  
  cat(paste0("Downloading ", nrow(selected_new), " additional images...\n"))
  
  # Download new images
  download_count <- 0
  for (j in 1:nrow(selected_new)) {
    if(j %% 5 == 0) cat(paste0(j, "..."))
    
    image <- selected_new[j, ]
    image_filename <- paste0(image$id, ".jpg")
    image_path <- file.path(lsoa_dir, image_filename)
    
    success <- download_mapillary_image(image$id, image_path, MAPILLARY_TOKEN)
    
    if (success) {
      download_count <- download_count + 1
      
      # Add to metadata
      all_new_metadata <- bind_rows(
        all_new_metadata,
        image %>%
          mutate(
            lsoa_code = lsoa$lsoa_code,
            borough = lsoa$borough,
            imd_quintile = lsoa$inner_imd_quintile,
            filename = image_filename,
            download_timestamp = Sys.time(),
            download_batch = "supplement"
          )
      )
    }
    
    Sys.sleep(0.1)
  }
  
  cat(paste0("\nDownloaded ", download_count, " new images. ",
             "Total for LSOA: ", existing_count + download_count, "\n"))
  
  # Update metadata file
  # Fix: Ensure consistent data types
  if ("id" %in% names(existing_metadata)) {
    existing_metadata$id <- as.character(existing_metadata$id)
  }
  
  selected_new_for_metadata <- selected_new %>%
    mutate(
      id = as.character(id),  # Ensure id is character
      lsoa_code = lsoa$lsoa_code,
      lsoa_name = lsoa$lsoa_name,
      borough = lsoa$borough,
      downloaded = file.exists(file.path(lsoa_dir, paste0(id, ".jpg")))
    )
  
  updated_metadata <- bind_rows(
    existing_metadata,
    selected_new_for_metadata
  )
  
  write_csv(updated_metadata, metadata_file)
  
  # Record results
  supplement_results <- bind_rows(supplement_results,
                                  tibble(
                                    lsoa_code = lsoa$lsoa_code,
                                    existing_count = existing_count,
                                    new_available = nrow(new_images),
                                    downloaded = download_count,
                                    total_count = existing_count + download_count
                                  ))
}

# Save results
cat("\n========================================\n")
cat("SUPPLEMENT COMPLETE\n")
cat("========================================\n\n")

# Save new metadata
if(nrow(all_new_metadata) > 0) {
  supplement_metadata_path <- here("data", "processed", "streetview_metadata_supplement.csv")
  write_csv(all_new_metadata, supplement_metadata_path)
  
  # Also update the complete metadata file
  if(file.exists(here("data", "processed", "streetview_metadata.csv"))) {
    original_metadata <- read_csv(here("data", "processed", "streetview_metadata.csv"), 
                                  show_col_types = FALSE)
    complete_metadata <- bind_rows(original_metadata, all_new_metadata)
    write_csv(complete_metadata, here("data", "processed", "streetview_metadata_complete.csv"))
  }
}

# Save supplement summary
write_csv(supplement_results, here("data", "processed", "supplement_summary.csv"))

# Print summary
summary_stats <- supplement_results %>%
  summarise(
    total_lsoas = n(),
    total_new_downloads = sum(downloaded),
    avg_per_lsoa = round(mean(downloaded), 1),
    min_total = min(total_count),
    max_total = max(total_count),
    avg_total = round(mean(total_count), 1)
  )

cat("SUMMARY:\n")
cat(paste0("LSOAs processed: ", summary_stats$total_lsoas, "\n"))
cat(paste0("New images downloaded: ", summary_stats$total_new_downloads, "\n"))
cat(paste0("Average new images per LSOA: ", summary_stats$avg_per_lsoa, "\n"))
cat(paste0("Total images per LSOA: ", summary_stats$min_total, " - ", 
           summary_stats$max_total, " (avg: ", summary_stats$avg_total, ")\n"))

# Check which LSOAs now have 100+ images
lsoas_100plus <- supplement_results %>%
  filter(total_count >= 100) %>%
  nrow()

cat(paste0("\nLSOAs with 100+ images: ", lsoas_100plus, "/", nrow(supplement_results), "\n"))

# Show any LSOAs with less than 100
lsoas_under_100 <- supplement_results %>%
  filter(total_count < 100) %>%
  arrange(total_count)

if(nrow(lsoas_under_100) > 0) {
  cat("\nLSOAs with less than 100 images:\n")
  for(i in 1:nrow(lsoas_under_100)) {
    cat(paste0("  ", lsoas_under_100$lsoa_code[i], ": ", 
               lsoas_under_100$total_count[i], " images\n"))
  }
}

log_info("Supplementary download completed")
cat("\nDone!\n")