# ===================================
# City Look: 单个LSOA测试脚本
# 基于成功经验的重新设计
# ===================================

library(tidyverse)
library(sf)
library(httr)
library(jsonlite)
library(here)

cat("========================================\n")
cat("City Look: 单个LSOA街景图像收集测试\n")
cat("开始时间：", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")
cat("========================================\n\n")

# API设置
MAPILLARY_ACCESS_TOKEN <- "MLY|9922859457805691|cef02444f32c339cf09761b104ca4bb5"

# 读取数据
selected_lsoas <- read_csv(here("data", "processed", "selected_lsoas.csv"), show_col_types = FALSE)

# 选择第一个LSOA进行测试
test_lsoa <- selected_lsoas[1, ]
cat("测试LSOA:", test_lsoa$lsoa_code, "-", test_lsoa$lsoa_name, "\n")
cat("Borough:", test_lsoa$borough, "\n")
cat("IMD Quintile:", test_lsoa$inner_imd_quintile, "\n\n")

# 加载边界
shp_path <- here("data", "raw", "LB_LSOA2021_shp", "LB_shp", 
                 paste0(test_lsoa$borough, ".shp"))
borough_boundaries <- st_read(shp_path, quiet = TRUE)

# 获取LSOA边界
lsoa_boundary <- borough_boundaries %>%
  filter(lsoa21cd == test_lsoa$lsoa_code) %>%
  st_transform(4326)

if(nrow(lsoa_boundary) == 0) {
  stop("未找到LSOA边界")
}

# 获取边界框
bbox <- st_bbox(lsoa_boundary)
cat("边界框坐标:\n")
cat("  xmin:", bbox["xmin"], "\n")
cat("  ymin:", bbox["ymin"], "\n")
cat("  xmax:", bbox["xmax"], "\n")
cat("  ymax:", bbox["ymax"], "\n\n")

# 基于成功脚本的API请求函数
search_mapillary_images_v2 <- function(bbox, access_token, limit = 100) {
  
  base_url <- "https://graph.mapillary.com/images"
  
  # 使用成功脚本的参数设置
  params <- list(
    access_token = access_token,
    fields = "id,captured_at,compass_angle,geometry,height,width",
    bbox = paste(bbox, collapse = ","),
    limit = limit
  )
  
  cat("发送API请求...\n")
  response <- GET(base_url, query = params)
  
  cat("响应状态码:", status_code(response), "\n")
  
  if(status_code(response) != 200) {
    cat("错误响应:\n")
    print(content(response, "text"))
    return(NULL)
  }
  
  # 解析响应
  data <- content(response, "parsed")
  
  if(is.null(data$data) || length(data$data) == 0) {
    cat("未找到图像数据\n")
    return(NULL)
  }
  
  cat("找到", length(data$data), "个图像\n")
  
  # 转换为数据框（使用成功脚本的方法）
  images_list <- list()
  
  for(i in 1:length(data$data)) {
    img <- data$data[[i]]
    
    # 提取坐标
    if(!is.null(img$geometry) && !is.null(img$geometry$coordinates)) {
      coords <- img$geometry$coordinates
      lon <- as.numeric(coords[1])
      lat <- as.numeric(coords[2])
    } else {
      next
    }
    
    # 处理时间戳
    captured_timestamp <- as.numeric(img$captured_at)
    if(!is.na(captured_timestamp)) {
      captured_date <- as.POSIXct(captured_timestamp/1000, origin="1970-01-01", tz="UTC")
      captured_year <- as.numeric(format(captured_date, "%Y"))
    } else {
      captured_date <- NA
      captured_year <- NA
    }
    
    images_list[[i]] <- data.frame(
      image_id = img$id,
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
  
  # 合并结果
  if(length(images_list) > 0) {
    images_df <- bind_rows(images_list)
    images_df <- images_df[!is.na(images_df$image_id), ]
    return(images_df)
  } else {
    return(NULL)
  }
}

# 测试标准边界框
cat("\n=== 测试1: 标准边界框 ===\n")
images_standard <- search_mapillary_images_v2(
  c(bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]),
  MAPILLARY_ACCESS_TOKEN,
  limit = 150
)

if(!is.null(images_standard)) {
  cat("\n图像年份分布:\n")
  year_stats <- images_standard %>%
    group_by(captured_year) %>%
    summarise(count = n(), .groups = 'drop') %>%
    arrange(desc(captured_year))
  print(year_stats)
  
  # 筛选2020年后的图像
  recent_images <- images_standard %>%
    filter(captured_year >= 2020)
  cat("\n2020年后的图像:", nrow(recent_images), "张\n")
}

# 如果图像不足，尝试扩大范围
if(is.null(images_standard) || nrow(images_standard) < 60) {
  cat("\n=== 测试2: 扩大搜索范围 ===\n")
  
  # 扩大20%
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
  
  images_expanded <- search_mapillary_images_v2(
    expanded_bbox,
    MAPILLARY_ACCESS_TOKEN,
    limit = 200
  )
  
  if(!is.null(images_expanded)) {
    cat("扩大范围后找到:", nrow(images_expanded), "张图像\n")
  }
}

# 创建测试输出文件夹
test_dir <- here("data", "raw", "mapillary_test")
dir.create(test_dir, showWarnings = FALSE, recursive = TRUE)

# 保存测试结果
if(!is.null(images_standard)) {
  # 添加LSOA信息
  images_standard$lsoa_code <- test_lsoa$lsoa_code
  images_standard$lsoa_name <- test_lsoa$lsoa_name
  images_standard$borough <- test_lsoa$borough
  
  # 保存元数据
  write_csv(images_standard, 
            file.path(test_dir, paste0(test_lsoa$lsoa_code, "_metadata.csv")))
  
  cat("\n✅ 测试成功！元数据已保存到:", test_dir, "\n")
  
  # 测试下载一张图像
  cat("\n=== 测试图像下载 ===\n")
  test_image <- images_standard[1, ]
  
  # 使用成功脚本的下载方法
  download_test <- function(image_id, save_path) {
    # 第一步：获取图像URL
    meta_url <- sprintf("https://graph.mapillary.com/%s?access_token=%s&fields=thumb_2048_url",
                        image_id, MAPILLARY_ACCESS_TOKEN)
    
    meta_response <- GET(meta_url)
    
    if(status_code(meta_response) == 200) {
      meta_data <- content(meta_response, "parsed")
      
      if(!is.null(meta_data$thumb_2048_url)) {
        # 第二步：下载图像
        img_response <- GET(meta_data$thumb_2048_url, 
                            write_disk(save_path, overwrite = TRUE))
        
        if(status_code(img_response) == 200) {
          return(TRUE)
        }
      }
    }
    return(FALSE)
  }
  
  test_path <- file.path(test_dir, "test_image.jpg")
  if(download_test(test_image$image_id, test_path)) {
    cat("✅ 图像下载成功！\n")
    cat("文件大小:", file.info(test_path)$size / 1024, "KB\n")
  } else {
    cat("❌ 图像下载失败\n")
  }
  
} else {
  cat("\n❌ 未找到任何图像\n")
}

cat("\n测试完成！\n")