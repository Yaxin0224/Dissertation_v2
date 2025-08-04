# ============================================
# City Look Dissertation v2
# config.R - 项目配置文件
# 
# 存放所有项目级别的配置参数
# 放置在 R/ 文件夹中
# 最后更新：2025-08-01
# ============================================

# ============================================
# 1. API配置
# ============================================

# Mapillary API Token
MAPILLARY_ACCESS_TOKEN <- "MLY|9922859457805691|cef02444f32c339cf09761b104ca4bb5"

# API端点
MAPILLARY_API_BASE <- "https://graph.mapillary.com"
MAPILLARY_IMAGE_ENDPOINT <- paste0(MAPILLARY_API_BASE, "/images")

# ============================================
# 2. 研究区域定义
# ============================================

# Inner London Boroughs
INNER_LONDON_BOROUGHS <- c(
  "Camden",
  "Greenwich", 
  "Hackney",
  "Hammersmith and Fulham",
  "Islington",
  "Kensington and Chelsea",
  "Lambeth",
  "Lewisham",
  "Southwark",
  "Tower Hamlets",
  "Wandsworth",
  "Westminster"
)

# ============================================
# 3. 采样参数
# ============================================

# LSOA采样
N_LSOAS_PER_QUINTILE <- 4
TOTAL_LSOAS <- 20

# 图像采样（更新为实际收集情况）
TARGET_IMAGES_PER_LSOA <- 100  # 实际每个LSOA收集了100张
MIN_IMAGES_PER_LSOA <- 60
MAX_IMAGES_PER_REQUEST <- 500  # 实际使用的请求限制

# 图像质量参数
MIN_IMAGE_WIDTH <- 1024   # 实际使用的最小宽度
MIN_IMAGE_HEIGHT <- 768   # 实际使用的最小高度

# 时间范围（更新为实际数据范围）
DATE_START <- "2014-01-01"  # 实际数据最早到2014年
DATE_END <- "2025-12-31"    # 实际数据包含2025年
# 注：虽然设定了范围，但实际优先选择较新的图像

# ============================================
# 4. 空间参数
# ============================================

# 搜索参数
MIN_IMAGE_DISTANCE <- 20    # 图像间最小距离（米）
SEARCH_EXPANSION <- 0.2     # 搜索范围扩展比例（20%）

# 坐标系统
WGS84_EPSG <- 4326          # 经纬度（Mapillary使用）
BNG_EPSG <- 27700           # British National Grid（距离计算）

# ============================================
# 5. GVI计算参数
# ============================================

# 语义分割模型
SEGMENTATION_MODEL <- "deeplab_v3"  # 可选：deeplab_v3, pspnet, segformer

# 植被类别定义（根据模型调整）
VEGETATION_CLASSES <- c(
  "tree",
  "grass", 
  "plant",
  "vegetation",
  "bush",
  "shrub"
)

# GVI计算设置
GVI_THRESHOLD <- 0.01  # 最小GVI值（1%）
BATCH_SIZE <- 32       # 批处理大小

# ============================================
# 6. 文件路径
# ============================================

# 使用相对路径
PATHS <- list(
  # 原始数据
  imd_data = "data/raw/IMD 2019/IMD2019_London.xlsx",
  lsoa_boundaries_dir = "data/raw/LB_LSOA2021_shp/LB_shp/",
  
  # 处理后数据
  selected_lsoas = "data/processed/selected_lsoas.csv",
  streetview_metadata = "data/processed/streetview_metadata_final.csv",  # 最终元数据
  download_summary = "data/processed/download_summary.csv",
  gvi_results = "data/processed/gvi_results.csv",
  
  # 图像存储
  mapillary_images = "data/raw/mapillary_images/",
  
  # 输出目录
  gvi_outputs = "output/gvi_results/",
  figures = "figures/",
  reports = "reports/",
  logs = "logs/"
)

# ============================================
# 7. 分析参数
# ============================================

# 统计分析
SIGNIFICANCE_LEVEL <- 0.05
CONFIDENCE_LEVEL <- 0.95

# 空间分析
SPATIAL_LAG_DISTANCE <- 500  # 米
MORAN_I_PERMUTATIONS <- 999

# ============================================
# 8. 可视化参数
# ============================================

# 颜色方案
COLOR_SCHEMES <- list(
  imd_quintiles = c("#2166ac", "#67a9cf", "#f7f7f7", "#fddbc7", "#b2182b"),
  gvi_gradient = c("#d73027", "#fc8d59", "#fee08b", "#d9ef8b", "#91cf60", "#1a9850"),
  borough = "Set3"
)

# 地图设置
MAP_SETTINGS <- list(
  width = 10,
  height = 8,
  dpi = 300,
  format = "png"
)

# ============================================
# 9. 系统设置
# ============================================

# 并行处理
USE_PARALLEL <- TRUE
N_CORES <- parallel::detectCores() - 1  # 保留一个核心

# 进度条
SHOW_PROGRESS <- TRUE

# 日志级别
LOG_LEVEL <- "INFO"  # DEBUG, INFO, WARNING, ERROR

# API速率限制
API_DELAY <- 0.1  # 请求间延迟（秒）

# ============================================
# 10. 项目统计（实际完成情况）
# ============================================

PROJECT_STATS <- list(
  total_lsoas = 20,
  images_per_lsoa = 100,
  total_images = 2000,
  collection_date = "2025-08-01",
  boroughs_covered = 12
)

# ============================================
# 11. 辅助函数
# ============================================

# 检查配置完整性
check_config <- function() {
  issues <- c()
  
  # 检查必要的目录
  required_dirs <- c("data", "data/raw", "data/processed", "output", "figures", "reports", "logs")
  for(dir in required_dirs) {
    if(!dir.exists(here::here(dir))) {
      dir.create(here::here(dir), recursive = TRUE, showWarnings = FALSE)
    }
  }
  
  # 检查关键文件
  if(!file.exists(here::here(PATHS$selected_lsoas))) {
    issues <- c(issues, "未找到selected_lsoas.csv文件")
  }
  
  if(!file.exists(here::here(PATHS$streetview_metadata))) {
    issues <- c(issues, "未找到streetview_metadata_final.csv文件")
  }
  
  # 检查图像文件夹
  if(!dir.exists(here::here(PATHS$mapillary_images))) {
    issues <- c(issues, "未找到mapillary_images文件夹")
  }
  
  if(length(issues) > 0) {
    cat("配置检查发现以下问题：\n")
    for(issue in issues) {
      cat("  -", issue, "\n")
    }
    return(FALSE)
  } else {
    cat("配置检查通过！\n")
    return(TRUE)
  }
}

# 获取配置摘要
get_config_summary <- function() {
  cat("=== City Look 项目配置摘要 ===\n")
  cat("研究区域：Inner London (", length(INNER_LONDON_BOROUGHS), "个boroughs)\n")
  cat("LSOA样本：", TOTAL_LSOAS, "个 (每个IMD五分位", N_LSOAS_PER_QUINTILE, "个)\n")
  cat("目标图像：每个LSOA", TARGET_IMAGES_PER_LSOA, "张\n")
  cat("时间范围：", DATE_START, "至", DATE_END, "\n")
  cat("=============================\n")
}

# 运行配置检查
if(interactive()) {
  check_config()
  get_config_summary()
}