# ============================================
# City Look Dissertation v2
# 01_lsoa_selection.R - Inner London LSOA选择
# 
# 目的：从Inner London选择20个LSOA进行分析
# 方法：基于IMD的分层随机抽样，确保空间分散性
# 输出：selected_lsoas.csv - 最终选中的20个LSOA
# ============================================

# 清理环境
rm(list = ls())

# 加载必要的包
library(tidyverse)
library(readxl)
library(sf)
library(tmap)

# 设置随机种子以确保可重现性
set.seed(20250114)

# ============================================
# 1. 配置参数
# ============================================

# Inner London boroughs定义
INNER_LONDON_BOROUGHS <- c(
  "Camden", "Greenwich", "Hackney", 
  "Hammersmith and Fulham", "Islington", 
  "Kensington and Chelsea", "Lambeth", 
  "Lewisham", "Southwark", "Tower Hamlets", 
  "Wandsworth", "Westminster"
)

# 采样参数
N_LSOAS_PER_QUINTILE <- 4
TOTAL_LSOAS <- 20

# 文件路径（使用相对路径）
IMD_FILE <- "data/raw/IMD 2019/IMD2019_London.xlsx"
LSOA_BOUNDARIES_DIR <- "data/raw/LB_LSOA2021_shp/LB_shp/"
OUTPUT_DIR <- "data/processed/"

# ============================================
# 2. 读取和清理IMD数据
# ============================================

cat("1. 读取IMD数据...\n")

# 读取IMD数据
imd_data <- read_excel(IMD_FILE, sheet = "IMD 2019")

# 重命名列
imd_clean <- imd_data %>%
  rename(
    lsoa_code = `LSOA code (2011)`,
    lsoa_name = `LSOA name (2011)`,
    la_code = `Local Authority District code (2019)`,
    borough = `Local Authority District name (2019)`,
    imd_score = `Index of Multiple Deprivation (IMD) Score`,
    imd_rank = `Index of Multiple Deprivation (IMD) Rank (where 1 is most deprived)`,
    imd_decile = `Index of Multiple Deprivation (IMD) Decile (where 1 is most deprived 10% of LSOAs)`
  )

# 检查Kensington的正确名称
kensington_variants <- unique(imd_clean$borough)[grep("Kensington", unique(imd_clean$borough))]
if(length(kensington_variants) > 0) {
  actual_kensington_name <- kensington_variants[1]
  INNER_LONDON_BOROUGHS[INNER_LONDON_BOROUGHS == "Kensington and Chelsea"] <- actual_kensington_name
  cat("  - Kensington的实际名称：", actual_kensington_name, "\n")
}

cat("  - IMD数据包含", nrow(imd_clean), "个LSOA\n")

# ============================================
# 3. 筛选Inner London LSOAs
# ============================================

cat("\n2. 筛选Inner London LSOAs...\n")

# 筛选Inner London（排除City of London）
inner_london_lsoas <- imd_clean %>%
  filter(
    borough %in% INNER_LONDON_BOROUGHS,
    borough != "City of London"
  )

cat("  - Inner London包含", nrow(inner_london_lsoas), "个LSOA\n")
cat("  - 涵盖", length(unique(inner_london_lsoas$borough)), "个borough\n")

# ============================================
# 4. 读取LSOA边界数据并验证
# ============================================

cat("\n3. 读取和验证LSOA边界数据...\n")

# 读取并合并所有Borough的LSOA边界
shp_files <- list.files(LSOA_BOUNDARIES_DIR, pattern = "\\.shp$", full.names = TRUE)

# 只读取Inner London的Borough文件
inner_london_patterns <- gsub(" ", ".", INNER_LONDON_BOROUGHS)
inner_london_patterns <- paste0("(", paste(inner_london_patterns, collapse = "|"), ")")
relevant_shp_files <- shp_files[grepl(inner_london_patterns, shp_files, ignore.case = TRUE)]

# 读取并合并
lsoa_boundaries <- NULL
for(shp_file in relevant_shp_files) {
  tryCatch({
    borough_data <- st_read(shp_file, quiet = TRUE)
    if(is.null(lsoa_boundaries)) {
      lsoa_boundaries <- borough_data
    } else {
      common_cols <- intersect(names(lsoa_boundaries), names(borough_data))
      lsoa_boundaries <- rbind(
        lsoa_boundaries[, common_cols],
        borough_data[, common_cols]
      )
    }
  }, error = function(e) {
    cat("    警告：无法读取", basename(shp_file), "\n")
  })
}

cat("  - 成功读取", nrow(lsoa_boundaries), "个LSOA边界\n")

# 获取边界文件中存在的LSOA代码
available_lsoa_codes <- unique(lsoa_boundaries$lsoa21cd)

# ============================================
# 5. 计算Inner London特定的IMD五分位
# ============================================

cat("\n4. 计算Inner London特定的IMD五分位...\n")

# 只保留在边界文件中存在的LSOA
inner_london_with_boundaries <- inner_london_lsoas %>%
  filter(lsoa_code %in% available_lsoa_codes)

cat("  - 有边界数据的LSOA数：", nrow(inner_london_with_boundaries), "\n")

# 计算五分位
inner_london_quintiles <- inner_london_with_boundaries %>%
  mutate(
    inner_imd_quintile = ntile(imd_score, 5),
    deprivation_category = case_when(
      inner_imd_quintile == 1 ~ "1_Least Deprived (Inner London)",
      inner_imd_quintile == 2 ~ "2_Less Deprived (Inner London)",
      inner_imd_quintile == 3 ~ "3_Average (Inner London)",
      inner_imd_quintile == 4 ~ "4_More Deprived (Inner London)",
      inner_imd_quintile == 5 ~ "5_Most Deprived (Inner London)"
    )
  )

# 显示五分位分布
quintile_summary <- inner_london_quintiles %>%
  group_by(inner_imd_quintile, deprivation_category) %>%
  summarise(
    n_lsoas = n(),
    min_imd = min(imd_score),
    max_imd = max(imd_score),
    mean_imd = mean(imd_score),
    .groups = 'drop'
  )

print(quintile_summary)

# ============================================
# 6. 分层随机抽样（确保空间分散性）
# ============================================

cat("\n5. 执行分层随机抽样...\n")

# 定义空间分散性抽样函数
stratified_spatial_sampling <- function(data, n_per_group = 4) {
  
  selected_lsoas <- data %>%
    group_by(inner_imd_quintile) %>%
    group_modify(~ {
      quintile_data <- .x
      n_boroughs <- length(unique(quintile_data$borough))
      
      # 策略1：如果borough数量充足，优先从不同borough选择
      if(n_boroughs >= n_per_group) {
        # 首先从每个borough随机选1个
        borough_sample <- quintile_data %>%
          group_by(borough) %>%
          sample_n(1) %>%
          ungroup()
        
        # 如果不足n_per_group个，从剩余的随机补充
        if(nrow(borough_sample) < n_per_group) {
          remaining <- quintile_data %>%
            anti_join(borough_sample, by = "lsoa_code") %>%
            sample_n(n_per_group - nrow(borough_sample))
          
          borough_sample <- bind_rows(borough_sample, remaining)
        } else {
          # 如果超过n_per_group个，随机选择n_per_group个
          borough_sample <- borough_sample %>%
            sample_n(n_per_group)
        }
        
        return(borough_sample)
        
      } else {
        # 策略2：如果borough数量不足，直接随机抽样
        # 但尽量确保来自不同borough
        selected <- data.frame()
        remaining_data <- quintile_data
        
        # 先从每个borough各选一个
        for(b in unique(quintile_data$borough)) {
          borough_lsoas <- remaining_data %>% filter(borough == b)
          if(nrow(borough_lsoas) > 0 && nrow(selected) < n_per_group) {
            selected <- bind_rows(selected, borough_lsoas %>% sample_n(1))
            remaining_data <- remaining_data %>% 
              filter(lsoa_code != selected$lsoa_code[nrow(selected)])
          }
        }
        
        # 如果还不够，从剩余的随机选
        if(nrow(selected) < n_per_group && nrow(remaining_data) > 0) {
          n_needed <- n_per_group - nrow(selected)
          additional <- remaining_data %>% sample_n(min(n_needed, nrow(remaining_data)))
          selected <- bind_rows(selected, additional)
        }
        
        return(selected)
      }
    }) %>%
    ungroup()
  
  return(selected_lsoas)
}

# 执行抽样
selected_lsoas <- stratified_spatial_sampling(inner_london_quintiles, N_LSOAS_PER_QUINTILE)

# ============================================
# 7. 验证抽样结果
# ============================================

cat("\n6. 验证抽样结果...\n")

# 检查每个五分位的样本数
quintile_check <- selected_lsoas %>%
  count(inner_imd_quintile, deprivation_category)
print(quintile_check)

# 检查borough分布
borough_distribution <- selected_lsoas %>%
  count(borough, inner_imd_quintile) %>%
  pivot_wider(names_from = inner_imd_quintile, values_from = n, values_fill = 0)

cat("\nBorough分布矩阵：\n")
print(borough_distribution)

# 统计覆盖的borough数量
n_boroughs_covered <- length(unique(selected_lsoas$borough))
cat("\n覆盖了", n_boroughs_covered, "个不同的borough\n")

# ============================================
# 8. 添加额外信息并保存
# ============================================

# 添加选择顺序和其他有用信息
selected_lsoas_final <- selected_lsoas %>%
  arrange(inner_imd_quintile, lsoa_code) %>%
  mutate(
    selection_id = row_number(),
    selection_label = paste0("LSOA_", str_pad(selection_id, 2, pad = "0"))
  ) %>%
  select(
    selection_id,
    selection_label,
    lsoa_code,
    lsoa_name,
    borough,
    inner_imd_quintile,
    deprivation_category,
    imd_score,
    imd_rank,
    everything()
  )

cat("\n7. 保存结果...\n")

# 创建输出目录（如果不存在）
if(!dir.exists(OUTPUT_DIR)) {
  dir.create(OUTPUT_DIR, recursive = TRUE)
}

# 保存为CSV（这是最终的文件）
output_file <- file.path(OUTPUT_DIR, "selected_lsoas.csv")
write_csv(selected_lsoas_final, output_file)
cat("  - 已保存到:", output_file, "\n")

# 保存详细报告
report_file <- file.path(OUTPUT_DIR, "lsoa_selection_report.txt")
sink(report_file)
cat("City Look Dissertation - LSOA Selection Report\n")
cat("生成时间:", Sys.time(), "\n")
cat("=====================================\n\n")

cat("1. 总体统计\n")
cat("   - Inner London LSOA总数:", nrow(inner_london_lsoas), "\n")
cat("   - 有边界数据的LSOA数:", nrow(inner_london_with_boundaries), "\n")
cat("   - 选中LSOA数:", nrow(selected_lsoas_final), "\n")
cat("   - 覆盖Borough数:", n_boroughs_covered, "\n\n")

cat("2. IMD五分位分布\n")
print(quintile_check)
cat("\n")

cat("3. Borough分布\n")
print(borough_distribution)
cat("\n")

cat("4. 选中的LSOA列表\n")
selected_lsoas_final %>%
  select(selection_id, lsoa_code, borough, deprivation_category, imd_score) %>%
  print(n = 20)

sink()
cat("  - 报告已保存到:", report_file, "\n")

# ============================================
# 9. 创建可视化
# ============================================

if(!is.null(lsoa_boundaries)) {
  cat("\n8. 创建地图可视化...\n")
  
  # 连接选中的LSOA与边界数据
  selected_boundaries <- lsoa_boundaries %>%
    inner_join(selected_lsoas_final, by = c("lsoa21cd" = "lsoa_code"))
  
  # 创建地图
  tmap_mode("plot")
  
  # 主题地图
  map_selected <- tm_shape(lsoa_boundaries %>% 
                             filter(lsoa21cd %in% inner_london_with_boundaries$lsoa_code)) +
    tm_polygons(col = "grey90", border.col = "grey80", lwd = 0.5) +
    tm_shape(selected_boundaries) +
    tm_polygons(
      col = "deprivation_category",
      palette = c("#2166ac", "#67a9cf", "#f7f7f7", "#fddbc7", "#b2182b"),
      title = "IMD Quintile",
      border.col = "black",
      lwd = 1.5
    ) +
    tm_layout(
      title = "Selected LSOAs in Inner London",
      frame = FALSE,
      legend.outside = TRUE
    )
  
  # 保存地图
  map_file <- file.path("figures", "selected_lsoas_map.png")
  if(!dir.exists("figures")) dir.create("figures")
  tmap_save(map_selected, map_file, width = 10, height = 8, dpi = 300)
  cat("  - 地图已保存到:", map_file, "\n")
  
  # 验证地图中的LSOA数量
  cat("  - 地图中显示的LSOA数：", nrow(selected_boundaries), "\n")
  
} else {
  cat("\n注意：LSOA边界数据未正确加载\n")
}

# ============================================
# 10. 完成
# ============================================

cat("\n========== LSOA选择完成 ==========\n")
cat("最终成功选择", nrow(selected_lsoas_final), "个LSOA\n")
cat("下一步：运行 02_streetview_download.R 下载街景图像\n")

# 显示最终统计
cat("\n最终选择的LSOA摘要：\n")
selected_lsoas_final %>%
  group_by(deprivation_category) %>%
  summarise(
    count = n(),
    boroughs = paste(unique(borough), collapse = ", "),
    .groups = 'drop'
  ) %>%
  print()