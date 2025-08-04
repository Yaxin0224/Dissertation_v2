# 诊断 LSOA 代码匹配问题
library(tidyverse)
library(sf)
library(here)

# 1. 读取选中的 LSOAs
selected_lsoas <- read_csv(here("data", "processed", "selected_lsoas.csv"), 
                           show_col_types = FALSE)

cat("=== Selected LSOAs 信息 ===\n")
cat("总数:", nrow(selected_lsoas), "\n")
cat("列名:\n")
print(names(selected_lsoas))

# 查找包含 lsoa 的列
lsoa_cols <- names(selected_lsoas)[grep("lsoa", names(selected_lsoas), ignore.case = TRUE)]
cat("\n包含 'lsoa' 的列:\n")
print(lsoa_cols)

# 显示不同 LSOA 代码列的内容
for (col in lsoa_cols) {
  if (col %in% names(selected_lsoas)) {
    cat("\n", col, "示例:\n")
    print(head(selected_lsoas[[col]], 3))
  }
}

# 2. 检查一个 borough 的 shapefile
test_borough <- unique(selected_lsoas$borough)[1]
cat("\n\n=== 检查", test_borough, "的 shapefile ===\n")

shp_path <- here("data", "raw", "LB_LSOA2021_shp", "LB_shp", 
                 paste0(test_borough, ".shp"))

if (file.exists(shp_path)) {
  borough_shp <- st_read(shp_path, quiet = TRUE)
  
  cat("\nShapefile 列名:\n")
  print(names(borough_shp))
  
  # 查找 LSOA 相关列
  shp_lsoa_cols <- names(borough_shp)[grep("lsoa", names(borough_shp), ignore.case = TRUE)]
  cat("\n包含 'lsoa' 的列:\n")
  print(shp_lsoa_cols)
  
  # 显示 LSOA 代码示例
  for (col in shp_lsoa_cols) {
    if (col %in% names(borough_shp)) {
      cat("\n", col, "示例:\n")
      print(head(borough_shp[[col]], 3))
    }
  }
  
  # 3. 尝试匹配
  cat("\n\n=== 尝试匹配 LSOA 代码 ===\n")
  
  # 获取这个 borough 的选中 LSOAs
  borough_selected <- selected_lsoas %>% 
    filter(borough == test_borough)
  
  cat("\n", test_borough, "中选中的 LSOA 数量:", nrow(borough_selected), "\n")
  
  # 尝试用不同的列匹配
  if ("lsoa_code" %in% names(borough_selected)) {
    test_codes <- borough_selected$lsoa_code
    cat("\n使用 lsoa_code 列:\n")
    print(head(test_codes, 3))
    
    for (col in shp_lsoa_cols) {
      matches <- sum(test_codes %in% borough_shp[[col]])
      cat("- 匹配", col, ":", matches, "/", length(test_codes), "\n")
    }
  }
  
  if ("lsoa11cd" %in% names(borough_selected)) {
    test_codes <- borough_selected$lsoa11cd
    cat("\n使用 lsoa11cd 列:\n")
    print(head(test_codes, 3))
    
    for (col in shp_lsoa_cols) {
      matches <- sum(test_codes %in% borough_shp[[col]])
      cat("- 匹配", col, ":", matches, "/", length(test_codes), "\n")
    }
  }
  
  # 4. 显示 shapefile 中的所有 LSOA 代码（前10个）
  if (length(shp_lsoa_cols) > 0) {
    main_col <- shp_lsoa_cols[1]
    cat("\n\nShapefile 中的 LSOA 代码（", main_col, "）前10个:\n")
    print(head(sort(borough_shp[[main_col]]), 10))
  }
}