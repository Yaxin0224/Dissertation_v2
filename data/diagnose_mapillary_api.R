# 详细检查 Mapillary API 响应结构
library(tidyverse)
library(httr)
library(jsonlite)
library(here)

# Mapillary 配置
MAPILLARY_TOKEN <- "MLY|9922859457805691|cef02444f32c339cf09761b104ca4bb5"
MAPILLARY_BASE_URL <- "https://graph.mapillary.com"

# 简单的边界框（伦敦市中心）
test_bbox <- "-0.1276,51.5007,-0.1236,51.5047"

cat("=== 测试 Mapillary API 响应结构 ===\n\n")

# 1. 基础请求
cat("1. 基础请求（无字段限制）\n")
params1 <- list(
  access_token = MAPILLARY_TOKEN,
  bbox = test_bbox,
  limit = 2
)

response1 <- GET(paste0(MAPILLARY_BASE_URL, "/images"), query = params1)
cat("状态码:", status_code(response1), "\n")

if (status_code(response1) == 200) {
  # 获取原始响应
  raw_content <- content(response1, "text")
  
  # 解析JSON
  data1 <- fromJSON(raw_content)
  
  cat("\n响应结构:\n")
  cat("- 顶层键:", names(data1), "\n")
  
  if ("data" %in% names(data1)) {
    cat("- data 的类型:", class(data1$data), "\n")
    cat("- data 的长度:", length(data1$data), "\n")
    
    if (length(data1$data) > 0) {
      cat("\n第一个元素的结构:\n")
      str(data1$data[1])
      
      cat("\n可用的字段:\n")
      if (is.data.frame(data1$data)) {
        print(names(data1$data))
      } else if (is.list(data1$data)) {
        print(names(data1$data[[1]]))
      }
    }
  }
}

# 2. 带字段参数的请求
cat("\n\n2. 带字段参数的请求\n")
params2 <- list(
  access_token = MAPILLARY_TOKEN,
  bbox = test_bbox,
  fields = "id,captured_at,geometry,camera_type,width,height",
  limit = 2
)

response2 <- GET(paste0(MAPILLARY_BASE_URL, "/images"), query = params2)
cat("状态码:", status_code(response2), "\n")

if (status_code(response2) == 200) {
  data2 <- fromJSON(content(response2, "text"))
  
  if ("data" %in% names(data2) && length(data2$data) > 0) {
    cat("\n数据结构:\n")
    str(data2$data)
    
    # 检查 geometry 字段
    if ("geometry" %in% names(data2$data)) {
      cat("\nGeometry 字段的结构:\n")
      print(class(data2$data$geometry))
      if (length(data2$data$geometry) > 0) {
        cat("第一个 geometry:\n")
        print(data2$data$geometry[1])
      }
    }
  }
}

# 3. 测试不同的解析方法
cat("\n\n3. 尝试不同的解析方法\n")
if (status_code(response2) == 200) {
  # 方法1：使用 simplifyVector = FALSE
  data3 <- fromJSON(content(response2, "text"), simplifyVector = FALSE)
  cat("使用 simplifyVector = FALSE:\n")
  cat("- 数据类型:", class(data3$data), "\n")
  if (length(data3$data) > 0) {
    cat("- 第一个元素的键:", names(data3$data[[1]]), "\n")
    if ("geometry" %in% names(data3$data[[1]])) {
      cat("- Geometry 结构:", names(data3$data[[1]]$geometry), "\n")
    }
  }
}

# 4. 保存一个响应样本供分析
cat("\n\n4. 保存响应样本\n")
if (exists("raw_content")) {
  sample_file <- here("data", "processed", "mapillary_api_sample.json")
  writeLines(raw_content, sample_file)
  cat("响应样本已保存到:", sample_file, "\n")
}