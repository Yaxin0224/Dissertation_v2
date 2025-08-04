#!/usr/bin/env Rscript
# ============================================
# City Look Dissertation v2
# 03_gvi_calculation.R - 绿视率(GVI)计算
# 
# 使用DeepLabv3+深度学习模型进行语义分割
# 计算街景图像中的植被比例
# 
# 作者：[您的名字]
# 最后更新：2025-01-14
# ============================================

# 清理环境
rm(list = ls())
gc()

# ============================================
# 1. 加载必要的包
# ============================================

# R包
library(tidyverse)
library(here)
library(reticulate)  # Python接口
library(magick)      # 图像处理
library(logger)      # 日志记录
library(tictoc)      # 计时
library(progress)    # 进度条
library(reticulate)
use_python("/opt/anaconda3/envs/pytorch/bin/python", required = TRUE)


# 设置项目路径
setwd(here())

# 配置日志
log_appender(appender_file(here("logs", paste0("gvi_calculation_", 
                                               format(Sys.Date(), "%Y%m%d"), ".log"))))
log_threshold(INFO)

# ============================================
# 2. Python环境设置
# ============================================

cat("设置Python环境...\n")

# 使用conda环境（根据您的系统调整）
# use_condaenv("pytorch", required = TRUE)
# 或使用虚拟环境
# use_virtualenv("~/venv/pytorch", required = TRUE)

# 导入Python模块
torch <- import("torch")
torchvision <- import("torchvision")
PIL <- import("PIL")
np <- import("numpy")
cv2 <- import("cv2")

# 检查CUDA可用性
cuda_available <- torch$cuda$is_available()
device <- ifelse(cuda_available, "cuda", "cpu")
cat(sprintf("使用设备: %s\n", device))
if(cuda_available) {
  cat(sprintf("CUDA设备: %s\n", torch$cuda$get_device_name(0L)))
}

# ============================================
# 3. 配置参数
# ============================================

# 测试参数
TEST_LSOA <- "E01000882"  # 替换为实际的LSOA代码
TEST_IMAGES <- 20         # 测试图像数量

# 模型参数
MODEL_NAME <- "deeplabv3_resnet101"
BATCH_SIZE <- ifelse(cuda_available, 8L, 2L)  # GPU时批量大小为8，CPU为2

# 植被类别（COCO/ADE20K数据集）
# DeepLabv3在COCO上的类别
COCO_VEGETATION_CLASSES <- c(
  "potted plant" = 63  # COCO中唯一的植物类
)

# 如果使用ADE20K预训练模型，植被类别更多
ADE20K_VEGETATION_CLASSES <- c(
  "tree" = 4,
  "grass" = 9,
  "plant" = 17,
  "flower" = 66,
  "bush" = 46
)

# 输出路径
OUTPUT_DIR <- here("output", "gvi_results")
dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

# ============================================
# 4. 加载DeepLabv3+模型
# ============================================

cat("\n加载DeepLabv3+模型...\n")

# 加载预训练模型
model <- torchvision$models$segmentation$deeplabv3_resnet101(
  pretrained = TRUE,
  progress = TRUE
)

# 设置为评估模式
model$eval()

# 移动到设备
if(cuda_available) {
  model <- model$cuda()
}

# 图像预处理
preprocess <- torchvision$transforms$Compose(list(
  torchvision$transforms$ToTensor(),
  torchvision$transforms$Normalize(
    mean = c(0.485, 0.456, 0.406),
    std = c(0.229, 0.224, 0.225)
  )
))

cat("模型加载完成！\n")

# ============================================
# 5. 辅助函数
# ============================================

#' 加载和预处理图像
#' @param image_path 图像文件路径
#' @return 预处理后的tensor
load_and_preprocess_image <- function(image_path) {
  # 使用PIL加载图像
  image <- PIL$Image$open(image_path)$convert("RGB")
  
  # 获取原始尺寸
  original_size <- c(image$size[[1]], image$size[[2]])  # width, height
  
  # 预处理
  input_tensor <- preprocess(image)
  
  # 添加batch维度
  input_batch <- input_tensor$unsqueeze(0L)
  
  return(list(
    tensor = input_batch,
    original_size = original_size,
    pil_image = image
  ))
}

#' 使用色彩阈值法作为备选方案
#' @param image_path 图像文件路径
#' @return GVI分数
calculate_gvi_color_threshold <- function(image_path) {
  # 读取图像
  img <- image_read(image_path)
  
  # 转换为数组
  img_array <- as.integer(image_data(img))
  
  # 获取RGB通道
  if(length(dim(img_array)) == 3) {
    R <- img_array[1,,]
    G <- img_array[2,,]
    B <- img_array[3,,]
  } else if(length(dim(img_array)) == 4) {
    R <- img_array[1,,,1]
    G <- img_array[2,,,1]
    B <- img_array[3,,,1]
  }
  
  # 植被检测条件（基于文献）
  # G > R 且 G > B
  vegetation_mask <- (G > R) & (G > B) & (G > 30)
  
  # 计算GVI
  total_pixels <- length(vegetation_mask)
  vegetation_pixels <- sum(vegetation_mask)
  gvi <- vegetation_pixels / total_pixels * 100
  
  return(list(
    gvi = gvi,
    vegetation_pixels = vegetation_pixels,
    total_pixels = total_pixels
  ))
}

#' 处理单张图像的GVI
#' @param image_path 图像路径
#' @param save_visualization 是否保存可视化结果
#' @return GVI结果
process_single_image <- function(image_path, save_visualization = FALSE) {
  
  tryCatch({
    # 加载图像
    img_data <- load_and_preprocess_image(image_path)
    
    # 移动到设备
    if(cuda_available) {
      input_batch <- img_data$tensor$cuda()
    } else {
      input_batch <- img_data$tensor
    }
    
    # 推理（无梯度计算）
    with(torch$no_grad(), {
      output <- model(input_batch)$'out'$squeeze(0L)
    })
    
    # 获取预测类别
    predictions <- output$argmax(0L)
    
    # 移回CPU并转换为numpy
    predictions_np <- predictions$cpu()$numpy()
    
    # 计算植被像素（注意：COCO数据集植被类别有限）
    # 这里我们使用简化的方法，可能需要切换到ADE20K模型
    vegetation_mask <- predictions_np == 63L  # potted plant类
    
    # 如果检测不到植被，尝试色彩阈值法
    vegetation_pixels <- sum(vegetation_mask)
    
    if(vegetation_pixels < 100) {  # 植被像素太少，可能是模型限制
      # 使用色彩阈值法作为补充
      color_result <- calculate_gvi_color_threshold(image_path)
      
      result <- list(
        image_path = image_path,
        method = "color_threshold",
        gvi = color_result$gvi,
        vegetation_pixels = color_result$vegetation_pixels,
        total_pixels = color_result$total_pixels,
        processing_time = NA
      )
    } else {
      # 使用深度学习结果
      total_pixels <- length(predictions_np)
      gvi <- vegetation_pixels / total_pixels * 100
      
      result <- list(
        image_path = image_path,
        method = "deeplab",
        gvi = gvi,
        vegetation_pixels = vegetation_pixels,
        total_pixels = total_pixels,
        processing_time = NA
      )
    }
    
    # 保存可视化（如果需要）
    if(save_visualization) {
      save_segmentation_visualization(
        image_path, 
        predictions_np, 
        result$gvi,
        output_dir = OUTPUT_DIR
      )
    }
    
    return(result)
    
  }, error = function(e) {
    log_error(paste("处理图像失败:", image_path, "-", e$message))
    return(list(
      image_path = image_path,
      method = "error",
      gvi = NA,
      vegetation_pixels = NA,
      total_pixels = NA,
      processing_time = NA,
      error = e$message
    ))
  })
}

#' 保存分割可视化结果
save_segmentation_visualization <- function(image_path, predictions, gvi, output_dir) {
  # 创建可视化
  # 这里简化处理，实际可以做得更精美
  
  basename <- tools::file_path_sans_ext(basename(image_path))
  output_path <- file.path(output_dir, paste0(basename, "_segmentation.png"))
  
  # 保存信息文件
  info_path <- file.path(output_dir, paste0(basename, "_info.txt"))
  writeLines(c(
    paste("Image:", basename(image_path)),
    paste("GVI:", round(gvi, 2), "%"),
    paste("Method:", ifelse(sum(predictions == 63) > 100, "DeepLab", "Color threshold"))
  ), info_path)
}

# ============================================
# 6. 主处理流程
# ============================================

cat("\n开始GVI计算测试...\n")
cat(paste("测试LSOA:", TEST_LSOA, "\n"))

# 获取LSOA图像目录
lsoa_dir <- here("data", "raw", "mapillary_images", TEST_LSOA)

if(!dir.exists(lsoa_dir)) {
  stop(paste("LSOA目录不存在:", lsoa_dir))
}

# 获取图像列表
image_files <- list.files(lsoa_dir, pattern = "\\.jpg$", full.names = TRUE)
cat(paste("找到", length(image_files), "张图像\n"))

# 选择测试图像
if(length(image_files) > TEST_IMAGES) {
  # 随机选择
  set.seed(123)
  test_images <- sample(image_files, TEST_IMAGES)
} else {
  test_images <- image_files
}

cat(paste("\n处理", length(test_images), "张测试图像...\n"))

# 创建进度条
pb <- progress_bar$new(
  format = "处理中 [:bar] :percent :elapsed",
  total = length(test_images),
  clear = FALSE
)

# 处理图像
results <- list()
tic()

for(i in seq_along(test_images)) {
  pb$tick()
  
  # 处理单张图像
  result <- process_single_image(
    test_images[i], 
    save_visualization = (i <= 5)  # 只保存前5张的可视化
  )
  
  results[[i]] <- result
}

processing_time <- toc()

# ============================================
# 7. 结果汇总和分析
# ============================================

cat("\n\n汇总结果...\n")

# 转换为数据框
results_df <- bind_rows(results) %>%
  mutate(
    lsoa_code = TEST_LSOA,
    image_id = basename(image_path) %>% str_remove("\\.jpg$")
  )

# 基本统计
summary_stats <- results_df %>%
  filter(!is.na(gvi)) %>%
  summarise(
    n_images = n(),
    mean_gvi = mean(gvi, na.rm = TRUE),
    median_gvi = median(gvi, na.rm = TRUE),
    sd_gvi = sd(gvi, na.rm = TRUE),
    min_gvi = min(gvi, na.rm = TRUE),
    max_gvi = max(gvi, na.rm = TRUE),
    deeplab_count = sum(method == "deeplab", na.rm = TRUE),
    color_threshold_count = sum(method == "color_threshold", na.rm = TRUE),
    error_count = sum(method == "error", na.rm = TRUE)
  )

# 打印结果
cat("\n========== GVI计算结果 ==========\n")
cat(sprintf("LSOA: %s\n", TEST_LSOA))
cat(sprintf("处理图像数: %d\n", summary_stats$n_images))
cat(sprintf("平均GVI: %.2f%%\n", summary_stats$mean_gvi))
cat(sprintf("中位数GVI: %.2f%%\n", summary_stats$median_gvi))
cat(sprintf("标准差: %.2f\n", summary_stats$sd_gvi))
cat(sprintf("GVI范围: %.2f%% - %.2f%%\n", summary_stats$min_gvi, summary_stats$max_gvi))
cat(sprintf("\n方法使用情况:\n"))
cat(sprintf("  DeepLab: %d\n", summary_stats$deeplab_count))
cat(sprintf("  色彩阈值: %d\n", summary_stats$color_threshold_count))
cat(sprintf("  错误: %d\n", summary_stats$error_count))
cat("==================================\n")

# ============================================
# 8. 保存结果
# ============================================

# 保存详细结果
output_file <- file.path(OUTPUT_DIR, paste0("gvi_test_", TEST_LSOA, "_", 
                                            format(Sys.Date(), "%Y%m%d"), ".csv"))
write_csv(results_df, output_file)
cat(paste("\n结果已保存到:", output_file, "\n"))

# 保存汇总统计
summary_file <- file.path(OUTPUT_DIR, paste0("gvi_summary_", TEST_LSOA, "_", 
                                             format(Sys.Date(), "%Y%m%d"), ".csv"))
write_csv(summary_stats, summary_file)

# ============================================
# 9. 创建可视化
# ============================================

library(ggplot2)

# GVI分布直方图
p1 <- results_df %>%
  filter(!is.na(gvi)) %>%
  ggplot(aes(x = gvi)) +
  geom_histogram(bins = 20, fill = "forestgreen", alpha = 0.7) +
  geom_vline(aes(xintercept = mean(gvi)), color = "red", linetype = "dashed", size = 1) +
  labs(
    title = paste("GVI Distribution -", TEST_LSOA),
    subtitle = paste("Mean GVI:", round(summary_stats$mean_gvi, 2), "%"),
    x = "Green View Index (%)",
    y = "Count"
  ) +
  theme_minimal()

# 保存图表
ggsave(file.path(OUTPUT_DIR, paste0("gvi_distribution_", TEST_LSOA, ".png")),
       p1, width = 8, height = 6, dpi = 300)

# ============================================
# 10. 建议和下一步
# ============================================

cat("\n========== 测试完成 ==========\n")
cat("下一步建议:\n")
cat("1. 检查output/gvi_results/中的可视化结果\n")
cat("2. 验证GVI计算的准确性\n")
cat("3. 如果结果满意，可以扩展到所有20个LSOA\n")
cat("4. 考虑是否需要使用ADE20K预训练模型（更多植被类别）\n")

# 检查是否需要改进
if(summary_stats$color_threshold_count > summary_stats$deeplab_count) {
  cat("\n⚠️ 警告：大部分图像使用了色彩阈值法而非深度学习\n")
  cat("   可能需要使用包含更多植被类别的模型\n")
}

cat("\n完成时间:", format(Sys.time()), "\n")

# ============================================
# 附录：批量处理函数（供后续使用）
# ============================================

#' 批量处理所有LSOA
#' @param lsoa_codes LSOA代码向量
#' @param images_per_lsoa 每个LSOA处理的图像数量
#' @param parallel 是否使用并行处理
process_all_lsoas <- function(lsoa_codes, images_per_lsoa = NULL, parallel = TRUE) {
  # 这个函数可以在测试成功后实现
  # 包含并行处理、批量推理、进度监控等功能
  cat("批量处理函数待实现...\n")
}

# ============================================
# END OF SCRIPT
# ============================================