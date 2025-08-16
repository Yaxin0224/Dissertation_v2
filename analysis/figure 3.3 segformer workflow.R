#!/usr/bin/env Rscript

# =============================================================================
# A) Technical Pipeline for GVI Calculation - Clean Version for Thesis
# =============================================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(grid)
  library(here)
})

# 输出目录
FIG_DIR <- here("figures")
dir.create(FIG_DIR, recursive = TRUE, showWarnings = FALSE)

# 创建流程图
create_pipeline_flowchart <- function() {
  steps <- data.frame(
    x = c(1, 2, 3, 4, 5, 6),
    y = c(1, 1, 1, 1, 1, 1),
    step = c(
      "Input\nMapillary\nImages",
      "Preprocessing\n512×512px\nStandardization", 
      "SegFormer-b0\nSemantic\nSegmentation",
      "Vegetation\nPixel\nExtraction",
      "GVI\nCalculation\n(% green)",
      "LSOA-level\nWeighted\nAggregation"
    )
  )
  
  p_pipeline <- ggplot(steps, aes(x = x, y = y)) +
    # 方框（无填充）
    geom_rect(aes(xmin = x - 0.5, xmax = x + 0.5, 
                  ymin = y - 0.35, ymax = y + 0.35), 
              fill = NA, color = "black", size = 1) +
    # 文字（Times New Roman, 黑色）
    geom_text(aes(label = step), 
              family = "Times New Roman",
              color = "black", fontface = "plain", 
              size = 5, lineheight = 0.9) +
    # 箭头
    geom_segment(data = data.frame(
      x1 = seq(1.5, 5.5, by = 1), 
      x2 = seq(1.7, 5.7, by = 1), 
      y1 = 1, y2 = 1
    ), aes(x = x1, xend = x2, y = y1, yend = y2),
    arrow = arrow(length = unit(0.3, "cm"), type = "closed"),
    size = 0.8, color = "black") +
    # 主题设置
    xlim(0.3, 6.7) +
    ylim(0.4, 1.6) +
    theme_void() +
    labs(title = "A) Technical Pipeline for GVI Calculation") +
    theme(
      plot.title = element_text(
        family = "Times New Roman",
        hjust = 0.5, size = 18, face = "bold", 
        margin = margin(b = 20),
        color = "black"
      ),
      plot.margin = margin(20, 20, 20, 20)
    )
  
  return(p_pipeline)
}

# 保存图像
save_pipeline <- function() {
  p <- create_pipeline_flowchart()
  ggsave(
    filename = file.path(FIG_DIR, "fig_3_3_pipeline_only.png"),
    plot = p,
    width = 10, height = 3, dpi = 300, bg = "white"
  )
  ggsave(
    filename = file.path(FIG_DIR, "fig_3_3_pipeline_only.pdf"),
    plot = p,
    width = 10, height = 3, bg = "white"
  )
  cat("✓ 流程图已保存到：", FIG_DIR, "\n")
}

# 主运行
if (!interactive()) {
  save_pipeline()
}
