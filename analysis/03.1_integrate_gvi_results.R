#!/usr/bin/env Rscript
# ============================================
# City Look Dissertation v2
# 03.2_gvi_analysis_simple.R - 简化版GVI分析
# 
# 不依赖外部RData文件，直接分析Python生成的结果
# ============================================

# 清理环境
rm(list = ls())

# 加载包
library(tidyverse)
library(here)
library(ggplot2)
library(viridis)

# 设置项目路径
setwd(here())

# ============================================
# 1. 读取数据
# ============================================

cat("1. 读取数据...\n")

# 读取Python生成的GVI结果
gvi_results <- read_csv(here("output", "gvi_results", "gvi_results_for_r.csv"))
cat(paste("  - 读取", nrow(gvi_results), "条GVI记录\n"))

# 读取LSOA信息
selected_lsoas <- read_csv(here("data", "processed", "selected_lsoas.csv"))
cat(paste("  - 读取", nrow(selected_lsoas), "个LSOA信息\n"))

# ============================================
# 2. 数据汇总
# ============================================

cat("\n2. 计算LSOA级别统计...\n")

# LSOA级别汇总
lsoa_gvi_summary <- gvi_results %>%
  group_by(lsoa_code) %>%
  summarise(
    n_images = n(),
    mean_gvi = mean(gvi, na.rm = TRUE),
    median_gvi = median(gvi, na.rm = TRUE),
    sd_gvi = sd(gvi, na.rm = TRUE),
    min_gvi = min(gvi, na.rm = TRUE),
    max_gvi = max(gvi, na.rm = TRUE),
    q25_gvi = quantile(gvi, 0.25, na.rm = TRUE),
    q75_gvi = quantile(gvi, 0.75, na.rm = TRUE),
    # 方法使用统计
    n_deeplearning = sum(method == "deeplearning"),
    n_color = sum(method == "color_threshold"),
    pct_deeplearning = round(n_deeplearning / n() * 100, 1),
    .groups = 'drop'
  )

# 与LSOA信息合并
lsoa_gvi_complete <- lsoa_gvi_summary %>%
  left_join(selected_lsoas, by = "lsoa_code") %>%
  arrange(inner_imd_quintile, lsoa_code)

# 保存汇总结果
write_csv(lsoa_gvi_complete, here("output", "lsoa_gvi_summary.csv"))
cat("  - LSOA汇总结果已保存\n")

# ============================================
# 3. 描述性统计
# ============================================

cat("\n3. 总体统计:\n")

# 总体统计
cat(paste("  - 处理LSOA数:", n_distinct(gvi_results$lsoa_code), "\n"))
cat(paste("  - 处理图像数:", nrow(gvi_results), "\n"))
cat(paste("  - 平均GVI:", round(mean(gvi_results$gvi), 2), "%\n"))
cat(paste("  - 中位数GVI:", round(median(gvi_results$gvi), 2), "%\n"))
cat(paste("  - 标准差:", round(sd(gvi_results$gvi), 2), "\n"))
cat(paste("  - GVI范围:", round(min(gvi_results$gvi), 2), "-", 
          round(max(gvi_results$gvi), 2), "%\n"))

# 方法使用统计
method_stats <- gvi_results %>%
  count(method) %>%
  mutate(percentage = round(n / sum(n) * 100, 1))

cat("\n  方法使用情况:\n")
print(method_stats)

# IMD五分位统计
cat("\n4. IMD五分位统计:\n")
imd_stats <- lsoa_gvi_complete %>%
  group_by(inner_imd_quintile) %>%
  summarise(
    n_lsoas = n(),
    mean_gvi = round(mean(mean_gvi), 2),
    sd_gvi = round(sd(mean_gvi), 2),
    median_gvi = round(median(median_gvi), 2),
    .groups = 'drop'
  ) %>%
  mutate(
    quintile_label = paste0("Q", inner_imd_quintile, 
                            ifelse(inner_imd_quintile == 1, " (最不贫困)", 
                                   ifelse(inner_imd_quintile == 5, " (最贫困)", "")))
  )

print(imd_stats)

# ============================================
# 4. 统计检验
# ============================================

cat("\n5. 统计检验:\n")

# 相关性检验
if("imd_score" %in% names(lsoa_gvi_complete)) {
  cor_test <- cor.test(lsoa_gvi_complete$mean_gvi, 
                       lsoa_gvi_complete$imd_score,
                       method = "pearson")
  
  cat(paste("  - Pearson相关系数: r =", round(cor_test$estimate, 3), 
            ", p =", format(cor_test$p.value, scientific = FALSE), "\n"))
}

# Kruskal-Wallis检验
kw_test <- kruskal.test(mean_gvi ~ as.factor(inner_imd_quintile), 
                        data = lsoa_gvi_complete)

cat(paste("  - Kruskal-Wallis检验: χ² =", round(kw_test$statistic, 2), 
          ", p =", format(kw_test$p.value, scientific = FALSE), "\n"))

# ============================================
# 5. 可视化
# ============================================

cat("\n6. 创建可视化...\n")

# 创建figures目录
if(!dir.exists(here("figures"))) {
  dir.create(here("figures"))
}

# 5.1 箱线图：GVI by IMD Quintile
p1 <- lsoa_gvi_complete %>%
  ggplot(aes(x = factor(inner_imd_quintile), y = mean_gvi)) +
  geom_boxplot(aes(fill = factor(inner_imd_quintile)), 
               alpha = 0.7, outlier.shape = 21) +
  geom_jitter(width = 0.2, alpha = 0.5, size = 3) +
  scale_fill_viridis_d(option = "viridis") +
  scale_x_discrete(labels = c("Q1\n最不贫困", "Q2", "Q3", "Q4", "Q5\n最贫困")) +
  labs(
    title = "Green View Index by IMD Quintile",
    subtitle = paste("Inner London,", nrow(lsoa_gvi_complete), "LSOAs"),
    x = "IMD Quintile",
    y = "Mean GVI (%)",
    caption = paste("Kruskal-Wallis test: p =", 
                    format(kw_test$p.value, digits = 3))
  ) +
  theme_minimal() +
  theme(
    legend.position = "none",
    plot.title = element_text(size = 16, face = "bold"),
    plot.subtitle = element_text(size = 12),
    axis.text = element_text(size = 10)
  )

ggsave(here("figures", "gvi_by_imd_quintile.png"), p1, 
       width = 10, height = 7, dpi = 300)
cat("  - 箱线图已保存\n")

# 5.2 条形图：每个LSOA的GVI
p2 <- lsoa_gvi_complete %>%
  arrange(mean_gvi) %>%
  mutate(lsoa_label = paste0(lsoa_code, " (Q", inner_imd_quintile, ")")) %>%
  ggplot(aes(x = reorder(lsoa_label, mean_gvi), y = mean_gvi)) +
  geom_col(aes(fill = factor(inner_imd_quintile)), alpha = 0.8) +
  geom_text(aes(label = round(mean_gvi, 1)), hjust = -0.2, size = 3) +
  scale_fill_viridis_d(option = "viridis", name = "IMD Quintile") +
  coord_flip() +
  labs(
    title = "Mean GVI by LSOA",
    subtitle = "Ordered by GVI value",
    x = "LSOA (IMD Quintile)",
    y = "Mean GVI (%)"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 16, face = "bold"),
    axis.text.y = element_text(size = 8)
  )

ggsave(here("figures", "gvi_by_lsoa_ranked.png"), p2, 
       width = 10, height = 10, dpi = 300)
cat("  - LSOA排名图已保存\n")

# 5.3 方法使用分布
p3 <- lsoa_gvi_complete %>%
  select(lsoa_code, inner_imd_quintile, n_deeplearning, n_color, pct_deeplearning) %>%
  ggplot(aes(x = reorder(lsoa_code, pct_deeplearning), y = pct_deeplearning)) +
  geom_col(aes(fill = factor(inner_imd_quintile)), alpha = 0.8) +
  geom_hline(yintercept = 95, linetype = "dashed", color = "red") +
  scale_fill_viridis_d(option = "viridis", name = "IMD Quintile") +
  coord_flip() +
  labs(
    title = "Deep Learning Method Usage by LSOA",
    subtitle = "Percentage of images processed with SegFormer",
    x = "LSOA",
    y = "Deep Learning Usage (%)",
    caption = "Red line indicates 95% threshold"
  ) +
  theme_minimal()

ggsave(here("figures", "method_usage_by_lsoa.png"), p3, 
       width = 10, height = 8, dpi = 300)
cat("  - 方法使用图已保存\n")

# 5.4 GVI分布直方图
p4 <- gvi_results %>%
  ggplot(aes(x = gvi)) +
  geom_histogram(bins = 50, fill = "forestgreen", alpha = 0.7) +
  geom_vline(aes(xintercept = mean(gvi)), 
             color = "red", linetype = "dashed", size = 1) +
  facet_wrap(~method, ncol = 1, scales = "free_y") +
  labs(
    title = "Distribution of GVI Values",
    subtitle = paste("All images (n =", nrow(gvi_results), ")"),
    x = "GVI (%)",
    y = "Count",
    caption = "Red line shows mean value"
  ) +
  theme_minimal()

ggsave(here("figures", "gvi_distribution_by_method.png"), p4, 
       width = 10, height = 8, dpi = 300)
cat("  - GVI分布图已保存\n")

# ============================================
# 6. 生成报告表格
# ============================================

cat("\n7. 生成汇总表格...\n")

# 创建汇总表
summary_table <- lsoa_gvi_complete %>%
  mutate(
    Borough = str_replace_all(borough, " and ", " & "),
    `IMD Q` = inner_imd_quintile,
    `Images` = n_images,
    `Mean GVI` = paste0(round(mean_gvi, 1), "%"),
    `SD` = round(sd_gvi, 1),
    `DL%` = paste0(pct_deeplearning, "%")
  ) %>%
  select(LSOA = lsoa_code, Borough, `IMD Q`, Images, `Mean GVI`, SD, `DL%`)

# 保存表格
write_csv(summary_table, here("output", "gvi_summary_table.csv"))
cat("  - 汇总表格已保存\n")

# 打印前10行
cat("\nLSOA GVI汇总（前10个）:\n")
print(summary_table %>% head(10))

# ============================================
# 7. 完成
# ============================================

cat("\n========== GVI分析完成 ==========\n")
cat("生成的文件:\n")
cat("  - output/lsoa_gvi_summary.csv - 详细汇总数据\n")
cat("  - output/gvi_summary_table.csv - 简洁汇总表\n")
cat("  - figures/gvi_*.png - 可视化图表\n")
cat("\n主要发现:\n")
cat(paste("  - 总体平均GVI:", round(mean(lsoa_gvi_complete$mean_gvi), 2), "%\n"))
cat(paste("  - 深度学习使用率:", 
          round(sum(gvi_results$method == "deeplearning") / nrow(gvi_results) * 100, 1), 
          "%\n"))
cat(paste("  - IMD五分位差异显著性: p =", 
          format(kw_test$p.value, digits = 3), "\n"))

# 问题LSOA提醒
problem_lsoas <- lsoa_gvi_complete %>%
  filter(pct_deeplearning < 90) %>%
  pull(lsoa_code)

if(length(problem_lsoas) > 0) {
  cat("\n注意：以下LSOA的深度学习使用率低于90%:\n")
  for(lsoa in problem_lsoas) {
    lsoa_info <- lsoa_gvi_complete %>% 
      filter(lsoa_code == lsoa) %>%
      slice(1)
    cat(paste("  -", lsoa, ":", lsoa_info$pct_deeplearning, "%\n"))
  }
}

cat("\n下一步建议:\n")
cat("  1. 检查figures/目录中的可视化结果\n")
cat("  2. 运行空间自相关分析\n")
cat("  3. 进行回归分析探索影响因素\n")

# 返回关键结果
invisible(list(
  lsoa_summary = lsoa_gvi_complete,
  imd_stats = imd_stats,
  kw_test = kw_test,
  method_stats = method_stats
))