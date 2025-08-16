#!/usr/bin/env Rscript
# ============================================
# City Look Dissertation v2
# 05_cluster_interpretation.R - 聚类结果深入分析
# 
# 目的：解释k=2聚类的含义，为Phase 2提供方向
# ============================================

# 清理环境
rm(list = ls())

# 加载必要的包
library(tidyverse)
library(here)
library(viridis)
library(gridExtra)
library(ggrepel)  # 用于更好的标签

select <- dplyr::select
everything <- dplyr::everything


# 设置路径
setwd(here())

cat("============================================\n")
cat("聚类结果深入分析\n")
cat("============================================\n\n")

# ============================================
# 1. 读取数据和模型
# ============================================

cat("1. 加载数据和模型...\n")

# 读取增强的分析数据
analysis_data <- read_csv(here("output", "lsoa_analysis_enhanced.csv"),
                          show_col_types = FALSE)

# 加载模型对象
load(here("output", "phase1_models.RData"))

cat("  - 数据加载完成 ✓\n")

# ============================================
# 2. 聚类特征分析
# ============================================

cat("\n2. 分析两个聚类的特征...\n")

# 详细的聚类对比
cluster_detailed <- analysis_data %>%
  filter(!is.na(cluster)) %>%
  group_by(cluster) %>%
  summarise(
    n = n(),
    # GVI统计
    mean_gvi = mean(mean_gvi),
    median_gvi = median(median_gvi),
    sd_gvi = mean(sd_gvi),
    gvi_range = paste0(round(min(mean_gvi), 1), "-", round(max(mean_gvi), 1)),
    
    # IMD统计
    mean_imd = mean(imd_score),
    imd_range = paste0(round(min(imd_score), 1), "-", round(max(imd_score), 1)),
    
    # Borough分布
    n_boroughs = n_distinct(borough),
    main_boroughs = paste(names(sort(table(borough), decreasing = TRUE)[1:3]), 
                          collapse = ", "),
    
    # 方法使用
    mean_dl_usage = mean(pct_deeplearning),
    
    .groups = 'drop'
  ) %>%
  mutate(across(where(is.numeric), ~round(., 2)))

cat("\n聚类详细对比:\n")
print(cluster_detailed)

# 为聚类命名
cluster_names <- analysis_data %>%
  filter(!is.na(cluster)) %>%
  group_by(cluster) %>%
  summarise(mean_gvi = mean(mean_gvi)) %>%
  arrange(desc(mean_gvi)) %>%
  mutate(cluster_name = c("High GVI Cluster", "Low GVI Cluster"))

# ============================================
# 3. 聚类可视化增强
# ============================================

cat("\n3. 创建增强的聚类可视化...\n")

# 合并聚类名称
analysis_data <- analysis_data %>%
  left_join(cluster_names %>% select(cluster, cluster_name), by = "cluster")

# 3.1 聚类散点图（增强版）
p_cluster_enhanced <- ggplot(analysis_data %>% filter(!is.na(cluster)), 
                             aes(x = imd_score, y = mean_gvi)) +
  geom_point(aes(color = cluster_name, size = sd_gvi), alpha = 0.7) +
  geom_text_repel(aes(label = lsoa_code), size = 2.5, max.overlaps = 15) +
  stat_ellipse(aes(color = cluster_name), level = 0.95, size = 1.2) +
  scale_color_manual(values = c("High GVI Cluster" = "#2E8B57", 
                                "Low GVI Cluster" = "#DC143C"),
                     name = "Cluster Type") +
  scale_size_continuous(name = "GVI Variability\n(SD)", range = c(3, 8)) +
  labs(
    title = "LSOA Clustering: Two Distinct Green Visibility Patterns",
    subtitle = "Ellipses show 95% confidence regions for each cluster",
    x = "IMD Score (Higher = More Deprived)",
    y = "Mean Green View Index (%)"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    legend.position = "right"
  )

ggsave(here("figures", "cluster_enhanced_scatter.png"), p_cluster_enhanced,
       width = 12, height = 8, dpi = 300)

# 3.2 Borough分布在聚类中
borough_cluster <- analysis_data %>%
  filter(!is.na(cluster)) %>%
  count(borough, cluster_name) %>%
  group_by(borough) %>%
  mutate(pct = n / sum(n) * 100)

p_borough_cluster <- ggplot(borough_cluster, 
                            aes(x = reorder(borough, pct), y = pct, 
                                fill = cluster_name)) +
  geom_col(position = "stack") +
  coord_flip() +
  scale_fill_manual(values = c("High GVI Cluster" = "#2E8B57", 
                               "Low GVI Cluster" = "#DC143C"),
                    name = "Cluster Type") +
  labs(
    title = "Borough Representation in Each Cluster",
    x = "Borough",
    y = "Percentage of LSOAs"
  ) +
  theme_minimal()

ggsave(here("figures", "borough_cluster_distribution.png"), p_borough_cluster,
       width = 10, height = 8, dpi = 300)

# 3.3 聚类特征雷达图准备
cat("\n  准备聚类特征对比...\n")

# 标准化用于雷达图的变量
radar_data <- analysis_data %>%
  filter(!is.na(cluster)) %>%
  group_by(cluster_name) %>%
  summarise(
    `Green Visibility` = mean(mean_gvi),
    `Deprivation` = mean(imd_score),
    `GVI Variability` = mean(sd_gvi),
    `DL Method Use` = mean(pct_deeplearning),
    `Sample Size` = mean(n_images, na.rm = TRUE),
    .groups = 'drop'
  ) %>%
  pivot_longer(-cluster_name, names_to = "variable", values_to = "value") %>%
  group_by(variable) %>%
  mutate(value_scaled = (value - min(value)) / (max(value) - min(value)) * 100)

# ============================================
# 4. 异常值和边界案例分析
# ============================================

cat("\n4. 分析异常值和边界案例...\n")

# 识别每个聚类中的极端案例
extreme_cases <- analysis_data %>%
  filter(!is.na(cluster)) %>%
  group_by(cluster) %>%
  mutate(
    gvi_zscore = (mean_gvi - mean(mean_gvi)) / sd(mean_gvi),
    is_extreme = abs(gvi_zscore) > 1.5
  ) %>%
  filter(is_extreme) %>%
  select(lsoa_code, borough, cluster_name, mean_gvi, imd_score, gvi_zscore)

cat("\n极端案例（z-score > 1.5）:\n")
print(as.data.frame(extreme_cases))

# 特别关注异常高GVI的LSOA
high_gvi_lsoa <- analysis_data %>%
  filter(lsoa_code == "E01003134")

cat("\n\n异常高GVI LSOA详情 (E01003134 - Lambeth):\n")
cat("  - Mean GVI:", round(high_gvi_lsoa$mean_gvi, 1), "%\n")
cat("  - IMD Score:", round(high_gvi_lsoa$imd_score, 1), "\n")
cat("  - IMD Quintile:", high_gvi_lsoa$inner_imd_quintile, "\n")
cat("  - 所属聚类:", high_gvi_lsoa$cluster_name, "\n")

# ============================================
# 5. 聚类转换分析
# ============================================

cat("\n5. 分析聚类之间的'边界'LSOA...\n")

# 计算每个LSOA到两个聚类中心的距离
cluster_centers <- analysis_data %>%
  filter(!is.na(cluster)) %>%
  group_by(cluster) %>%
  summarise(
    center_gvi = mean(mean_gvi),
    center_imd = mean(imd_score),
    .groups = 'drop'
  )

# 计算距离
analysis_data <- analysis_data %>%
  mutate(
    dist_to_cluster1 = sqrt((mean_gvi - cluster_centers$center_gvi[1])^2 + 
                              (imd_score - cluster_centers$center_imd[1])^2),
    dist_to_cluster2 = sqrt((mean_gvi - cluster_centers$center_gvi[2])^2 + 
                              (imd_score - cluster_centers$center_imd[2])^2),
    dist_ratio = pmin(dist_to_cluster1, dist_to_cluster2) / 
      pmax(dist_to_cluster1, dist_to_cluster2)
  )

# 识别边界案例（距离比接近1的）
boundary_cases <- analysis_data %>%
  filter(!is.na(cluster), dist_ratio > 0.8) %>%
  select(lsoa_code, borough, cluster_name, mean_gvi, imd_score, dist_ratio) %>%
  arrange(desc(dist_ratio))

cat("\n边界案例（可能转换聚类的LSOA）:\n")
print(as.data.frame(boundary_cases))

# ============================================
# 6. 政策启示分析
# ============================================

cat("\n\n6. 基于聚类的政策启示...\n")

# 为每个聚类生成政策建议
policy_implications <- list(
  high_gvi = analysis_data %>%
    filter(cluster_name == "High GVI Cluster", !is.na(cluster_name)) %>%
    summarise(
      n = n(),
      mean_gvi = mean(mean_gvi),
      mean_imd = mean(imd_score),
      boroughs = paste(unique(borough), collapse = ", ")
    ),
  
  low_gvi = analysis_data %>%
    filter(cluster_name == "Low GVI Cluster", !is.na(cluster_name)) %>%
    summarise(
      n = n(),
      mean_gvi = mean(mean_gvi),
      mean_imd = mean(imd_score),
      boroughs = paste(unique(borough), collapse = ", ")
    )
)

# ============================================
# 7. 生成聚类分析报告
# ============================================

cat("\n7. 生成聚类分析报告...\n")

report <- paste0(
  "City Look Dissertation - 聚类深入分析报告\n",
  "生成时间: ", Sys.time(), "\n",
  "=====================================\n\n",
  
  "1. 聚类概述\n",
  "------------\n",
  "通过K-means聚类（k=2）识别出两个不同的LSOA类型：\n\n",
  
  "高GVI聚类（n=", policy_implications$high_gvi$n, "）:\n",
  "  - 平均GVI: ", round(policy_implications$high_gvi$mean_gvi, 1), "%\n",
  "  - 平均IMD: ", round(policy_implications$high_gvi$mean_imd, 1), "\n",
  "  - 主要Borough: ", policy_implications$high_gvi$boroughs, "\n\n",
  
  "低GVI聚类（n=", policy_implications$low_gvi$n, "）:\n",
  "  - 平均GVI: ", round(policy_implications$low_gvi$mean_gvi, 1), "%\n",
  "  - 平均IMD: ", round(policy_implications$low_gvi$mean_imd, 1), "\n",
  "  - 主要Borough: ", policy_implications$low_gvi$boroughs, "\n\n",
  
  "2. 关键发现\n",
  "------------\n",
  "- 两个聚类的GVI差异（", 
  round(policy_implications$high_gvi$mean_gvi - policy_implications$low_gvi$mean_gvi, 1),
  "%）远大于IMD差异（",
  round(abs(policy_implications$high_gvi$mean_imd - policy_implications$low_gvi$mean_imd), 1),
  "分）\n",
  "- 这表明绿化水平主要由其他因素决定，而非贫困程度\n",
  "- Borough层面的聚集效应明显\n\n",
  
  "3. 异常案例\n",
  "------------\n",
  "E01003134 (Lambeth): GVI=46.6%，远高于其他所有LSOA\n",
  "需要实地调查了解：\n",
  "  - 是否邻近大型公园或绿地？\n",
  "  - 是否有特殊的历史保护政策？\n",
  "  - 街道设计是否与众不同？\n\n",
  
  "4. Phase 2 数据收集建议\n",
  "------------------------\n",
  "基于聚类分析，建议优先收集：\n\n",
  
  "A. 高GVI聚类的成功因素：\n",
  "  - 历史规划文件（特别是Lewisham）\n",
  "  - 保护政策和绿化标准\n",
  "  - 社区参与和维护机制\n\n",
  
  "B. 低GVI聚类的限制因素：\n",
  "  - 高密度开发历史\n",
  "  - 交通基础设施优先\n",
  "  - 土地使用冲突\n\n",
  
  "C. 边界案例的转化潜力：\n"
)

if(nrow(boundary_cases) > 0) {
  report <- paste0(report,
                   "  以下LSOA具有转化潜力：\n",
                   paste("  -", boundary_cases$lsoa_code, "(", boundary_cases$borough, ")\n", 
                         collapse = "")
  )
}

report <- paste0(report,
                 "\n5. 政策建议\n",
                 "------------\n",
                 "- 针对低GVI聚类：实施针对性的绿化干预\n",
                 "- 学习高GVI聚类的最佳实践\n",
                 "- 优先改造边界案例LSOA\n",
                 "- 建立Borough间的绿化经验分享机制\n"
)

# 保存报告
writeLines(report, here("output", "cluster_analysis_report.txt"))
cat("  - 聚类分析报告已保存 ✓\n")

# ============================================
# 8. 创建综合可视化面板
# ============================================

cat("\n8. 创建综合可视化...\n")

# 准备4个子图
p1 <- p_cluster_enhanced

p2 <- ggplot(analysis_data %>% filter(!is.na(cluster)), 
             aes(x = cluster_name, y = mean_gvi)) +
  geom_boxplot(aes(fill = cluster_name), alpha = 0.7) +
  geom_jitter(width = 0.2, alpha = 0.5) +
  scale_fill_manual(values = c("High GVI Cluster" = "#2E8B57", 
                               "Low GVI Cluster" = "#DC143C"),
                    guide = "none") +
  labs(title = "GVI Distribution by Cluster", 
       x = "Cluster", y = "Mean GVI (%)") +
  theme_minimal()

p3 <- ggplot(analysis_data %>% filter(!is.na(cluster)), 
             aes(x = cluster_name, y = imd_score)) +
  geom_boxplot(aes(fill = cluster_name), alpha = 0.7) +
  geom_jitter(width = 0.2, alpha = 0.5) +
  scale_fill_manual(values = c("High GVI Cluster" = "#2E8B57", 
                               "Low GVI Cluster" = "#DC143C"),
                    guide = "none") +
  labs(title = "IMD Distribution by Cluster", 
       x = "Cluster", y = "IMD Score") +
  theme_minimal()

p4 <- p_borough_cluster

# 组合图
combined <- grid.arrange(
  p2, p3, p4,
  ncol = 3,
  top = "Cluster Analysis Summary"
)

ggsave(here("figures", "cluster_summary_panel.png"), combined,
       width = 18, height = 6, dpi = 300)

cat("  - 综合可视化已保存 ✓\n")

# ============================================
# 9. 导出关键结果
# ============================================

# 保存聚类成员列表
cluster_membership <- analysis_data %>%
  filter(!is.na(cluster)) %>%
  select(lsoa_code, lsoa_name, borough, cluster, cluster_name, 
         mean_gvi, imd_score, inner_imd_quintile) %>%
  arrange(cluster, desc(mean_gvi))

write_csv(cluster_membership, here("output", "cluster_membership.csv"))

cat("\n\n============================================")
cat("\n聚类分析完成！")
cat("\n============================================\n")

cat("\n关键洞察：\n")
cat("1. 识别出两个明显不同的LSOA类型\n")
cat("2. 绿化水平差异主要不是由贫困程度决定\n")
cat("3. Borough和空间因素可能更重要\n")
cat("4. 存在明显的政策干预机会\n")

cat("\n生成的文件：\n")
cat("- figures/cluster_enhanced_scatter.png\n")
cat("- figures/borough_cluster_distribution.png\n")
cat("- figures/cluster_summary_panel.png\n")
cat("- output/cluster_analysis_report.txt\n")
cat("- output/cluster_membership.csv\n")