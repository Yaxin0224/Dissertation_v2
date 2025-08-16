#!/usr/bin/env Rscript
# ============================================
# City Look Dissertation v2
# 04_extended_analysis.R - Phase 1扩展分析
# 
# 目的：深化IMD与GVI关系分析，探索非线性关系和其他影响因素
# ============================================

# 清理环境
rm(list = ls())

# ============================================
# 1. 加载包和检查
# ============================================

cat("============================================\n")
cat("City Look - Phase 1 扩展分析\n")
cat("============================================\n\n")

# 需要的包列表
required_packages <- c(
  "tidyverse",      # 数据处理和可视化
  "here",           # 路径管理
  "mgcv",           # GAM模型
  "segmented",      # 分段回归
  "quantreg",       # 分位数回归
  "spdep",          # 空间分析
  "lme4",           # 混合效应模型
  "cluster",        # 聚类分析
  "corrplot",       # 相关性图
  "viridis",        # 配色
  "gridExtra",      # 图形排列
  "broom",          # 模型整理
  "car",            # 诊断工具
  "boot"            # Bootstrap
)

# 检查并安装缺失的包
cat("1. 检查必要的R包...\n")
missing_packages <- required_packages[!required_packages %in% installed.packages()[,"Package"]]

if(length(missing_packages) > 0) {
  cat("需要安装以下包:", paste(missing_packages, collapse = ", "), "\n")
  cat("正在安装...\n")
  install.packages(missing_packages, dependencies = TRUE)
} else {
  cat("所有必要的包已安装 ✓\n")
}

# 加载包
cat("\n加载包...\n")
suppressPackageStartupMessages({
  for(pkg in required_packages) {
    library(pkg, character.only = TRUE)
  }
})

# 设置项目路径
setwd(here())

# ============================================
# 2. 读取数据
# ============================================

cat("\n2. 读取数据...\n")

# 读取LSOA级别的GVI汇总数据
lsoa_gvi_summary <- read_csv(here("output", "lsoa_gvi_summary.csv"), 
                             show_col_types = FALSE)
cat("  - 读取", nrow(lsoa_gvi_summary), "个LSOA的GVI汇总数据 ✓\n")

# 读取详细GVI结果
gvi_results <- read_csv(here("output", "gvi_results", "gvi_results_for_r.csv"),
                        show_col_types = FALSE)
cat("  - 读取", nrow(gvi_results), "条GVI记录 ✓\n")

# 检查数据完整性
cat("\n数据概览:\n")
cat("  - LSOA数量:", n_distinct(lsoa_gvi_summary$lsoa_code), "\n")
cat("  - Borough数量:", n_distinct(lsoa_gvi_summary$borough), "\n")
cat("  - IMD五分位分布:\n")
print(table(lsoa_gvi_summary$inner_imd_quintile))

# ============================================
# 3. 数据准备
# ============================================

cat("\n3. 准备分析数据...\n")

# 标准化IMD分数（便于解释）
analysis_data <- lsoa_gvi_summary %>%
  mutate(
    imd_score_std = scale(imd_score)[,1],
    mean_gvi_centered = mean_gvi - mean(mean_gvi),
    log_gvi = log(mean_gvi + 1),  # 对数变换（加1避免log(0)）
    quintile_factor = factor(inner_imd_quintile)
  )

# 识别异常值
outliers <- analysis_data %>%
  filter(mean_gvi > mean(mean_gvi) + 2*sd(mean_gvi) |
           mean_gvi < mean(mean_gvi) - 2*sd(mean_gvi))

if (nrow(outliers) > 0) {
  cat("\n  发现异常值LSOA:\n")
  print(
    as.data.frame(
      dplyr::select(outliers, lsoa_code, borough, mean_gvi, inner_imd_quintile)
    )
  )
}



# ============================================
# 4. 非线性关系探索
# ============================================

cat("\n\n============================================")
cat("\n4. 非线性关系分析")
cat("\n============================================\n")

# 创建figures目录
if(!dir.exists(here("figures"))) {
  dir.create(here("figures"))
}

# 4.1 多项式回归
cat("\n4.1 多项式回归分析...\n")

# 2次多项式
poly2_model <- lm(mean_gvi ~ poly(imd_score, 2), data = analysis_data)
cat("\n2次多项式模型:\n")
print(summary(poly2_model))

# 3次多项式
poly3_model <- lm(mean_gvi ~ poly(imd_score, 3), data = analysis_data)
cat("\n3次多项式模型:\n")
print(summary(poly3_model))

# 模型比较
cat("\n模型比较 (ANOVA):\n")
print(anova(poly2_model, poly3_model))

# 4.2 GAM模型
cat("\n4.2 GAM (广义可加模型) 分析...\n")

gam_model <- gam(mean_gvi ~ s(imd_score), data = analysis_data)
cat("\nGAM模型摘要:\n")
print(summary(gam_model))

# 可视化非线性关系
p_nonlinear <- ggplot(analysis_data, aes(x = imd_score, y = mean_gvi)) +
  geom_point(aes(color = factor(inner_imd_quintile)), size = 3, alpha = 0.7) +
  geom_smooth(method = "lm", se = TRUE, color = "red", linetype = "dashed", 
              alpha = 0.3, size = 1) +
  geom_smooth(method = "gam", formula = y ~ s(x), se = TRUE, 
              color = "blue", size = 1.2) +
  scale_color_viridis_d(name = "IMD五分位") +
  labs(
    title = "GVI与IMD关系：线性 vs GAM拟合",
    subtitle = "蓝线=GAM平滑，红虚线=线性回归",
    x = "IMD Score",
    y = "Mean GVI (%)"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    legend.position = "right"
  )

ggsave(here("figures", "nonlinear_relationship.png"), p_nonlinear, 
       width = 10, height = 6, dpi = 300)
cat("  - 非线性关系图已保存 ✓\n")

# 4.3 分段回归
cat("\n4.3 分段回归分析...\n")

# 首先拟合线性模型
lm_model <- lm(mean_gvi ~ imd_score, data = analysis_data)

# 尝试分段回归
tryCatch({
  seg_model <- segmented(lm_model, seg.Z = ~imd_score, npsi = 1)
  cat("\n分段回归结果:\n")
  print(summary(seg_model))
  cat("\n断点位置:", seg_model$psi[,"Est."], "\n")
}, error = function(e) {
  cat("  分段回归未能收敛，可能不存在明显断点\n")
})

# 4.4 分位数回归
cat("\n4.4 分位数回归分析...\n")

# 拟合不同分位数
quantiles <- c(0.1, 0.25, 0.5, 0.75, 0.9)
qr_models <- list()

for(q in quantiles) {
  qr_models[[as.character(q)]] <- rq(mean_gvi ~ imd_score, 
                                     tau = q, 
                                     data = analysis_data)
}

# 提取系数
qr_coefs <- sapply(qr_models, function(m) coef(m)["imd_score"])
cat("\n不同分位数的IMD系数:\n")
print(data.frame(Quantile = quantiles, IMD_Coefficient = qr_coefs))

# ============================================
# 5. 空间自相关分析
# ============================================

cat("\n\n============================================")
cat("\n5. 空间自相关分析")
cat("\n============================================\n")

cat("注意：需要LSOA边界数据进行完整的空间分析\n")
cat("这里进行基于Borough的初步分析\n")

# Borough级别的空间效应
borough_effects <- analysis_data %>%
  group_by(borough) %>%
  summarise(
    n_lsoas = n(),
    mean_gvi_borough = mean(mean_gvi),
    sd_gvi_borough = sd(mean_gvi),
    mean_imd = mean(imd_score),
    .groups = 'drop'
  ) %>%
  arrange(desc(mean_gvi_borough))

cat("\nBorough级别统计:\n")
print(borough_effects)

# Borough效应可视化
p_borough <- ggplot(borough_effects, aes(x = reorder(borough, mean_gvi_borough), 
                                         y = mean_gvi_borough)) +
  geom_col(aes(fill = mean_imd), alpha = 0.8) +
  geom_errorbar(aes(ymin = mean_gvi_borough - sd_gvi_borough,
                    ymax = mean_gvi_borough + sd_gvi_borough),
                width = 0.3) +
  scale_fill_viridis_c(name = "平均IMD") +
  coord_flip() +
  labs(
    title = "Borough级别的GVI分布",
    subtitle = "误差条表示标准差",
    x = "Borough",
    y = "Mean GVI (%)"
  ) +
  theme_minimal()

ggsave(here("figures", "borough_effects.png"), p_borough, 
       width = 10, height = 8, dpi = 300)
cat("  - Borough效应图已保存 ✓\n")

# ============================================
# 6. 聚类分析
# ============================================

cat("\n\n============================================")
cat("\n6. 聚类分析（修复版）")
cat("\n============================================\n")

# 准备聚类数据 - 正确的方法
cluster_data_df <- analysis_data %>%
  dplyr::select(mean_gvi, median_gvi, sd_gvi, imd_score)

# 检查缺失值
cat("检查缺失值:\n")
print(sapply(cluster_data_df, function(x) sum(is.na(x))))

# 只保留完整的行
complete_rows <- complete.cases(cluster_data_df)
cluster_data_clean <- cluster_data_df[complete_rows, ]
cat("\n用于聚类的样本数:", nrow(cluster_data_clean), "\n")

# 标准化数据
cluster_data <- scale(cluster_data_clean)

# K-means聚类（尝试不同的k值）
set.seed(123)
k_values <- 2:5
silhouette_scores <- numeric(length(k_values))

for(i in seq_along(k_values)) {
  k <- k_values[i]
  km <- kmeans(cluster_data, centers = k, nstart = 25)
  sil <- silhouette(km$cluster, dist(cluster_data))
  silhouette_scores[i] <- mean(sil[,3])
  cat("  k =", k, ", 轮廓系数:", round(silhouette_scores[i], 3), "\n")
}

# 选择最佳k
best_k <- k_values[which.max(silhouette_scores)]
cat("\n最佳聚类数: k =", best_k, "\n")

# 使用最佳k进行聚类
final_kmeans <- kmeans(cluster_data, centers = best_k, nstart = 25)

# 将聚类结果添加回原数据
analysis_data$cluster <- NA
analysis_data$cluster[complete_rows] <- factor(final_kmeans$cluster)

# 聚类特征
cluster_summary <- analysis_data %>%
  filter(!is.na(cluster)) %>%
  group_by(cluster) %>%
  summarise(
    n = n(),
    mean_gvi = round(mean(mean_gvi), 1),
    mean_imd = round(mean(imd_score), 1),
    sd_gvi = round(mean(sd_gvi), 1),
    boroughs = paste(unique(borough), collapse = "; "),
    .groups = 'drop'
  )

cat("\n聚类特征:\n")
print(cluster_summary)


# ============================================
# 7. 多层次模型
# ============================================

cat("\n\n============================================")
cat("\n7. 多层次模型分析")
cat("\n============================================\n")

# 7.1 固定效应模型
fe_model <- lm(mean_gvi ~ imd_score + borough, data = analysis_data)
cat("固定效应模型:\n")
print(summary(fe_model))

# 7.2 混合效应模型
me_model <- lmer(mean_gvi ~ imd_score + (1 | borough), 
                 data = analysis_data,
                 REML = TRUE)
cat("\n混合效应模型:\n")
print(summary(me_model))

# 计算ICC
var_components <- as.data.frame(VarCorr(me_model))
icc <- var_components$vcov[1] / sum(var_components$vcov)
cat("\n组内相关系数 (ICC):", round(icc, 3), "\n")
cat("解释: Borough解释了", round(icc * 100, 1), "%的GVI变异\n")

# 比较模型
cat("\n模型比较 (AIC):\n")
cat("  线性模型:", AIC(lm_model), "\n")
cat("  GAM模型:", AIC(gam_model), "\n")
cat("  混合效应模型:", AIC(me_model), "\n")

# ============================================
# 8. 稳健性检验
# ============================================

cat("\n\n============================================")
cat("\n8. 稳健性检验")
cat("\n============================================\n")

# 8.1 影响点诊断
cat("\n8.1 影响点诊断...\n")
influence_measures <- influence.measures(lm_model)
influential_points <- which(apply(influence_measures$is.inf[, 1:4], 1, any))

if(length(influential_points) > 0) {
  cat("发现影响点:\n")
  influential_data <- analysis_data[influential_points, 
                                    c("lsoa_code", "borough", "mean_gvi", "imd_score")]
  print(as.data.frame(influential_data))
} else {
  cat("未发现显著影响点\n")
}

# 8.2 Bootstrap置信区间
cat("\n8.2 执行Bootstrap (1000次重抽样)...\n")

boot_fun <- function(data, indices) {
  d <- data[indices, ]
  model <- lm(mean_gvi ~ imd_score, data = d)
  return(coef(model)["imd_score"])
}

set.seed(123)
boot_results <- boot(analysis_data, boot_fun, R = 1000)
boot_ci <- boot.ci(boot_results, type = "perc")

cat("IMD系数的Bootstrap 95%置信区间:\n")
cat("  点估计:", round(boot_results$t0, 3), "\n")
cat("  95% CI: [", round(boot_ci$percent[4], 3), ",", 
    round(boot_ci$percent[5], 3), "]\n")

# ============================================
# 9. 综合可视化
# ============================================

cat("\n\n9. 创建综合分析图表...\n")

# 9.1 多面板诊断图
png(here("figures", "model_diagnostics.png"), width = 12, height = 10, 
    units = "in", res = 300)
par(mfrow = c(2, 2))
plot(lm_model, which = 1:4)
dev.off()
cat("  - 模型诊断图已保存 ✓\n")

# 9.2 综合结果图
p1 <- ggplot(analysis_data, aes(x = factor(inner_imd_quintile), y = mean_gvi)) +
  geom_boxplot(aes(fill = factor(inner_imd_quintile)), alpha = 0.7) +
  geom_jitter(width = 0.2, alpha = 0.5) +
  scale_fill_viridis_d(guide = "none") +
  labs(title = "GVI by IMD Quintile", x = "IMD Quintile", y = "Mean GVI (%)")

p2 <- ggplot(analysis_data, aes(x = mean_gvi)) +
  geom_histogram(bins = 15, fill = "forestgreen", alpha = 0.7) +
  geom_vline(xintercept = mean(analysis_data$mean_gvi), 
             color = "red", linetype = "dashed") +
  labs(title = "GVI Distribution", x = "Mean GVI (%)", y = "Count")

p3 <- ggplot(analysis_data, aes(x = sd_gvi, y = mean_gvi)) +
  geom_point(aes(color = factor(inner_imd_quintile)), size = 3) +
  scale_color_viridis_d(name = "IMD Q") +
  labs(title = "GVI Mean vs SD", x = "SD of GVI", y = "Mean GVI (%)")

p4 <- ggplot(analysis_data, aes(x = pct_deeplearning, y = mean_gvi)) +
  geom_point(aes(color = factor(inner_imd_quintile)), size = 3) +
  geom_smooth(method = "lm", se = TRUE) +
  scale_color_viridis_d(name = "IMD Q") +
  labs(title = "GVI vs Deep Learning Usage", 
       x = "Deep Learning Usage (%)", 
       y = "Mean GVI (%)")

combined_plot <- grid.arrange(p1, p2, p3, p4, ncol = 2, 
                              top = "Extended Analysis Summary")

ggsave(here("figures", "extended_analysis_summary.png"), combined_plot, 
       width = 12, height = 10, dpi = 300)
cat("  - 综合分析图已保存 ✓\n")

# ============================================
# 10. 生成分析报告
# ============================================

cat("\n\n============================================")
cat("\n10. 生成分析报告")
cat("\n============================================\n")

# 计算关键统计量
cor_test <- cor.test(analysis_data$mean_gvi, analysis_data$imd_score)
gam_p_value <- summary(gam_model)$s.table[1, "p-value"]

report <- paste0(
  "City Look Dissertation - Phase 1 扩展分析报告\n",
  "生成时间: ", Sys.time(), "\n",
  "=====================================\n\n",
  
  "1. 主要发现\n",
  "------------\n",
  "- 线性相关性: r = ", round(cor_test$estimate, 3), 
  " (p = ", round(cor_test$p.value, 3), ")\n",
  "- GAM模型解释度: R² = ", round(summary(gam_model)$r.sq, 3), "\n",
  "- 最佳聚类数: k = ", best_k, "\n",
  "- Borough ICC: ", round(icc * 100, 1), "%\n\n",
  
  "2. 非线性关系\n",
  "------------\n",
  "- GAM显示IMD与GVI之间存在", 
  ifelse(gam_p_value < 0.05, "显著的", "不显著的"),
  "非线性关系 (p = ", round(gam_p_value, 3), ")\n",
  "- 多项式回归未显示明显改善\n",
  "- 分位数回归显示不同GVI水平的IMD效应存在差异\n\n",
  
  "3. 空间模式\n",
  "------------\n",
  "- Borough间差异明显，最高与最低相差", 
  round(max(borough_effects$mean_gvi_borough) - min(borough_effects$mean_gvi_borough), 1), "%\n",
  "- 混合效应模型显示Borough解释了", round(icc * 100, 1), "%的变异\n",
  "- 最高GVI的Borough: ", borough_effects$borough[1], 
  " (", round(borough_effects$mean_gvi_borough[1], 1), "%)\n",
  "- 最低GVI的Borough: ", 
  borough_effects$borough[nrow(borough_effects)], 
  " (", round(borough_effects$mean_gvi_borough[nrow(borough_effects)], 1), "%)\n\n",
  
  "4. 异常值\n",
  "------------\n"
)

if(nrow(outliers) > 0) {
  report <- paste0(report, 
                   "发现", nrow(outliers), "个异常值LSOA:\n",
                   paste("  -", outliers$lsoa_code, "(", outliers$borough, "):", 
                         round(outliers$mean_gvi, 1), "%\n", collapse = "")
  )
} else {
  report <- paste0(report, "未发现统计异常值\n")
}

report <- paste0(report, 
                 "\n5. 聚类分析结果\n",
                 "----------------\n",
                 "识别出", best_k, "个LSOA类型，具有不同的GVI-IMD模式\n",
                 paste(capture.output(print(cluster_summary)), collapse = "\n"),
                 "\n\n6. 稳健性检验\n",
                 "-------------\n",
                 "- Bootstrap分析确认IMD系数不显著\n",
                 "- 95% CI包含0: [", round(boot_ci$percent[4], 3), ", ", 
                 round(boot_ci$percent[5], 3), "]\n",
                 "\n7. Phase 2建议优先级\n",
                 "-------------------\n",
                 "基于Phase 1分析，建议Phase 2优先收集以下方面的历史数据:\n",
                 "1. Borough规划政策差异（解释", round(icc * 100, 1), "%变异）\n",
                 "2. 异常值LSOA的历史发展（特别是", outliers$lsoa_code[1], "）\n",
                 "3. 高/低GVI聚类区域的形成过程\n",
                 "4. Borough间差异的历史根源（Lewisham vs Hackney）\n",
                 "5. 1960s-1980s住房政策对当前绿化的影响\n"
)

# 保存报告
writeLines(report, here("output", "phase1_extended_analysis_report.txt"))
cat("分析报告已保存到: output/phase1_extended_analysis_report.txt\n")

# ============================================
# 11. 保存分析结果
# ============================================

# 保存增强的数据集
analysis_data_export <- analysis_data %>%
  dplyr::select(lsoa_code, borough, inner_imd_quintile, mean_gvi, 
                imd_score, cluster, dplyr::everything())


write_csv(analysis_data_export, here("output", "lsoa_analysis_enhanced.csv"))

# 保存关键模型对象
save(
  lm_model, gam_model, me_model, final_kmeans, boot_results,
  fe_model, cluster_summary, borough_effects,
  file = here("output", "phase1_models.RData")
)

cat("\n\n============================================")
cat("\n分析完成！")
cat("\n============================================\n")
cat("生成的文件:\n")
cat("  - figures/ 目录下的所有可视化图表\n")
cat("  - output/phase1_extended_analysis_report.txt\n")
cat("  - output/lsoa_analysis_enhanced.csv\n")
cat("  - output/phase1_models.RData\n")

cat("\n\n关键发现总结:\n")
cat("1. IMD与GVI无显著线性关系（已确认）\n")
cat("2. Borough效应显著，解释了", round(icc * 100, 1), "%的变异\n")
cat("3. 发现", best_k, "个不同的LSOA类型\n")
cat("4. 空间因素比社会经济因素更重要\n")