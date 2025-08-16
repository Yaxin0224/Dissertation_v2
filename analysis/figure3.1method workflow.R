# ==========================================
# Methodology Workflow — compact outline, black text
# 方框：仅彩色描边、黑字；列距更紧；方框稍长
# ==========================================
library(ggplot2)
library(grid)  # arrow(), unit()

# 1) 边框配色
cols <- c(
  "Data Sources"       = "#8da0cb",
  "Methods"            = "#66c2a5",
  "Research Questions" = "#fc8d62",
  "Validation"         = "#7f7f7f"
)

# 2) 三列中心位置（调近一些）
col_x <- c(25, 50, 75)   # ← 比之前更靠近
row_y <- c(90, 69, 48, 27)
pad   <- unit(0.75, "lines")  # ← 较大的统一留白，让方框更“长”

# 3) 组装文本
lab <- function(title, lines) {
  paste0(title, "\n", paste(paste0("- ", lines), collapse = "\n"))
}

boxes <- rbind(
  data.frame(phase=1, type="Data Sources", x=col_x[1], y=row_y[1],
             label=lab("Street-View Imagery", c(
               "Mapillary API (138,000+ images",
               "50–100 images per LSOA",
               "2020–2024 temporal range",
               "Adaptive search strategy)"))),
  data.frame(phase=1, type="Data Sources", x=col_x[2], y=row_y[1],
             label=lab("Socioeconomic Data", c(
               "IMD 2019 (all domains)",
               "LSOA boundaries (ONS)",
               "1,731 spatial units",
               "Quintile classification"))),
  data.frame(phase=1, type="Data Sources", x=col_x[3], y=row_y[1],
             label=lab("Urban Morphology", c(
               "Population density",
               "PTAL scores (TfL)",
               "Building density",
               "Road network density"))),
  
  data.frame(phase=2, type="Methods", x=col_x[1], y=row_y[2],
             label=lab("Semantic Segmentation", c(
               "SegFormer model",
               "Vegetation pixel identification",
               "GPU-accelerated processing",
               "Quality validation checks"))),
  data.frame(phase=2, type="Methods", x=col_x[2], y=row_y[2],
             label=lab("GVI Computation", c(
               "Pixel ratio calculation",
               "LSOA-level aggregation",
               "Mean/median statistics",
               "1,660 LSOAs measured"))),
  data.frame(phase=2, type="Methods", x=col_x[3], y=row_y[2],
             label=lab("Data Imputation", c(
               "Random Forest model",
               "71 LSOAs predicted",
               "Cross-validation (R^2 = 0.72)",
               "Uncertainty quantification"))),
  
  data.frame(phase=3, type="Research Questions", x=col_x[1], y=row_y[3],
             label=lab("RQ1: Correlation", c(
               "Pearson & Spearman",
               "Non-linear testing (GAM)",
               "Quintile comparisons",
               "Bootstrap CI (r = -0.327)"))),
  data.frame(phase=3, type="Research Questions", x=col_x[2], y=row_y[3],
             label=lab("RQ2: Spatial Patterns", c(
               "Global Moran's I (0.487)",
               "LISA clustering (31.5%)",
               "Spatial regression models",
               "GWR analysis (local R^2)"))),
  data.frame(phase=3, type="Research Questions", x=col_x[3], y=row_y[3],
             label=lab("RQ3: Mechanisms", c(
               "Interaction effects",
               "Mediation analysis",
               "Multi-level models",
               "Borough-level variation"))),
  
  data.frame(phase=4, type="Validation", x=col_x[1], y=row_y[4],
             label=lab("Sensitivity Analysis", c(
               "Imputation impact assessment",
               "Spatial weight alternatives",
               "Outlier influence testing"))),
  data.frame(phase=4, type="Validation", x=col_x[2], y=row_y[4],
             label=lab("Cross-Validation", c(
               "Bootstrap resampling (1000×)",
               "Leave-one-out validation",
               "Model comparison (AIC/BIC)"))),
  data.frame(phase=4, type="Validation", x=col_x[3], y=row_y[4],
             label=lab("Quality Metrics", c(
               "Coverage: 96% measured",
               "Prediction accuracy: R^2 = 0.72",
               "Spatial coverage: 100%")))
)

phase_titles <- data.frame(
  x = 6,
  y = row_y + 9,
  txt = c("Phase 1: Data Collection & Integration",
          "Phase 2: Image Processing & GVI Calculation",
          "Phase 3: Statistical Analysis by Research Question",
          "Phase 4: Robustness & Validation")
)

arrows <- rbind(
  data.frame(x=col_x[2], xend=col_x[2], y=row_y[1]-7, yend=row_y[2]+7),
  data.frame(x=col_x[2], xend=col_x[2], y=row_y[2]-7, yend=row_y[3]+7),
  data.frame(x=col_x[2], xend=col_x[2], y=row_y[3]-7, yend=row_y[4]+7)
)

stats <- data.frame(
  x = c(18, 38, 62, 82),
  y = 9,
  label = c("1,731\nLSOAs Analysed",
            "138K+\nStreet Images",
            "96%\nDirect Measurement",
            "r = -0.327\nGVI-IMD Correlation")
)

# 4) 绘图
title_y_offset <- 11  # 比原来+9再抬高，避免接触方框
phase_titles$y <- row_y + title_y_offset

p <- ggplot() +
  # 第一层：仅用于“彩色边框 + 计算尺寸”
  geom_label(
    data = boxes,
    aes(x = x, y = y, label = label, color = type),
    fill = NA, label.size = 0.9, label.r = unit(0, "lines"),
    lineheight = 1.08, size = 3.2, label.padding = pad, show.legend = TRUE
  ) +
  # 第二层：同位置再画一次，但无边框、黑色文字
  geom_label(
    data = boxes,
    aes(x = x, y = y, label = label),
    fill = NA, label.size = 0, color = "#111111",
    label.r = unit(0, "lines"),
    lineheight = 1.08, size = 3.2, label.padding = pad, show.legend = FALSE
  ) +
  # 阶段连接箭头
  geom_segment(
    data = arrows,
    aes(x = x, y = y, xend = xend, yend = yend),
    color = "#7d7d7d", linewidth = 0.6,
    arrow = arrow(type = "closed", length = unit(6, "pt"))
  ) +
  # 底部统计
  geom_label(
    data = stats, aes(x = x, y = y, label = label),
    color = "#111111", fill = NA, label.size = 0.6,
    label.r = unit(0, "lines"), lineheight = 1.05,
    size = 4.2, label.padding = unit(0.55, "lines")
  ) +
  # ——把阶段标题放到最后，这样永远在最上层，不会被遮挡——
  geom_text(
    data = phase_titles, aes(x = x, y = y, label = txt),
    hjust = 0, vjust = 0, size = 4.2, fontface = "bold", color = "#333333"
  ) +
  scale_color_manual(
    values = cols, name = "Category",
    guide = guide_legend(override.aes = list(fill = NA, linewidth = 1.2))
  ) +
  coord_cartesian(xlim = c(0, 100), ylim = c(0, 100), expand = FALSE, clip = "off") +
  labs(
    title = "Methodology Workflow",
    subtitle = "City Look: Urban Character Assessment Using Street-View Image Analysis"
  ) +
  theme_void() +
  theme(
    plot.title    = element_text(size = 18, face = "bold", hjust = 0.5, margin = margin(b = 4)),
    plot.subtitle = element_text(size = 11.5, hjust = 0.5, margin = margin(b = 12)),
    legend.position = "bottom",
    legend.title = element_text(size = 10),
    legend.text  = element_text(size = 9),
    plot.margin  = margin(14, 18, 14, 18)
  )

print(p)


out_dir  <- "/Users/yaxin/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Attachments/004/methodology/Dissertation_v2/figures"
out_file <- file.path(out_dir, "Figure 3.1.png")

# 如不存在则创建目录（安全起见）
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

# 导出 PNG（论文尺寸与分辨率）
ggsave(filename = out_file, plot = p,
       width = 16.5, height = 11.5, units = "cm", dpi = 320)

