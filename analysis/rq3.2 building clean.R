# ==== packages ====
pkgs <- c("readODS", "dplyr", "readr", "stringr", "janitor", "tidyr", "purrr")
inst <- rownames(installed.packages())
if (length(setdiff(pkgs, inst)) > 0) {
  install.packages(setdiff(pkgs, inst), repos = "https://cloud.r-project.org")
}
library(readODS); library(dplyr); library(readr); library(stringr); library(janitor); library(tidyr); library(purrr)

# ==== paths (按需修改) ====
raw_dir  <- "C:/Users/z1782/OneDrive - University College London/Attachments/004/methodology/Dissertation_v2/data/raw/rq3"
proc_dir <- "C:/Users/z1782/OneDrive - University College London/Attachments/004/methodology/Dissertation_v2/data/processed/rq3"
ods_path <- file.path(raw_dir,  "Live_Tables_-_Land_Use_Stock_2022_-_LSOA.ods")
inner_csv <- file.path(proc_dir, "population_density_lsoa.csv")  # 第一列是 LSOA code

# ==== 读入 1771 个 Inner London LSOA 清单 ====
inner_codes <- read_csv(inner_csv, show_col_types = FALSE, col_select = 1) |>
  pull(1) |>
  as.character() |>
  str_trim() |>
  unique() |>
  na.omit()

# ---- 小检查（可选）----
if (length(inner_codes) != 1771) {
  message("提醒：清单里共有 ", length(inner_codes), " 个 LSOA（期望 1771）。")
}

# ==== 读取 ODS 的 P404a（比例表，单位为 %）====
raw <- read_ods(ods_path, sheet = "P404a", col_names = FALSE)
hdr_row <- which(raw[[1]] == "LSOA code")[1]
stopifnot(!is.na(hdr_row))

tbl <- raw[(hdr_row + 1):nrow(raw), ]
names(tbl) <- raw[hdr_row, ] |> unlist() |> as.character()
tbl <- clean_names(tbl)  # e.g. "LSOA code" -> lsoa_code

# 仅保留真实 LSOA 行，删除 MSOA 列
tbl <- tbl |>
  filter(!is.na(lsoa_code), str_detect(lsoa_code, "^E010")) |>
  select(-any_of(c("msoa_code", "msoa_name")))

# 把 “.” “-” 空白转 NA，并把数值列转 numeric
num_cols <- setdiff(names(tbl), c("lsoa_code", "lsoa_name"))
tbl <- tbl |>
  mutate(across(all_of(num_cols), ~{
    x <- as.character(.)
    x[x %in% c(".", "-", "")] <- NA
    readr::parse_number(x)
  }))

# 只保留 Inner London 1771 个 LSOA
tbl <- tbl |> filter(lsoa_code %in% inner_codes)

# ==== 计算：land_use_type（主导土地利用）====
# 选择“基础类别”列：排除所有带 total 的汇总列和 grand_total
base_cols <- setdiff(num_cols, num_cols[str_detect(num_cols, "total$|grand_total")])

# 找每行最大占比的列名作为 land_use_type
mat <- as.matrix(select(tbl, all_of(base_cols)))
max_idx <- max.col(mat, ties.method = "first")
tbl$land_use_type <- base_cols[max_idx]

# ==== 计算：building_density（建筑覆盖率，%）====
# 这里把“建筑”理解为：住宅建筑 + 机构/公共建筑 + 产业/商业建筑 + 农业建筑 + 未识别建筑
# 你可以根据 names(tbl) 微调下面的匹配规则
building_patterns <- c(
  "^residential$",                          # 住宅（注意避免 residential_gardens）
  "community_buildings$",                   # 社区建筑
  "leisure_and_recreational_buildings$",    # 文体建筑
  "^industry$", "^offices$", "^retail$", "storage_and_warehousing$",  # 产业商业
  "institutional.*communal.*accommodations",# 机构/集体宿舍等
  "agricultural_buildings$",                # 农业建筑
  "unidentified_building$"                  # 未识别建筑
)
building_cols <- base_cols[str_detect(base_cols, paste(building_patterns, collapse = "|"))]

# 如果没匹配到，给出提示
if (length(building_cols) == 0) {
  stop("未匹配到建筑相关列，请先运行 names(tbl) 查看列名后调整 building_patterns。")
}

message("用于计算 building_density 的列：\n  ",
        paste(building_cols, collapse = ", "))

tbl <- tbl |>
  mutate(building_density = rowSums(across(all_of(building_cols)), na.rm = TRUE))
# building_density 单位为 “百分比”，0–100；若需要比例 0–1，可再 /100

# ==== 输出最终数据 ====
out <- tbl |>
  select(lsoa_code, building_density, land_use_type)

out_path <- file.path(proc_dir, "land_use_building_density_inner_london.csv")
write_csv(out, out_path, na = "")

# 一些友好提示
missing_codes <- setdiff(inner_codes, out$lsoa_code)
if (length(missing_codes) > 0) {
  message("有 ", length(missing_codes), " 个清单内 LSOA 在 P404a 中未找到（或被过滤）：\n",
          paste(missing_codes, collapse = ", "))
}
message("完成：", out_path, " ；行数 = ", nrow(out))
