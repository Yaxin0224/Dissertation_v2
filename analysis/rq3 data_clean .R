# --- Direct TS006 (Census 2021) LSOA population density ---

suppressPackageStartupMessages({
  library(tidyverse)
  library(here)
  library(readr)
})

cat("\n=== Downloading Census 2021 TS006 (Population density) ===\n")
zip_url <- "https://www.nomisweb.co.uk/output/census/2021/census2021-ts006.zip"

tmp_zip <- tempfile(fileext = ".zip")
download.file(zip_url, tmp_zip, mode = "wb", quiet = TRUE)

# 列出压缩包文件，找包含 LSOA 的 CSV（有时大小写不同）
zf <- unzip(tmp_zip, list = TRUE)
lsoa_csv_name <- zf$Name[grep("lsoa.*\\.csv$", tolower(zf$Name))][1]
if (is.na(lsoa_csv_name)) stop("未在 TS006 压缩包内找到 LSOA CSV")

tmp_csv <- tempfile(fileext = ".csv")
unzip(tmp_zip, files = lsoa_csv_name, exdir = tempdir(), overwrite = TRUE)
file.copy(file.path(tempdir(), lsoa_csv_name), tmp_csv, overwrite = TRUE)

# 读入 LSOA 表；TS006 各列名在不同版本可能略有差异，这里做兼容处理
ts006 <- read_csv(tmp_csv, show_col_types = FALSE)
nm <- tolower(names(ts006))

# 猜测代码列（优先 geography_code / lsoa21cd / lsoa11cd）
code_col <- c("geography_code","lsoa21cd","lsoa11cd","geography code")
code_col <- code_col[code_col %in% nm][1]
if (is.na(code_col)) stop("没找到 LSOA 代码列")

# 猜测“人口密度”列（TS006 通常叫 obs_value 或 v1）
value_col <- c("obs_value","v1","population_density","density","value")
value_col <- value_col[value_col %in% nm][1]
if (is.na(value_col)) stop("没找到人口密度取值列")

ts006_lsoa <- ts006 %>%
  rename(lsoa_code = all_of(names(ts006)[match(code_col, nm)]),
         population_density = all_of(names(ts006)[match(value_col, nm)])) %>%
  mutate(lsoa_code = as.character(lsoa_code)) %>%
  select(lsoa_code, population_density)

# 读入你的 Inner London LSOA 清单
inner_london <- read_csv(
  here("output","full_london_gvi","inner_london_all_lsoas_complete.csv"),
  show_col_types = FALSE
) %>% mutate(lsoa_code = as.character(lsoa_code))

# 只保留 Inner London
pop_inner <- ts006_lsoa %>% semi_join(inner_london, by = "lsoa_code")

cat(paste("Inner London 命中 LSOA 数：", nrow(pop_inner), "\n"))

# 输出到 processed 目录（新文件名）
out_dir <- here("data","processed","rq3")
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

out_file <- file.path(out_dir, "population_density_lsoa_from_TS006.csv")
write_csv(pop_inner, out_file)
cat(paste("已保存：", out_file, "\n"))
