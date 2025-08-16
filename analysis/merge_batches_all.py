#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_batches_all.py
把 .../output/full_london_gvi 目录中的所有 batch_*.csv 完整合并。
- all_rows: 纯拼接，保留所有行和所有列（列自动按并集对齐）。
- latest_per_lsoa: 按 lsoa_code 去重，默认保留最新批次（文件名时间）那一条。
可选：输出一个 xlsx，含两个工作表。

用法：
    python merge_batches_all.py
    python merge_batches_all.py --folder "D:\...\full_london_gvi" --keep last --excel
"""

from pathlib import Path
import argparse
import re
import pandas as pd

# 根据你的项目结构设默认目录
DEFAULT_BASE = Path(r"C:\Users\z1782\OneDrive - University College London\Attachments\004\methodology\Dissertation_v2")
DEFAULT_FOLDER = DEFAULT_BASE / "output" / "full_london_gvi"

def parse_args():
    ap = argparse.ArgumentParser(description="合并 full_london_gvi 的所有 batch_*.csv（保留所有行与所有列）")
    ap.add_argument("--folder", type=str, default=str(DEFAULT_FOLDER), help="batch 文件所在文件夹")
    ap.add_argument("--keep", choices=["last", "first"], default="last",
                    help="同一 lsoa_code 多条记录时保留哪一条（按文件时间顺序）")
    ap.add_argument("--excel", action="store_true", help="额外导出 XLSX（两个工作表）")
    return ap.parse_args()

def sort_key(p: Path):
    # 解析 batch_YYYYMMDD_HHMMSS.csv 中的时间，用于排序
    m = re.search(r"batch_(\d{8})_(\d{6})\.csv$", p.name)
    return (m.group(1), m.group(2)) if m else ("00000000", "000000")

def main():
    args = parse_args()
    folder = Path(args.folder)
    folder.mkdir(parents=True, exist_ok=True)

    files = sorted(folder.glob("batch_*.csv"), key=sort_key)
    if not files:
        print(f"⚠️ 未在 {folder} 找到任何 batch_*.csv")
        return

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            # 增加来源信息，便于追踪
            m = re.search(r"batch_(\d{8})_(\d{6})\.csv$", f.name)
            ts = f"{m.group(1)}_{m.group(2)}" if m else ""
            df["__source_file"] = f.name
            df["__batch_time"] = ts
            dfs.append(df)
        except Exception as e:
            print(f"⚠️ 读取失败：{f.name} -> {e}")

    if not dfs:
        print("⚠️ 没有可合并的数据。")
        return

    # 1) 纯拼接：列按并集对齐，缺失补 NaN —— 保留所有行和所有列
    all_rows = pd.concat(dfs, ignore_index=True, sort=False)

    # 2) 去重版：按时间顺序，保留同一 lsoa_code 的最新/最早一条
    if "lsoa_code" in all_rows.columns:
        # 按批次时间排序以控制“最新/最早”
        all_rows = all_rows.copy()
        all_rows["__order"] = all_rows["__batch_time"]
        all_rows.sort_values(["lsoa_code", "__order"], inplace=True)
        keep = "last" if args.keep == "last" else "first"
        latest = all_rows.drop_duplicates(subset=["lsoa_code"], keep=keep).drop(columns="__order")
    else:
        latest = all_rows.copy()  # 没有 lsoa_code 就不做去重

    # 输出 CSV
    out_all = folder / "inner_london_gvi_results_all_rows.csv"
    out_latest = folder / "inner_london_gvi_results_latest_per_lsoa.csv"
    all_rows.to_csv(out_all, index=False)
    latest.to_csv(out_latest, index=False)

    # 可选：输出一个 Excel，两个工作表
    if args.excel:
        xlsx = folder / "merged_results.xlsx"
        with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
            # 注意：Excel 单表最多 ~1,048,576 行，如超限建议只写 latest
            all_rows.to_excel(w, index=False, sheet_name="all_rows")
            latest.to_excel(w, index=False, sheet_name="latest_per_lsoa")

    # 打印摘要
    print("✅ 合并完成")
    print(f"   批次数：{len(files)}")
    print(f"   all_rows 行数：{len(all_rows)}  ->  {out_all}")
    if "mean_gvi" in latest.columns:
        print(f"   latest_per_lsoa 行数：{len(latest)}，平均 GVI：{latest['mean_gvi'].mean():.2f}%  ->  {out_latest}")
    else:
        print(f"   latest_per_lsoa 行数：{len(latest)} -> {out_latest}")

if __name__ == "__main__":
    main()
