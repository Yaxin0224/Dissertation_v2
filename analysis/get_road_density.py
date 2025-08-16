#!/usr/bin/env python3
# =============================================================================
# City Look Dissertation - Road Network Density Extraction (OSMnx 2.x)
# Using OSMnx to calculate road density for Inner London LSOAs
# Author: You
# Date: 2025
# =============================================================================

import os
import time
import argparse
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
from shapely.geometry import Polygon
from tqdm import tqdm

warnings.filterwarnings("ignore")

# -----------------------------
# OSMnx settings (v2.x style)
# -----------------------------
ox.settings.use_cache = True
ox.settings.log_console = True
ox.settings.timeout = 180                # 秒
ox.settings.overpass_rate_limit = True   # 礼貌限流

# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Extract road network density for Inner London LSOAs (OSMnx 2.x)"
    )
    p.add_argument(
        "--base-dir",
        default=r"C:\Users\z1782\OneDrive - University College London\Attachments\004\methodology\Dissertation_v2",
        help="项目根目录（包含 data/ 与 output/）",
    )
    p.add_argument(
        "--network-type",
        default="drive",
        choices=["drive", "walk", "bike", "all", "drive_service"],
        help="OSMnx 网络类型",
    )
    p.add_argument(
        "--test",
        action="store_true",
        help="测试模式：仅处理前 N 个 LSOA（配合 --limit）",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=10,
        help="测试模式下要处理的 LSOA 数量",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="每批处理的 LSOA 数量",
    )
    p.add_argument(
        "--sleep",
        type=float,
        default=2.0,
        help="每批结束后的暂停秒数",
    )
    p.add_argument(
        "--no-plots",
        action="store_true",
        help="跳过可视化图表生成",
    )
    return p.parse_args()


# -----------------------------
# 路网提取（单个 Polygon，输入为 EPSG:27700）
# -----------------------------
def get_road_network_for_polygon(polygon_27700, network_type="drive"):
    """
    polygon_27700: shapely Polygon in EPSG:27700 (meters)
    return: dict with stats
    """
    try:
        # 1) 用经纬度边界抓图（OSM 用 WGS84）
        poly_wgs84 = gpd.GeoSeries([polygon_27700], crs=27700).to_crs(4326).iloc[0]

        G = ox.graph_from_polygon(
            poly_wgs84,
            network_type=network_type,
            simplify=True,
            retain_all=False,
        )

        # 2) 面积（m²，EPSG:27700）
        area_m2 = gpd.GeoSeries([polygon_27700], crs=27700).area.iloc[0]

        # 3) OSMnx 统计（area 单位必须是 m²）
        stats = ox.stats.basic_stats(G, area=area_m2)

        result = {
            "n_nodes": G.number_of_nodes(),
            "n_edges": G.number_of_edges(),
            "total_edge_length": stats.get("edge_length_total", 0.0),  # meters
            "edge_density_km_per_sqkm": stats.get("edge_density_km", np.nan),
            "street_density_km_per_sqkm": stats.get("street_density_km", np.nan),
            "avg_street_length": stats.get("street_length_avg", np.nan),
            "circuity_avg": stats.get("circuity_avg", np.nan),
            "status": "success",
        }

    except Exception as e:
        result = {
            "n_nodes": 0,
            "n_edges": 0,
            "total_edge_length": 0.0,
            "edge_density_km_per_sqkm": np.nan,
            "street_density_km_per_sqkm": np.nan,
            "avg_street_length": np.nan,
            "circuity_avg": np.nan,
            "status": f"error: {e}",
        }

    return result


# -----------------------------
# 批处理
# -----------------------------
def process_lsoas_batch(lsoa_gdf_27700, network_type="drive", batch_size=20, sleep_time=2):
    results = []
    total = len(lsoa_gdf_27700)
    print(f"\nProcessing {total} LSOAs in batches of {batch_size}...")

    with tqdm(total=total, desc="Extracting road networks") as pbar:
        for i, (_, row) in enumerate(lsoa_gdf_27700.iterrows(), start=1):
            res = get_road_network_for_polygon(row.geometry, network_type=network_type)

            # 基本标识
            res["lsoa_code"] = row["LSOA11CD"]
            res["lsoa_name"] = row["LSOA11NM"]

            # 面积（km²）
            area_km2 = row.geometry.area / 1_000_000.0
            res["area_sqkm"] = area_km2

            # 额外密度（m/m²），可与 OSMnx 的密度作对照
            if res["total_edge_length"] and area_km2 > 0:
                res["road_density"] = res["total_edge_length"] / (area_km2 * 1_000_000.0)
            else:
                res["road_density"] = 0.0

            results.append(res)
            pbar.update(1)

            # 每批暂停
            if (i % batch_size == 0) and (i < total):
                time.sleep(sleep_time)

    return pd.DataFrame(results)


# -----------------------------
# 主流程
# -----------------------------
def main():
    args = parse_args()

    BASE_DIR = args.base_dir
    DATA_DIR = os.path.join(BASE_DIR, "data")
    OUTPUT_DIR = os.path.join(DATA_DIR, "processed", "rq3")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("ROAD NETWORK DENSITY EXTRACTION FOR INNER LONDON LSOAs")
    print("=" * 60)
    print("\n[1/6] Setting up paths and loading data...")

    # 1) LSOA 列表（必须包含 lsoa_code 列；可选 borough 列）
    lsoa_file = os.path.join(BASE_DIR, "output", "full_london_gvi", "inner_london_all_lsoas_complete.csv")
    inner_london = pd.read_csv(lsoa_file)
    if "lsoa_code" not in inner_london.columns:
        raise ValueError("inner_london CSV 中需要包含列 'lsoa_code'。")

    print(f"✓ Loaded {len(inner_london)} Inner London LSOAs")

    # 2) LSOA 边界
    shp = os.path.join(
        DATA_DIR,
        "raw",
        "statistical-gis-boundaries-london",
        "ESRI",
        "LSOA_2011_London_gen_MHW.shp",
    )
    lsoa_boundaries = gpd.read_file(shp)
    print(f"✓ Loaded LSOA boundaries shapefile with {len(lsoa_boundaries)} features")

    # 3) 仅 Inner London
    lsoa_boundaries_inner = lsoa_boundaries[lsoa_boundaries["LSOA11CD"].isin(inner_london["lsoa_code"])]
    print(f"✓ Filtered to {len(lsoa_boundaries_inner)} Inner London LSOAs")

    # 4) 投影到米制（EPSG:27700）以获得准确面积
    if lsoa_boundaries_inner.crs is None:
        raise ValueError("LSOA 边界缺少 CRS，请检查 shapefile。")
    lsoa_boundaries_inner = lsoa_boundaries_inner.to_crs(27700)
    print("✓ Reprojected to EPSG:27700 (meters)")

    # 测试模式：只取前 N 个
    if args.test:
        lsoa_boundaries_inner = lsoa_boundaries_inner.head(args.limit)
        print(f"⚠ TEST MODE: Processing only first {len(lsoa_boundaries_inner)} LSOAs")

    # -------------------------
    # [3/6] 提取路网
    # -------------------------
    print("\n[3/6] Extracting road networks from OpenStreetMap...")
    print("Note: Full run for all LSOAs may take 30–60 minutes.")

    start_time = time.time()
    road_df = process_lsoas_batch(
        lsoa_boundaries_inner,
        network_type=args.network_type,
        batch_size=args.batch_size,
        sleep_time=args.sleep,
    )
    elapsed = time.time() - start_time

    print(f"\n✓ Completed in {elapsed/60:.1f} minutes")
    print(f"✓ Successfully processed {road_df[road_df['status']=='success'].shape[0]} LSOAs")
    print(f"⚠ Failed for {road_df[road_df['status']!='success'].shape[0]} LSOAs")

    # -------------------------
    # [4/6] 质量检查
    # -------------------------
    print("\n[4/6] Performing quality checks...")
    if "road_density" not in road_df.columns or road_df["road_density"].dropna().empty:
        print("⚠ road_density 列缺失或全空，跳过统计。")
        outliers = road_df.iloc[0:0].copy()
    else:
        Q1 = road_df["road_density"].quantile(0.25)
        Q3 = road_df["road_density"].quantile(0.75)
        IQR = Q3 - Q1
        outliers = road_df[
            (road_df["road_density"] < Q1 - 1.5 * IQR)
            | (road_df["road_density"] > Q3 + 1.5 * IQR)
        ]

        print("\nRoad Density Statistics:")
        print(f"  Mean:   {road_df['road_density'].mean():.6f} m/m²")
        print(f"  Median: {road_df['road_density'].median():.6f} m/m²")
        print(f"  Std:    {road_df['road_density'].std():.6f}")
        print(f"  Min:    {road_df['road_density'].min():.6f}")
        print(f"  Max:    {road_df['road_density'].max():.6f}")
        print(f"  Outliers: {len(outliers)} LSOAs")

    # 按 borough 汇总（如果有）
    has_borough = "borough" in inner_london.columns
    if has_borough:
        road_with_borough = road_df.merge(
            inner_london[["lsoa_code", "borough"]], on="lsoa_code", how="left"
        )
        borough_summary = road_with_borough.groupby("borough").agg(
            road_density_mean=("road_density", "mean"),
            road_density_median=("road_density", "median"),
            road_density_std=("road_density", "std"),
            street_density_km_per_sqkm_mean=("street_density_km_per_sqkm", "mean"),
            n_edges_sum=("n_edges", "sum"),
        ).round(4)
        print("\nRoad Density by Borough (mean/median/std etc.):")
        print(borough_summary)

    # -------------------------
    # [5/6] 处理缺失
    # -------------------------
    print("\n[5/6] Handling missing data...")
    all_codes = set(inner_london["lsoa_code"])
    processed_codes = set(road_df["lsoa_code"])
    missing_codes = all_codes - processed_codes

    if missing_codes:
        print(f"⚠ Missing data for {len(missing_codes)} LSOAs")

        if has_borough:
            tmp_merge = road_df.merge(
                inner_london[["lsoa_code", "borough"]], on="lsoa_code", how="left"
            )
            borough_means = tmp_merge.groupby("borough")["road_density"].mean()
        else:
            borough_means = pd.Series(dtype=float)

        overall_mean = road_df["road_density"].mean()

        miss_rows = []
        for code in missing_codes:
            if has_borough:
                borough = inner_london.loc[
                    inner_london["lsoa_code"] == code, "borough"
                ].values[0]
                imputed_density = borough_means.get(borough, overall_mean)
            else:
                imputed_density = overall_mean

            miss_rows.append(
                {
                    "lsoa_code": code,
                    "road_density": imputed_density,
                    "street_density_km_per_sqkm": np.nan,
                    "edge_density_km_per_sqkm": np.nan,
                    "n_nodes": np.nan,
                    "n_edges": np.nan,
                    "total_edge_length": np.nan,
                    "avg_street_length": np.nan,
                    "circuity_avg": np.nan,
                    "status": "imputed",
                }
            )

        road_complete = pd.concat([road_df, pd.DataFrame(miss_rows)], ignore_index=True)
    else:
        print("✓ All LSOAs processed successfully!")
        road_complete = road_df

    # -------------------------
    # [6/6] 保存
    # -------------------------
    print("\n[6/6] Saving results...")

    final_cols = [
        "lsoa_code",
        "road_density",
        "street_density_km_per_sqkm",
        "edge_density_km_per_sqkm",
        "n_nodes",
        "n_edges",
        "total_edge_length",
        "avg_street_length",
        "circuity_avg",
        "status",
    ]
    existing = [c for c in final_cols if c in road_complete.columns]
    road_final = road_complete[existing].copy()

    output_file = os.path.join(OUTPUT_DIR, "road_network_density_lsoa.csv")
    detailed_file = os.path.join(OUTPUT_DIR, "road_network_density_detailed.csv")
    report_file = os.path.join(OUTPUT_DIR, "road_density_extraction_report.txt")

    road_final.to_csv(output_file, index=False)
    road_complete.to_csv(detailed_file, index=False)

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("Road Network Density Extraction Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated: {pd.Timestamp.now()}\n\n")
        f.write(f"Total LSOAs processed: {len(road_complete)}\n")
        f.write(f"Successful extractions: {len(road_complete[road_complete['status']=='success'])}\n")
        f.write(f"Failed/Imputed: {len(road_complete[road_complete['status']!='success'])}\n\n")
        if not road_final.empty:
            f.write("Summary Statistics (numeric columns):\n")
            f.write(road_final.describe().to_string())

    print(f"✓ Saved main CSV to: {output_file}")
    print(f"✓ Saved detailed CSV to: {detailed_file}")
    print(f"✓ Report saved to: {report_file}")

    # -------------------------
    # 可选：图表
    # -------------------------
    if not args.no_plots:
        try:
            import matplotlib.pyplot as plt

            plot_file = os.path.join(OUTPUT_DIR, "road_density_summary_plots.png")
            plt.figure(figsize=(8, 6))
            plt.hist(road_final["road_density"].dropna(), bins=30, edgecolor="black")
            plt.xlabel("Road Density (m/m²)")
            plt.ylabel("Number of LSOAs")
            plt.title("Distribution of Road Network Density")
            plt.tight_layout()
            plt.savefig(plot_file, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"✓ Plots saved to: {plot_file}")
        except Exception as e:
            print(f"⚠ Skipping plots: {e}")

    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE!")
    print("=" * 60)
    print(f"\nFinal dataset contains {len(road_final)} LSOAs")
    print(f"Main output: {output_file}")
    print("\nNext steps:")
    print("1) Review the road density statistics")
    print("2) Check anomalies/outliers")
    print("3) Merge with other RQ3 variables for analysis")


if __name__ == "__main__":
    main()
