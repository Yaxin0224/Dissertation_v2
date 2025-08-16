#!/usr/bin/env python3
# Fix failed LSOAs and finalize dataset with km/km² column
# Requires: osmnx>=2, geopandas, shapely, tqdm, pandas, numpy, matplotlib (optional)

import os
import time
import argparse
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ---------- OSMnx v2 settings ----------
ox.settings.use_cache = True
ox.settings.log_console = True
ox.settings.overpass_rate_limit = True
ox.settings.timeout = 300  # 默认更宽松

def parse_args():
    p = argparse.ArgumentParser("Retry failed LSOAs and add km/km² density column")
    p.add_argument("--base-dir",
                   default=r"C:\Users\z1782\OneDrive - University College London\Attachments\004\methodology\Dissertation_v2",
                   help="你的项目根目录（含 data/ 与 output/）")
    p.add_argument("--primary-network-type", default="drive",
                   choices=["drive", "walk", "bike", "all", "drive_service"],
                   help="主网络类型（先用它重试）")
    p.add_argument("--fallback-types", default="walk,all",
                   help="备用网络类型列表，逗号分隔（留空则不使用备用）")
    p.add_argument("--batch-size", type=int, default=5,
                   help="重试时每批处理的 LSOA 数")
    p.add_argument("--sleep", type=float, default=6.0,
                   help="每批之间的暂停秒数")
    p.add_argument("--timeout", type=int, default=300,
                   help="Overpass 超时（秒），会写入 ox.settings.timeout")
    p.add_argument("--max-retries", type=int, default=3,
                   help="每个网络类型上的最大重试次数（HTTP/超时等）")
    p.add_argument("--no-plots", action="store_true", help="不生成直方图")
    p.add_argument("--overwrite", action="store_true", help="覆盖原 CSV（谨慎）")
    return p.parse_args()

def get_stats_for_polygon(polygon_27700, network_type):
    """Return dict stats for one polygon (EPSG:27700)."""
    try:
        # 1) 用 WGS84 去抓
        poly_wgs84 = gpd.GeoSeries([polygon_27700], crs=27700).to_crs(4326).iloc[0]
        G = ox.graph_from_polygon(poly_wgs84,
                                  network_type=network_type,
                                  simplify=True,
                                  retain_all=False)
        # 2) 面积（m²）
        area_m2 = gpd.GeoSeries([polygon_27700], crs=27700).area.iloc[0]
        stats = ox.stats.basic_stats(G, area=area_m2)

        res = {
            "n_nodes": G.number_of_nodes(),
            "n_edges": G.number_of_edges(),
            "total_edge_length": float(stats.get("edge_length_total", np.nan)),
            "edge_density_km_per_sqkm": float(stats.get("edge_density_km", np.nan)),
            "street_density_km_per_sqkm": float(stats.get("street_density_km", np.nan)),
            "avg_street_length": float(stats.get("street_length_avg", np.nan)),
            "circuity_avg": float(stats.get("circuity_avg", np.nan)),
            "status": "success",
        }
    except Exception as e:
        res = {
            "n_nodes": np.nan,
            "n_edges": np.nan,
            "total_edge_length": np.nan,
            "edge_density_km_per_sqkm": np.nan,
            "street_density_km_per_sqkm": np.nan,
            "avg_street_length": np.nan,
            "circuity_avg": np.nan,
            "status": f"error: {e}",
        }
    return res

def retry_one(polygon_27700, network_types, max_retries):
    """Try several network types with retries; return first successful dict + info."""
    for nt in network_types:
        for attempt in range(1, max_retries + 1):
            res = get_stats_for_polygon(polygon_27700, nt)
            ok = (res["status"] == "success") and pd.notna(res["total_edge_length"]) and (res["total_edge_length"] > 0)
            if ok:
                res["used_network_type"] = nt
                res["attempt"] = attempt
                return res
            time.sleep(1.5)  # 小等一下再试
    # 全失败
    res["used_network_type"] = None
    res["attempt"] = max_retries
    return res

def main():
    args = parse_args()
    ox.settings.timeout = args.timeout

    BASE = args.base_dir
    DATA = os.path.join(BASE, "data")
    OUT  = os.path.join(DATA, "processed", "rq3")
    os.makedirs(OUT, exist_ok=True)

    # 读取主数据（详细版）
    detailed_path = os.path.join(OUT, "road_network_density_detailed.csv")
    if not os.path.exists(detailed_path):
        raise FileNotFoundError(f"找不到 {detailed_path}，请先运行主脚本生成。")

    df = pd.read_csv(detailed_path)

    # 读取边界（为重试准备几何）
    lsoa_file = os.path.join(BASE, "output", "full_london_gvi", "inner_london_all_lsoas_complete.csv")
    inner = pd.read_csv(lsoa_file)
    shp = os.path.join(DATA, "raw", "statistical-gis-boundaries-london", "ESRI", "LSOA_2011_London_gen_MHW.shp")
    gdf = gpd.read_file(shp)
    gdf = gdf[gdf["LSOA11CD"].isin(inner["lsoa_code"])]
    gdf = gdf.to_crs(27700)  # 面积与抓取逻辑一致

    # 找出失败/可疑样本
    if "status" not in df.columns:
        df["status"] = "unknown"
    fail_mask = (df["status"]!="success") | (df["total_edge_length"].isna()) | (df["total_edge_length"]<=0)
    failed = df.loc[fail_mask, "lsoa_code"].dropna().unique().tolist()

    if not failed:
        print("✅ 没有需要修复的 LSOA，直接做单位列并导出。")
    else:
        print(f"需重试的 LSOA 数：{len(failed)}")

        # 网络类型顺序：主类型 + 备用类型
        fallback = [t.strip() for t in args.fallback_types.split(",") if t.strip()] if args.fallback_types else []
        net_types = [args.primary_network_type] + fallback

        updates = []
        # 小批量处理
        for i in tqdm(range(0, len(failed), args.batch_size), desc="Retrying failed LSOAs"):
            batch_codes = failed[i:i+args.batch_size]
            sub = gdf[gdf["LSOA11CD"].isin(batch_codes)]
            for _, row in sub.iterrows():
                res = retry_one(row.geometry, net_types, args.max_retries)
                # 面积（km²）和自算密度
                area_km2 = row.geometry.area / 1_000_000.0
                if (res["status"]=="success") and pd.notna(res["total_edge_length"]) and area_km2>0:
                    rd = res["total_edge_length"] / (area_km2 * 1_000_000.0)  # m/m²
                else:
                    rd = np.nan

                res.update({
                    "lsoa_code": row["LSOA11CD"],
                    "lsoa_name": row.get("LSOA11NM", np.nan),
                    "area_sqkm": area_km2,
                    "road_density": rd,
                })
                updates.append(res)

            if i + args.batch_size < len(failed):
                time.sleep(args.sleep)

        upd = pd.DataFrame(updates)

        # 将更新写回 df（以 lsoa_code 为键替换）
        df = df.set_index("lsoa_code")
        upd = upd.set_index("lsoa_code")
        cols_to_replace = [
            "n_nodes","n_edges","total_edge_length",
            "edge_density_km_per_sqkm","street_density_km_per_sqkm",
            "avg_street_length","circuity_avg","status","area_sqkm","road_density"
        ]
        for c in cols_to_replace:
            if c in upd.columns:
                df.loc[upd.index, c] = upd[c]

        df = df.reset_index()

    # 统一新增 km/km² 列（论文主指标）
    df["road_density_km_per_sqkm"] = df["road_density"] * 1000.0

    # 输出文件名
    main_name   = "road_network_density_lsoa"
    detail_name = "road_network_density_detailed"
    suffix = "" if args.overwrite else "_clean"

    out_main   = os.path.join(OUT, f"{main_name}{suffix}.csv")
    out_detail = os.path.join(OUT, f"{detail_name}{suffix}.csv")
    report     = os.path.join(OUT, f"road_density_extraction_report{suffix}.txt")

    # 主表选择常用列
    keep_cols = [
        "lsoa_code",
        "road_density_km_per_sqkm",  # NEW 主指标
        "road_density",              # m/m²（保留，便于复核）
        "street_density_km_per_sqkm",
        "edge_density_km_per_sqkm",
        "n_nodes","n_edges","total_edge_length",
        "avg_street_length","circuity_avg",
        "status"
    ]
    exist_keep = [c for c in keep_cols if c in df.columns]
    df_main = df[exist_keep].copy()

    # 保存
    df_main.to_csv(out_main, index=False)
    df.to_csv(out_detail, index=False)

    # 报告
    ok = df["status"].eq("success").sum()
    bad = (~df["status"].eq("success")).sum()
    with open(report, "w", encoding="utf-8") as f:
        f.write("Road Density Finalization Report\n")
        f.write("="*50 + "\n")
        f.write(f"Generated: {pd.Timestamp.now()}\n\n")
        f.write(f"Total LSOAs: {len(df)}\n")
        f.write(f"Successful: {ok}\n")
        f.write(f"Failed/Imputed/Other: {bad}\n\n")
        if "road_density_km_per_sqkm" in df_main.columns:
            f.write("Summary (road_density_km_per_sqkm):\n")
            f.write(df_main["road_density_km_per_sqkm"].describe().to_string())

    print(f"\n✅ Saved main CSV to: {out_main}")
    print(f"✅ Saved detailed CSV to: {out_detail}")
    print(f"✅ Saved report to: {report}")

    # 可选直方图
    if not args.no_plots:
        try:
            import matplotlib.pyplot as plt
            plot_file = os.path.join(OUT, f"road_density_summary_plots{suffix}.png")
            plt.figure(figsize=(8,6))
            plt.hist(df_main["road_density_km_per_sqkm"].dropna(), bins=30, edgecolor="black")
            plt.xlabel("Road Density (km/km²)")
            plt.ylabel("Number of LSOAs")
            plt.title("Distribution of Road Network Density (km/km²)")
            plt.tight_layout()
            plt.savefig(plot_file, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"✅ Plot saved to: {plot_file}")
        except Exception as e:
            print(f"⚠ Skipping plot: {e}")

if __name__ == "__main__":
    main()
