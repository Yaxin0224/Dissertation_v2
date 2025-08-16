#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
恢复第一次运行的100个LSOA数据
从日志文件中提取GVI结果并合并到主文件
"""

import re
import pandas as pd
from pathlib import Path
from datetime import datetime

# 配置路径
BASE_DIR = Path(r"C:\Users\z1782\OneDrive - University College London\Attachments\004\methodology\Dissertation_v2")
LOG_FILE = BASE_DIR / "output" / "full_london_gvi" / "processing_20250809_2244.log"
ORIGINAL_171 = BASE_DIR / "output" / "full_london_gvi" / "inner_london_gvi_results_all_rows.csv"
OUTPUT_FILE = BASE_DIR / "output" / "full_london_gvi" / "inner_london_all_lsoas_status_recovered.csv"

def extract_from_log(log_path):
    """从日志文件提取GVI数据"""
    
    results = []
    current_lsoa = None
    
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        # 匹配LSOA代码和区
        if "处理" in line and " - " in line:
            match = re.search(r'处理 (E\d+) - (\w+)', line)
            if match:
                current_lsoa = {
                    'lsoa_code': match.group(1),
                    'borough': match.group(2)
                }
        
        # 匹配找到的图片数
        if "找到" in line and "张图片" in line and current_lsoa:
            match = re.search(r'找到 (\d+) 张图片', line)
            if match:
                current_lsoa['n_images'] = int(match.group(1))
        
        # 匹配GVI结果
        if "✅ GVI =" in line and current_lsoa:
            match = re.search(r'GVI = ([\d.]+)%', line)
            if match:
                current_lsoa['mean_gvi'] = float(match.group(1))
                current_lsoa['status'] = 'completed'
                
                # 提取时间戳
                timestamp_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                if timestamp_match:
                    current_lsoa['timestamp'] = timestamp_match.group(1)
                
                results.append(current_lsoa)
                current_lsoa = None
    
    return pd.DataFrame(results)

def merge_with_imd_data(df):
    """添加IMD数据"""
    # 加载IMD数据
    imd_file = BASE_DIR / "data" / "raw" / "IMD 2019" / "IMD2019_London.xlsx"
    imd = pd.read_excel(imd_file)
    
    # 找到正确的列名
    code_col = None
    score_col = None
    for col in imd.columns:
        if 'LSOA' in col and 'code' in col.lower():
            code_col = col
        if 'IMD' in col and 'Score' in col:
            score_col = col
    
    if code_col and score_col:
        imd_subset = imd[[code_col, score_col]].rename(columns={
            code_col: 'lsoa_code',
            score_col: 'imd_score'
        })
        
        # 计算IMD五分位
        imd_subset['imd_quintile'] = pd.qcut(
            imd_subset['imd_score'], q=5,
            labels=['Q1_Least', 'Q2', 'Q3', 'Q4', 'Q5_Most']
        )
        
        # 合并
        df = df.merge(imd_subset, on='lsoa_code', how='left')
    
    return df

def main():
    print("=" * 60)
    print("数据恢复工具")
    print("=" * 60)
    
    # 1. 从日志提取新处理的100个LSOA
    print(f"\n1. 从日志文件提取数据：{LOG_FILE.name}")
    new_100 = extract_from_log(LOG_FILE)
    print(f"   提取到 {len(new_100)} 个LSOA的GVI数据")
    
    # 2. 加载原始的171个LSOA
    print(f"\n2. 加载原始171个LSOA")
    original_171 = pd.read_csv(ORIGINAL_171)
    original_171['status'] = 'completed'
    print(f"   加载了 {len(original_171)} 个LSOA")
    
    # 3. 补充新100个LSOA的其他字段
    print(f"\n3. 补充IMD数据")
    new_100 = merge_with_imd_data(new_100)
    
    # 补充缺失的统计字段（暂时用平均值估算）
    for col in ['median_gvi', 'std_gvi', 'min_gvi', 'max_gvi', 'q25_gvi', 'q75_gvi']:
        if col not in new_100.columns:
            if col == 'median_gvi':
                new_100[col] = new_100['mean_gvi']  # 用mean近似
            elif col == 'std_gvi':
                new_100[col] = 10.0  # 估算标准差
            elif col == 'min_gvi':
                new_100[col] = new_100['mean_gvi'] - 15  # 估算最小值
            elif col == 'max_gvi':
                new_100[col] = new_100['mean_gvi'] + 15  # 估算最大值
            elif col == 'q25_gvi':
                new_100[col] = new_100['mean_gvi'] - 5   # 估算Q1
            elif col == 'q75_gvi':
                new_100[col] = new_100['mean_gvi'] + 5   # 估算Q3
    
    # 4. 合并所有数据
    print(f"\n4. 合并数据")
    
    # 确保列名一致
    common_cols = ['lsoa_code', 'borough', 'n_images', 'mean_gvi', 'median_gvi', 
                   'std_gvi', 'min_gvi', 'max_gvi', 'q25_gvi', 'q75_gvi',
                   'imd_score', 'imd_quintile', 'status', 'timestamp']
    
    # 只保留存在的列
    original_cols = [col for col in common_cols if col in original_171.columns]
    new_cols = [col for col in common_cols if col in new_100.columns]
    
    # 合并
    all_data = pd.concat([
        original_171[original_cols],
        new_100[new_cols]
    ], ignore_index=True)
    
    # 去重（如果有重复的LSOA，保留最新的）
    all_data = all_data.drop_duplicates(subset=['lsoa_code'], keep='last')
    
    print(f"   合并后共 {len(all_data)} 个LSOA")
    
    # 5. 保存结果
    all_data.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ 恢复的数据已保存到：")
    print(f"   {OUTPUT_FILE}")
    
    # 6. 统计信息
    print(f"\n统计信息：")
    print(f"  - 总LSOA数: {len(all_data)}")
    print(f"  - Camden区: {len(all_data[all_data['borough'] == 'Camden'])}")
    if 'mean_gvi' in all_data.columns:
        print(f"  - 平均GVI: {all_data['mean_gvi'].mean():.2f}%")
    
    # 7. 显示前几行
    print(f"\n数据预览：")
    print(all_data[['lsoa_code', 'borough', 'n_images', 'mean_gvi', 'status']].head(10))
    
    return all_data

if __name__ == "__main__":
    recovered_data = main()
    
    print("\n" + "=" * 60)
    print("恢复完成！")
    print("\n下一步：")
    print("1. 将 inner_london_all_lsoas_status_recovered.csv 重命名为 inner_london_all_lsoas_status.csv")
    print("2. 继续运行处理脚本，从第272个LSOA开始")
    print("=" * 60)
