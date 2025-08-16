#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试IMD文件读取
"""

import pandas as pd
from pathlib import Path
import os

# 设置基础路径
BASE_DIR = Path(r"C:\Users\z1782\OneDrive - University College London\Attachments\004\methodology\Dissertation_v2")

print("=" * 60)
print("测试IMD文件读取")
print("=" * 60)

# 正确的文件路径（注意是ID不是IMD）
imd_file = BASE_DIR / "data" / "raw" / "IMD 2019" / "ID 2019 for London.xlsx"

print(f"\n尝试读取: {imd_file}")
print(f"文件存在: {imd_file.exists()}")

if imd_file.exists():
    # 读取文件
    df = pd.read_excel(imd_file)
    print(f"✅ 成功读取！")
    print(f"   数据行数: {len(df)}")
    print(f"   数据列数: {len(df.columns)}")
    
    # 显示列名
    print("\n列名（前20个）:")
    for i, col in enumerate(df.columns[:20]):
        print(f"   {i+1}. {col}")
    
    # Inner London Boroughs
    INNER_LONDON_BOROUGHS = [
        "Camden", "Greenwich", "Hackney",
        "Hammersmith and Fulham", "Islington",
        "Kensington and Chelsea", "Lambeth",
        "Lewisham", "Southwark", "Tower Hamlets",
        "Wandsworth", "Westminster"
    ]
    
    # 查找Borough列
    borough_col = None
    for col in df.columns:
        if 'Local Authority District name' in col or 'Borough' in col:
            borough_col = col
            break
    
    if borough_col:
        print(f"\n找到Borough列: {borough_col}")
        # 筛选Inner London
        inner = df[df[borough_col].isin(INNER_LONDON_BOROUGHS)]
        print(f"Inner London LSOAs数量: {len(inner)}")
    
    # 显示需要的列名映射
    print("\n正确的列名映射:")
    print("  lsoa_code: 'LSOA code (2011)'")
    print("  lsoa_name: 'LSOA name (2011)'") 
    print("  imd_score: 'Index of Multiple Deprivation (IMD) Score'")
    print(f"  borough: '{borough_col}'")
    
else:
    print("❌ 文件不存在！")
    print("\n检查其他可能的位置:")
    
    # 检查其他可能的路径
    possible_paths = [
        BASE_DIR / "data" / "raw" / "IMD 2019" / "IMD2019_London.xlsx",
        BASE_DIR / "data" / "raw" / "ID 2019 for London.xlsx",
        BASE_DIR / "data" / "ID 2019 for London.xlsx",
    ]
    
    for path in possible_paths:
        print(f"  {path.name}: {'✅ 存在' if path.exists() else '❌ 不存在'}")

# 测试Token
print("\n" + "=" * 60)
print("检查Mapillary Token")
print("=" * 60)

token = os.getenv("MAPILLARY_TOKEN", "")
if token:
    print(f"✅ Token已设置 (长度: {len(token)})")
else:
    print("❌ Token未设置！")
    print("请运行: set MAPILLARY_TOKEN=你的token")
