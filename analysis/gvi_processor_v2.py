#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
City Look Dissertation v2
GVI处理脚本 - 基于实际文件路径
处理已有的20个LSOA图片，计算GVI
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import torch
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
from tqdm import tqdm
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# ============================================
# 配置 - 基于你的实际路径
# ============================================

class Config:
    """项目配置"""
    # 基础路径
    BASE_DIR = Path(r"C:\Users\z1782\OneDrive - University College London\Attachments\004\methodology\Dissertation_v2")
    
    # 数据文件路径（根据截图确认的路径）
    IMD_FILE = BASE_DIR / "data" / "raw" / "IMD 2019" / "ID 2019 for London.xlsx"
    SELECTED_LSOAS = BASE_DIR / "analysis" / "selected_lsoas.xlsx"  # Excel格式
    MAPILLARY_DIR = BASE_DIR / "data" / "raw" / "mapillary_images"
    
    # 输出路径
    OUTPUT_DIR = BASE_DIR / "output" / "gvi_results"
    
    # 处理参数
    MAX_IMAGES_PER_LSOA = 100  # 每个LSOA最多处理100张
    TEST_MODE_IMAGES = 20  # 测试模式每个LSOA只处理20张

# ============================================
# GVI计算器
# ============================================

class GVICalculator:
    """GVI计算器"""
    
    def __init__(self, model_size='b0'):
        """
        初始化
        model_size: 'b0'(最小), 'b1', 'b2'等
        """
        print("初始化GVI计算器...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 加载SegFormer模型
        model_name = f"nvidia/segformer-{model_size}-finetuned-ade-512-512"
        print(f"加载模型: {model_name}")
        
        try:
            self.feature_extractor = SegformerFeatureExtractor.from_pretrained(model_name)
            self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            print("✅ 模型加载成功")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise
        
        # ADE20K数据集中的植被类别ID
        # 4=tree, 9=grass, 17=plant, 29=field, 46=bush, 66=flower, 90=leaves
        self.vegetation_classes = [4, 9, 17, 29, 46, 66, 90]
    
    def calculate_gvi(self, image_path):
        """计算单张图片的GVI"""
        try:
            # 读取图片
            image = Image.open(image_path).convert('RGB')
            
            # 缩小图片以加快处理（可选）
            max_size = 512
            if max(image.size) > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            # 预处理
            inputs = self.feature_extractor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 推理
            with torch.no_grad():
                outputs = self.model(**inputs)
                # 获取预测结果
                logits = outputs.logits
                pred = logits.argmax(dim=1).cpu().numpy()[0]
            
            # 计算植被像素
            vegetation_mask = np.isin(pred, self.vegetation_classes)
            total_pixels = pred.size
            vegetation_pixels = np.sum(vegetation_mask)
            
            # 计算GVI百分比
            gvi = (vegetation_pixels / total_pixels) * 100
            
            return {
                'gvi': gvi,
                'vegetation_pixels': int(vegetation_pixels),
                'total_pixels': int(total_pixels)
            }
            
        except Exception as e:
            print(f"  ⚠️ 处理失败 {Path(image_path).name}: {e}")
            return None

# ============================================
# LSOA处理器
# ============================================

class LSOAProcessor:
    """处理LSOA的图片"""
    
    def __init__(self, calculator):
        self.calculator = calculator
        
    def process_lsoa(self, lsoa_code, max_images=None):
        """处理单个LSOA的所有图片"""
        
        # 图片文件夹路径
        image_dir = Config.MAPILLARY_DIR / lsoa_code
        
        if not image_dir.exists():
            print(f"❌ 找不到LSOA文件夹: {lsoa_code}")
            return None
        
        # 获取所有jpg图片
        image_files = list(image_dir.glob("*.jpg"))
        n_total = len(image_files)
        
        if n_total == 0:
            print(f"❌ {lsoa_code} 没有图片")
            return None
        
        # 限制处理数量
        if max_images and n_total > max_images:
            image_files = image_files[:max_images]
            print(f"  限制处理数量: {n_total} → {max_images}")
        
        print(f"\n处理 {lsoa_code}: {len(image_files)} 张图片")
        
        # 计算每张图片的GVI
        results = []
        for img_path in tqdm(image_files, desc=f"  {lsoa_code}", leave=False):
            result = self.calculator.calculate_gvi(img_path)
            if result:
                result['image_name'] = img_path.name
                results.append(result)
        
        if len(results) == 0:
            print(f"❌ {lsoa_code} 没有成功处理的图片")
            return None
        
        # 计算统计数据
        gvi_values = [r['gvi'] for r in results]
        
        summary = {
            'lsoa_code': lsoa_code,
            'n_images_total': n_total,
            'n_images_processed': len(results),
            'mean_gvi': np.mean(gvi_values),
            'median_gvi': np.median(gvi_values),
            'std_gvi': np.std(gvi_values),
            'min_gvi': np.min(gvi_values),
            'max_gvi': np.max(gvi_values),
            'q25_gvi': np.percentile(gvi_values, 25),
            'q75_gvi': np.percentile(gvi_values, 75)
        }
        
        print(f"  ✅ 完成: 平均GVI = {summary['mean_gvi']:.2f}%")
        
        return summary, results

# ============================================
# 主程序
# ============================================

def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description='City Look GVI处理')
    parser.add_argument('--test', action='store_true', 
                       help='测试模式（每个LSOA只处理20张图片）')
    parser.add_argument('--lsoas', type=int, default=None,
                       help='处理的LSOA数量（默认全部20个）')
    parser.add_argument('--model', type=str, default='b0',
                       choices=['b0', 'b1', 'b2'],
                       help='SegFormer模型大小')
    
    args = parser.parse_args()
    
    print("="*60)
    print("City Look - GVI处理程序")
    print("="*60)
    
    # 检查关键路径
    print("\n📁 检查文件路径:")
    paths_to_check = [
        ("Mapillary图片文件夹", Config.MAPILLARY_DIR),
        ("输出文件夹", Config.OUTPUT_DIR),
    ]
    
    for name, path in paths_to_check:
        exists = path.exists()
        print(f"  {'✅' if exists else '❌'} {name}: {path}")
    
    # 确保输出目录存在
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 获取LSOA列表
    print("\n📋 获取LSOA列表:")
    lsoa_folders = [f for f in Config.MAPILLARY_DIR.iterdir() if f.is_dir()]
    print(f"  找到 {len(lsoa_folders)} 个LSOA文件夹")
    
    if len(lsoa_folders) == 0:
        print("❌ 没有找到LSOA文件夹，退出")
        return
    
    # 确定要处理的LSOA数量
    if args.lsoas:
        n_process = min(args.lsoas, len(lsoa_folders))
    else:
        n_process = len(lsoa_folders)
    
    lsoa_folders = lsoa_folders[:n_process]
    
    # 确定每个LSOA处理的图片数量
    max_images = Config.TEST_MODE_IMAGES if args.test else Config.MAX_IMAGES_PER_LSOA
    
    print(f"\n⚙️ 处理设置:")
    print(f"  - 模式: {'测试' if args.test else '完整'}")
    print(f"  - LSOA数量: {n_process}")
    print(f"  - 每个LSOA最多: {max_images} 张图片")
    print(f"  - 模型: SegFormer-{args.model}")
    
    # 初始化计算器
    print("\n🚀 开始处理...")
    calculator = GVICalculator(model_size=args.model)
    processor = LSOAProcessor(calculator)
    
    # 处理每个LSOA
    all_summaries = []
    all_details = []
    
    for i, folder in enumerate(lsoa_folders, 1):
        print(f"\n[{i}/{n_process}]", end="")
        
        result = processor.process_lsoa(folder.name, max_images=max_images)
        
        if result:
            summary, details = result
            all_summaries.append(summary)
            
            # 保存详细结果（可选）
            if args.test:  # 测试模式保存详细结果
                details_df = pd.DataFrame(details)
                details_df['lsoa_code'] = folder.name
                all_details.append(details_df)
    
    # 保存结果
    if all_summaries:
        print("\n\n💾 保存结果...")
        
        # 汇总数据
        summary_df = pd.DataFrame(all_summaries)
        
        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存汇总结果
        summary_file = Config.OUTPUT_DIR / f"gvi_summary_{timestamp}.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"  ✅ 汇总结果: {summary_file}")
        
        # 保存详细结果（如果有）
        if all_details:
            details_df = pd.concat(all_details, ignore_index=True)
            details_file = Config.OUTPUT_DIR / f"gvi_details_{timestamp}.csv"
            details_df.to_csv(details_file, index=False)
            print(f"  ✅ 详细结果: {details_file}")
        
        # 为R集成创建简化版本
        r_file = Config.OUTPUT_DIR / "lsoa_gvi_summary.csv"
        summary_df.to_csv(r_file, index=False)
        print(f"  ✅ R集成文件: {r_file}")
        
        # 显示统计信息
        print("\n📊 处理统计:")
        print(f"  - 成功处理: {len(summary_df)} 个LSOA")
        print(f"  - 平均GVI: {summary_df['mean_gvi'].mean():.2f}%")
        print(f"  - GVI范围: {summary_df['mean_gvi'].min():.2f}% - {summary_df['mean_gvi'].max():.2f}%")
        print(f"  - 标准差: {summary_df['mean_gvi'].std():.2f}%")
        
        # 按GVI排序显示前5个和后5个
        print("\n🌳 GVI最高的5个LSOA:")
        top5 = summary_df.nlargest(5, 'mean_gvi')[['lsoa_code', 'mean_gvi']]
        for _, row in top5.iterrows():
            print(f"  {row['lsoa_code']}: {row['mean_gvi']:.2f}%")
        
        print("\n🏢 GVI最低的5个LSOA:")
        bottom5 = summary_df.nsmallest(5, 'mean_gvi')[['lsoa_code', 'mean_gvi']]
        for _, row in bottom5.iterrows():
            print(f"  {row['lsoa_code']}: {row['mean_gvi']:.2f}%")
    
    else:
        print("\n❌ 没有成功处理任何LSOA")
    
    print("\n" + "="*60)
    print("处理完成！")
    print("="*60)

if __name__ == "__main__":
    main()
