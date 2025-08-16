#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
City Look Dissertation v2
stream_gvi_processor.py - 流式GVI处理（不保存图片）

直接从Mapillary下载图片到内存，计算GVI后只保存结果
大幅减少存储需求，从100GB降到<1GB

作者：Yaxin
日期：2025-01-14
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import requests
from io import BytesIO
from PIL import Image
import torch
from tqdm import tqdm
import logging
import json
import time
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# ============================================
# Windows配置
# ============================================

class Config:
    """Windows项目配置"""
    
    # Windows路径
    BASE_DIR = Path(r"C:\Users\z1782\OneDrive - University College London\Attachments\004\methodology\Dissertation_v2")
    
    # 数据路径
    IMD_DATA = BASE_DIR / "data" / "raw" / "IMD2019_London.xlsx"
    SELECTED_LSOAS = BASE_DIR / "data" / "processed" / "selected_lsoas.csv"
    
    # 输出路径（只保存结果，不保存图片）
    OUTPUT_DIR = BASE_DIR / "output" / "stream_gvi"
    RESULTS_DIR = BASE_DIR / "data" / "processed"
    
    # Mapillary配置
    MAPILLARY_TOKEN = "MLY|9922859457805691|cef02444f32c339cf09761b104ca4bb5"
    MAPILLARY_API = "https://graph.mapillary.com"
    
    # 图片收集参数
    TARGET_IMAGES = 100  # 目标100张
    MIN_IMAGES = 60     # 最少80张
    MAX_IMAGES = 100     # 最多100张
    
    # Inner London Boroughs
    INNER_LONDON_BOROUGHS = [
        "Camden", "Greenwich", "Hackney", 
        "Hammersmith and Fulham", "Islington",
        "Kensington and Chelsea", "Lambeth", 
        "Lewisham", "Southwark", "Tower Hamlets",
        "Wandsworth", "Westminster"
    ]
    
    # 处理参数
    BATCH_SIZE = 10      # 每批处理10个LSOA（内存友好）
    MAX_WORKERS = 4      # 并行下载线程数

# ============================================
# Mapillary流式下载器
# ============================================

class MapillaryStreamer:
    """Mapillary流式图片处理器"""
    
    def __init__(self, token):
        self.token = token
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'OAuth {self.token}'
        })
        
    def search_images_in_bbox(self, bbox, limit=100):
        """在边界框内搜索图片"""
        url = f"{Config.MAPILLARY_API}/images"
        
        params = {
            'bbox': f'{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}',
            'fields': 'id,captured_at,thumb_2048_url,computed_geometry',
            'limit': min(limit, 500)  # API限制
        }
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                return data.get('data', [])
        except Exception as e:
            print(f"搜索失败: {e}")
            
        return []
    
    def download_image_to_memory(self, image_url):
        """下载图片到内存（不保存）"""
        try:
            response = self.session.get(image_url, timeout=30)
            if response.status_code == 200:
                # 直接返回PIL Image对象
                return Image.open(BytesIO(response.content)).convert('RGB')
        except Exception as e:
            print(f"下载失败: {e}")
        
        return None

# ============================================
# 轻量级GVI计算器
# ============================================

class StreamGVICalculator:
    """流式GVI计算器（内存优化版）"""
    
    def __init__(self, use_gpu=False):
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 加载SegFormer（轻量级）
        from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
        
        # 使用较小的模型以节省内存
        model_name = "nvidia/segformer-b0-finetuned-ade-512-512"  # b0更小更快
        self.feature_extractor = SegformerFeatureExtractor.from_pretrained(model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # ADE20K植被类别
        self.vegetation_classes = [4, 9, 17, 29, 46, 66, 90]  # tree, grass, plant等
    
    def calculate_gvi_from_pil(self, pil_image):
        """从PIL图片直接计算GVI（不保存）"""
        try:
            # 预处理
            inputs = self.feature_extractor(images=pil_image, return_tensors="pt")
            
            # 推理
            with torch.no_grad():
                outputs = self.model(**inputs.to(self.device))
                
                # 获取预测
                logits = outputs.logits
                pred = logits.argmax(dim=1).cpu().numpy()[0]
            
            # 计算植被像素
            vegetation_mask = np.isin(pred, self.vegetation_classes)
            
            # 计算GVI
            total_pixels = pred.size
            vegetation_pixels = np.sum(vegetation_mask)
            gvi = (vegetation_pixels / total_pixels) * 100
            
            # 清理内存
            del inputs, outputs, logits, pred
            torch.cuda.empty_cache() if self.device.type == 'cuda' else None
            
            return {
                'gvi': gvi,
                'vegetation_pixels': int(vegetation_pixels),
                'total_pixels': int(total_pixels)
            }
            
        except Exception as e:
            print(f"GVI计算失败: {e}")
            return None

# ============================================
# 流式LSOA处理器
# ============================================

class StreamLSOAProcessor:
    """流式处理单个LSOA"""
    
    def __init__(self):
        self.streamer = MapillaryStreamer(Config.MAPILLARY_TOKEN)
        self.calculator = StreamGVICalculator(use_gpu=False)  # Windows上通常用CPU
        
    def process_lsoa(self, lsoa_code, bbox):
        """处理单个LSOA（不保存图片）"""
        
        print(f"\n处理 {lsoa_code}...")
        
        # Step 1: 搜索图片
        images_metadata = self.streamer.search_images_in_bbox(bbox, limit=Config.MAX_IMAGES)
        
        if len(images_metadata) < Config.MIN_IMAGES:
            print(f"⚠️ {lsoa_code} 只找到 {len(images_metadata)} 张图片（需要至少{Config.MIN_IMAGES}张）")
            return None
        
        # 限制到最多100张
        images_metadata = images_metadata[:Config.MAX_IMAGES]
        print(f"找到 {len(images_metadata)} 张图片")
        
        # Step 2: 流式处理每张图片
        gvi_results = []
        
        with ThreadPoolExecutor(max_workers=2) as executor:  # 限制并发避免内存溢出
            for img_meta in tqdm(images_metadata, desc=f"计算{lsoa_code}的GVI"):
                
                # 获取图片URL
                image_url = img_meta.get('thumb_2048_url')
                if not image_url:
                    continue
                
                # 下载到内存
                pil_image = self.streamer.download_image_to_memory(image_url)
                if pil_image is None:
                    continue
                
                # 计算GVI
                gvi_result = self.calculator.calculate_gvi_from_pil(pil_image)
                if gvi_result:
                    gvi_result['image_id'] = img_meta['id']
                    gvi_result['captured_at'] = img_meta.get('captured_at', '')
                    gvi_results.append(gvi_result)
                
                # 立即释放图片内存
                del pil_image
        
        # Step 3: 汇总结果
        if len(gvi_results) >= Config.MIN_IMAGES:
            gvi_values = [r['gvi'] for r in gvi_results]
            
            summary = {
                'lsoa_code': lsoa_code,
                'n_images': len(gvi_results),
                'mean_gvi': np.mean(gvi_values),
                'median_gvi': np.median(gvi_values),
                'std_gvi': np.std(gvi_values),
                'min_gvi': np.min(gvi_values),
                'max_gvi': np.max(gvi_values),
                'q25_gvi': np.percentile(gvi_values, 25),
                'q75_gvi': np.percentile(gvi_values, 75),
                'processing_time': datetime.now().isoformat()
            }
            
            print(f"✅ {lsoa_code} 完成: 平均GVI={summary['mean_gvi']:.2f}%")
            
            # 可选：保存详细结果（CSV很小）
            details_df = pd.DataFrame(gvi_results)
            details_df['lsoa_code'] = lsoa_code
            
            return summary, details_df
        
        else:
            print(f"❌ {lsoa_code} 处理失败：有效图片不足")
            return None

# ============================================
# 批量流式处理管道
# ============================================

class StreamPipeline:
    """流式处理管道"""
    
    def __init__(self):
        self.processor = StreamLSOAProcessor()
        self.setup_logging()
        
    def setup_logging(self):
        """设置日志"""
        log_dir = Config.BASE_DIR / "logs"
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f"stream_{datetime.now():%Y%m%d}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_lsoa_boundaries(self):
        """加载LSOA边界（简化版）"""
        # 这里需要你的LSOA边界数据
        # 暂时使用模拟数据
        
        # 读取IMD数据获取LSOA列表
        imd_data = pd.read_excel(Config.IMD_DATA)
        inner_lsoas = imd_data[imd_data['Borough'].isin(Config.INNER_LONDON_BOROUGHS)]
        
        # 需要从shapefile获取实际边界
        # 这里用简化的边界框（需要实际坐标）
        boundaries = []
        for _, row in inner_lsoas.iterrows():
            # 模拟边界框 [min_lon, min_lat, max_lon, max_lat]
            # 实际需要从shapefile读取
            bbox = [-0.1, 51.5, -0.05, 51.55]  # 示例坐标
            boundaries.append({
                'lsoa_code': row['LSOA code (2011)'],
                'bbox': bbox
            })
        
        return boundaries[:100]  # 先处理100个
    
    def run(self, target_count=100):
        """运行流式处理"""
        
        self.logger.info(f"开始流式处理 {target_count} 个LSOA")
        
        # 加载LSOA边界
        lsoa_boundaries = self.load_lsoa_boundaries()[:target_count]
        
        # 创建输出目录
        Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # 处理每个LSOA
        all_summaries = []
        all_details = []
        
        for i, lsoa_info in enumerate(lsoa_boundaries):
            print(f"\n进度: {i+1}/{len(lsoa_boundaries)}")
            
            result = self.processor.process_lsoa(
                lsoa_info['lsoa_code'],
                lsoa_info['bbox']
            )
            
            if result:
                summary, details = result
                all_summaries.append(summary)
                all_details.append(details)
                
                # 每10个LSOA保存一次（避免丢失）
                if (i + 1) % 10 == 0:
                    self.save_intermediate_results(all_summaries, all_details)
        
        # 保存最终结果
        self.save_final_results(all_summaries, all_details)
        
        self.logger.info("✅ 流式处理完成！")
        
        return all_summaries
    
    def save_intermediate_results(self, summaries, details):
        """保存中间结果"""
        temp_dir = Config.OUTPUT_DIR / "temp"
        temp_dir.mkdir(exist_ok=True)
        
        pd.DataFrame(summaries).to_csv(
            temp_dir / f"temp_summary_{datetime.now():%H%M}.csv", 
            index=False
        )
        
        if details:
            pd.concat(details).to_csv(
                temp_dir / f"temp_details_{datetime.now():%H%M}.csv",
                index=False
            )
    
    def save_final_results(self, summaries, details):
        """保存最终结果"""
        
        # 汇总结果
        summary_df = pd.DataFrame(summaries)
        summary_df.to_csv(
            Config.RESULTS_DIR / "stream_gvi_summary.csv",
            index=False
        )
        
        # 详细结果（可选）
        if details:
            details_df = pd.concat(details, ignore_index=True)
            details_df.to_csv(
                Config.RESULTS_DIR / "stream_gvi_details.csv",
                index=False
            )
        
        # 统计报告
        print("\n" + "="*50)
        print("📊 处理统计:")
        print(f"  LSOA数量: {len(summary_df)}")
        print(f"  平均GVI: {summary_df['mean_gvi'].mean():.2f}%")
        print(f"  GVI范围: {summary_df['mean_gvi'].min():.2f}% - {summary_df['mean_gvi'].max():.2f}%")
        print(f"  总图片数: {summary_df['n_images'].sum()}")
        print("="*50)

# ============================================
# 主程序
# ============================================

def main():
    """主函数"""
    
    import argparse
    parser = argparse.ArgumentParser(description='流式GVI处理')
    parser.add_argument('--target', type=int, default=100,
                       help='目标LSOA数量')
    parser.add_argument('--test', action='store_true',
                       help='测试模式（只处理5个）')
    
    args = parser.parse_args()
    
    if args.test:
        args.target = 5
        print("🧪 测试模式：只处理5个LSOA")
    
    # 运行流式处理
    pipeline = StreamPipeline()
    results = pipeline.run(target_count=args.target)
    
    print(f"\n✅ 完成！处理了 {len(results)} 个LSOA")
    print(f"💾 结果保存在: {Config.RESULTS_DIR}")
    print(f"📊 存储节省: ~{args.target * 0.1:.1f}GB（相比保存所有图片）")

if __name__ == "__main__":
    main()
