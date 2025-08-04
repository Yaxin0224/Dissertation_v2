#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
City Look Dissertation v2
03_gvi_calculation.py - 绿视率(GVI)计算（Python版本）

使用更适合植被检测的深度学习模型
输出CSV格式结果，便于R项目集成

作者：[您的名字]
日期：2025-01-14
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# 深度学习相关
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import cv2

# 进度条
from tqdm import tqdm

# 设置随机种子
np.random.seed(42)
torch.manual_seed(42)

class GVICalculator:
    """GVI计算器类"""
    
    def __init__(self, model_type='segformer', device=None):
        """
        初始化GVI计算器
        
        Args:
            model_type: 模型类型 ('segformer', 'deeplabv3_ade20k', 'hrnet')
            device: 计算设备
        """
        self.model_type = model_type
        
        # 设置设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        print(f"使用设备: {self.device}")
        
        # 加载模型
        self.model = self._load_model()
        
        # ADE20K数据集的植被类别（150个类别）
        self.vegetation_classes = {
            4: 'tree',
            9: 'grass', 
            17: 'plant',
            46: 'bush',
            66: 'flower',
            29: 'field',
            90: 'leaves'
        }
        
    def _load_model(self):
        """加载语义分割模型"""
        
        if self.model_type == 'deeplabv3_ade20k':
            # 使用ADE20K预训练的DeepLabv3+
            print("加载DeepLabv3+ (ADE20K)模型...")
            
            # 注意：需要使用mmsegmentation或其他支持ADE20K的实现
            # 这里使用一个简化的实现
            model = models.segmentation.deeplabv3_resnet101(
                pretrained=False,
                num_classes=150  # ADE20K类别数
            )
            
            # 加载ADE20K预训练权重（需要下载）
            # checkpoint_url = "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/deeplabv3plus_r101-d8_512x512_160k_ade20k/deeplabv3plus_r101-d8_512x512_160k_ade20k_20200615_123232-38af86bb.pth"
            # 实际项目中需要下载并加载权重
            
        elif self.model_type == 'segformer':
            # 使用SegFormer（推荐，性能更好）
            print("加载SegFormer模型...")
            from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
            
            model_name = "nvidia/segformer-b2-finetuned-ade-512-512"
            self.feature_extractor = SegformerFeatureExtractor.from_pretrained(model_name)
            model = SegformerForSemanticSegmentation.from_pretrained(model_name)
            
        else:
            # 默认使用带权重的DeepLabv3（COCO+植被微调）
            print("加载标准DeepLabv3+模型...")
            model = models.segmentation.deeplabv3_resnet101(
                weights='COCO_WITH_VOC_LABELS_V1'
            )
            
        model.to(self.device)
        model.eval()
        
        return model
    
    def preprocess_image(self, image_path):
        """预处理图像"""
        image = Image.open(image_path).convert('RGB')
        
        if self.model_type == 'segformer':
            # SegFormer预处理
            inputs = self.feature_extractor(images=image, return_tensors="pt")
            return inputs, image
        else:
            # 标准预处理
            preprocess = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            input_tensor = preprocess(image).unsqueeze(0)
            return input_tensor.to(self.device), image
    
    def segment_image(self, image_path):
        """对图像进行语义分割"""
        
        # 预处理
        inputs, original_image = self.preprocess_image(image_path)
        
        # 推理
        with torch.no_grad():
            if self.model_type == 'segformer':
                outputs = self.model(**inputs.to(self.device))
                
                # 上采样到原始大小
                logits = outputs.logits
                upsampled_logits = nn.functional.interpolate(
                    logits,
                    size=original_image.size[::-1],  # (height, width)
                    mode="bilinear",
                    align_corners=False
                )
                pred = upsampled_logits.argmax(dim=1).cpu().numpy()[0]
                
            else:
                outputs = self.model(inputs)['out']
                pred = outputs.argmax(1).squeeze().cpu().numpy()
        
        return pred, original_image
    
    def calculate_gvi_deeplearning(self, image_path):
        """使用深度学习计算GVI"""
        try:
            # 语义分割
            segmentation, original_image = self.segment_image(image_path)
            
            # 计算植被像素
            vegetation_mask = np.zeros_like(segmentation, dtype=bool)
            
            if self.model_type in ['segformer', 'deeplabv3_ade20k']:
                # ADE20K类别
                for class_id in self.vegetation_classes.keys():
                    vegetation_mask |= (segmentation == class_id)
            else:
                # COCO类别（只有potted plant）
                vegetation_mask = (segmentation == 63)
            
            # 计算GVI
            total_pixels = segmentation.size
            vegetation_pixels = np.sum(vegetation_mask)
            gvi = (vegetation_pixels / total_pixels) * 100
            
            return {
                'method': 'deeplearning',
                'model': self.model_type,
                'gvi': gvi,
                'vegetation_pixels': int(vegetation_pixels),
                'total_pixels': int(total_pixels),
                'segmentation': segmentation,
                'vegetation_mask': vegetation_mask
            }
            
        except Exception as e:
            print(f"深度学习处理失败: {e}")
            return None
    
    def calculate_gvi_color(self, image_path):
        """使用色彩阈值法计算GVI（备用方法）"""
        
        # 读取图像
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 转换到HSV色彩空间（更适合检测绿色）
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 定义绿色的HSV范围
        # 绿色的色调(H)大约在36-86之间
        lower_green = np.array([25, 40, 40])
        upper_green = np.array([86, 255, 255])
        
        # 创建绿色掩码
        mask_hsv = cv2.inRange(hsv, lower_green, upper_green)
        
        # 额外的RGB条件（G > R 且 G > B）
        r, g, b = image_rgb[:,:,0], image_rgb[:,:,1], image_rgb[:,:,2]
        mask_rgb = (g > r) & (g > b) & (g > 30)
        
        # 组合掩码
        vegetation_mask = mask_hsv.astype(bool) & mask_rgb
        
        # 形态学操作去噪
        kernel = np.ones((3,3), np.uint8)
        vegetation_mask = cv2.morphologyEx(vegetation_mask.astype(np.uint8), 
                                          cv2.MORPH_OPEN, kernel)
        vegetation_mask = cv2.morphologyEx(vegetation_mask, 
                                          cv2.MORPH_CLOSE, kernel)
        
        # 计算GVI
        total_pixels = image.shape[0] * image.shape[1]
        vegetation_pixels = np.sum(vegetation_mask)
        gvi = (vegetation_pixels / total_pixels) * 100
        
        return {
            'method': 'color_threshold',
            'model': 'hsv_rgb',
            'gvi': gvi,
            'vegetation_pixels': int(vegetation_pixels),
            'total_pixels': int(total_pixels),
            'vegetation_mask': vegetation_mask.astype(bool)
        }
    
    def process_image(self, image_path, fallback_to_color=True):
        """处理单张图像"""
        
        # 首先尝试深度学习方法
        result = self.calculate_gvi_deeplearning(image_path)
        
        # 如果深度学习失败或植被太少，使用色彩方法
        if result is None or (fallback_to_color and result['vegetation_pixels'] < 500):
            color_result = self.calculate_gvi_color(image_path)
            
            # 如果深度学习有结果，比较两种方法
            if result is not None:
                # 选择检测到更多植被的方法
                if color_result['vegetation_pixels'] > result['vegetation_pixels']:
                    result = color_result
                else:
                    result['fallback_attempted'] = True
            else:
                result = color_result
        
        # 添加文件信息
        result['image_path'] = image_path
        result['image_id'] = Path(image_path).stem
        result['timestamp'] = datetime.now().isoformat()
        
        return result
    
    def save_visualization(self, image_path, result, output_dir):
        """保存可视化结果"""
        
        # 读取原图
        original = cv2.imread(image_path)
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        
        # 创建可视化
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 原图
        axes[0].imshow(original_rgb)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # 植被掩码
        if 'vegetation_mask' in result:
            axes[1].imshow(result['vegetation_mask'], cmap='Greens')
            axes[1].set_title(f'Vegetation Mask ({result["method"]})')
        else:
            axes[1].text(0.5, 0.5, 'No mask available', ha='center', va='center')
        axes[1].axis('off')
        
        # 分割结果（如果有）
        if 'segmentation' in result:
            axes[2].imshow(result['segmentation'], cmap='tab20')
            axes[2].set_title('Semantic Segmentation')
        else:
            # 显示统计信息
            axes[2].text(0.1, 0.9, f'GVI: {result["gvi"]:.2f}%', 
                        transform=axes[2].transAxes, fontsize=16)
            axes[2].text(0.1, 0.7, f'Method: {result["method"]}', 
                        transform=axes[2].transAxes, fontsize=12)
            axes[2].text(0.1, 0.5, f'Vegetation pixels: {result["vegetation_pixels"]:,}', 
                        transform=axes[2].transAxes, fontsize=12)
            axes[2].text(0.1, 0.3, f'Total pixels: {result["total_pixels"]:,}', 
                        transform=axes[2].transAxes, fontsize=12)
        axes[2].axis('off')
        
        # 保存
        output_path = Path(output_dir) / f"{result['image_id']}_gvi_visualization.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(output_path)


def process_lsoa(lsoa_code, base_dir, output_dir, model_type='segformer', 
                sample_size=None, save_viz=True):
    """处理单个LSOA的所有图像"""
    
    print(f"\n处理LSOA: {lsoa_code}")
    
    # 图像目录
    image_dir = Path(base_dir) / "data/raw/mapillary_images" / lsoa_code
    
    if not image_dir.exists():
        print(f"错误：找不到LSOA目录 {image_dir}")
        return None
    
    # 获取所有图像
    image_files = list(image_dir.glob("*.jpg"))
    print(f"找到 {len(image_files)} 张图像")
    
    # 采样（如果指定）
    if sample_size and len(image_files) > sample_size:
        image_files = np.random.choice(image_files, sample_size, replace=False)
        print(f"随机采样 {sample_size} 张图像")
    
    # 初始化计算器
    calculator = GVICalculator(model_type=model_type)
    
    # 处理所有图像
    results = []
    
    for image_path in tqdm(image_files, desc="计算GVI"):
        result = calculator.process_image(str(image_path))
        result['lsoa_code'] = lsoa_code
        
        # 保存可视化（前5张）
        if save_viz and len(results) < 5:
            viz_dir = Path(output_dir) / "visualizations" / lsoa_code
            viz_dir.mkdir(parents=True, exist_ok=True)
            
            # 需要matplotlib
            global plt
            import matplotlib.pyplot as plt
            calculator.save_visualization(str(image_path), result, viz_dir)
        
        # 只保留必要字段到结果
        results.append({
            'lsoa_code': result['lsoa_code'],
            'image_id': result['image_id'],
            'gvi': result['gvi'],
            'method': result['method'],
            'model': result['model'],
            'vegetation_pixels': result['vegetation_pixels'],
            'total_pixels': result['total_pixels'],
            'timestamp': result['timestamp']
        })
    
    # 转换为DataFrame
    df = pd.DataFrame(results)
    
    # 计算汇总统计
    summary = {
        'lsoa_code': lsoa_code,
        'n_images': len(df),
        'mean_gvi': df['gvi'].mean(),
        'median_gvi': df['gvi'].median(),
        'std_gvi': df['gvi'].std(),
        'min_gvi': df['gvi'].min(),
        'max_gvi': df['gvi'].max(),
        'deeplearning_count': len(df[df['method'] == 'deeplearning']),
        'color_threshold_count': len(df[df['method'] == 'color_threshold']),
        'processing_time': datetime.now().isoformat()
    }
    
    print(f"\n完成！平均GVI: {summary['mean_gvi']:.2f}%")
    print(f"深度学习: {summary['deeplearning_count']}, 色彩阈值: {summary['color_threshold_count']}")
    
    return df, summary


def main():
    """主函数"""
    
    parser = argparse.ArgumentParser(description='计算街景图像的绿视率(GVI)')
    parser.add_argument('--lsoa', type=str, default='E01000882',
                       help='LSOA代码')
    parser.add_argument('--base-dir', type=str, 
                       default='/Users/yaxin/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Attachments/004/methodology/Dissertation_v2',
                       help='项目根目录')
    parser.add_argument('--output-dir', type=str, default='output/gvi_results',
                       help='输出目录')
    parser.add_argument('--model', type=str, default='segformer',
                       choices=['segformer', 'deeplabv3_ade20k', 'deeplabv3_coco'],
                       help='使用的模型')
    parser.add_argument('--sample-size', type=int, default=20,
                       help='每个LSOA的采样数量（None表示全部）')
    parser.add_argument('--all-lsoas', action='store_true',
                       help='处理所有LSOA')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.base_dir) / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.all_lsoas:
        # 读取所有LSOA
        lsoa_file = Path(args.base_dir) / "data/processed/selected_lsoas.csv"
        lsoas_df = pd.read_csv(lsoa_file)
        
        all_results = []
        all_summaries = []
        
        for lsoa_code in lsoas_df['lsoa_code']:
            df, summary = process_lsoa(
                lsoa_code, 
                args.base_dir, 
                output_dir,
                model_type=args.model,
                sample_size=args.sample_size
            )
            
            if df is not None:
                all_results.append(df)
                all_summaries.append(summary)
        
        # 合并结果
        final_results = pd.concat(all_results, ignore_index=True)
        final_summary = pd.DataFrame(all_summaries)
        
    else:
        # 处理单个LSOA
        final_results, summary = process_lsoa(
            args.lsoa,
            args.base_dir,
            output_dir,
            model_type=args.model,
            sample_size=args.sample_size
        )
        
        final_summary = pd.DataFrame([summary])
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 详细结果
    results_file = output_dir / f"gvi_results_{timestamp}.csv"
    final_results.to_csv(results_file, index=False)
    print(f"\n详细结果已保存到: {results_file}")
    
    # 汇总结果
    summary_file = output_dir / f"gvi_summary_{timestamp}.csv"
    final_summary.to_csv(summary_file, index=False)
    print(f"汇总结果已保存到: {summary_file}")
    
    # 为R集成创建简化版本
    r_integration_file = output_dir / "gvi_results_for_r.csv"
    final_results.to_csv(r_integration_file, index=False)
    print(f"\nR集成文件已保存到: {r_integration_file}")
    
    print("\n完成！")


if __name__ == "__main__":
    main()
