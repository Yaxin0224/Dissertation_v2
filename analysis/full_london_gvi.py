#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
City Look Dissertation v2 - 完整版
full_london_gvi_complete.py - 整合现有数据并继续收集

功能：
1. 整合已有的171个LSOA结果
2. 继续收集剩余LSOA数据
3. 对n≥30的计算GVI，n<30的只记录
4. 所有结果保存到一个CSV文件
"""

import os
import sys
import json
import time
import math
import logging
import warnings
from io import BytesIO
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageFile
from tqdm import tqdm

warnings.filterwarnings("ignore")
ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    import geopandas as gpd
except Exception:
    gpd = None

try:
    import torch
except Exception:
    torch = None

# ============================================
# 配置
# ============================================

class Config:
    """项目配置"""
    BASE_DIR = Path(r"C:\Users\z1782\OneDrive - University College London\Attachments\004\methodology\Dissertation_v2")
    
    # 数据文件
    IMD_FILE = BASE_DIR / "data" / "raw" / "IMD 2019" / "IMD2019_London.xlsx"
    LSOA_BOUNDARIES = BASE_DIR / "data" / "raw" / "statistical-gis-boundaries-london" / "ESRI" / "LSOA_2011_London_gen_MHW.shp"
    
    # 已有结果文件
    EXISTING_RESULTS = BASE_DIR / "output" / "full_london_gvi" / "inner_london_gvi_results_all_rows.csv"
    
    # 输出路径
    OUTPUT_DIR = BASE_DIR / "output" / "full_london_gvi"
    MAIN_OUTPUT_FILE = OUTPUT_DIR / "inner_london_all_lsoas_status.csv"  # 主输出文件
    CHECKPOINT_FILE = OUTPUT_DIR / "processing_checkpoint.json"
    DETAIL_DIR = OUTPUT_DIR / "image_details"
    
    # Mapillary
    MAPILLARY_TOKEN = os.getenv("MAPILLARY_TOKEN", "MLY|9922859457805691|cef02444f32c339cf09761b104ca4bb5")
    MAPILLARY_API = "https://graph.mapillary.com"
    
    # Inner London Boroughs
    INNER_LONDON_BOROUGHS = [
        "Camden", "Greenwich", "Hackney", "Hammersmith and Fulham",
        "Islington", "Kensington and Chelsea", "Lambeth", "Lewisham",
        "Southwark", "Tower Hamlets", "Wandsworth", "Westminster"
    ]
    
    # 处理参数
    MIN_IMAGES_FOR_GVI = 30  # 最少30张才计算GVI
    TARGET_IMAGES = 80        # 目标图片数
    MAX_IMAGES = 100          # 最多处理图片数
    
    # 批处理
    BATCH_SIZE = 50           # 每批处理50个LSOA
    MAX_WORKERS = 4           # 并行线程数
    
    # 模型参数
    MODEL_SIZE = os.getenv("SEGFORMER_SIZE", "b1" if (torch and torch.cuda.is_available()) else "b0")
    
    # 搜索策略
    SEARCH_RADIUS_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.5]  # 边界扩展比例
    TIME_RANGES = [
        ("2022-01-01", "2025-12-31"),
        ("2020-01-01", "2025-12-31"),
        ("2018-01-01", "2025-12-31"),
        ("2015-01-01", "2025-12-31"),
        (None, None)
    ]
    
    # 列名候选（用于读取IMD数据）
    IMD_LSOA_CODE_CANDS = ["LSOA code (2011)", "LSOA11CD", "lsoa11cd", "LSOA code", "LSOA_CODE"]
    IMD_LSOA_NAME_CANDS = ["LSOA name (2011)", "LSOA11NM", "lsoa11nm", "LSOA name"]
    IMD_SCORE_CANDS = ["Index of Multiple Deprivation (IMD) Score", "IMD Score", "imd_score"]
    IMD_BOROUGH_CANDS = ["Borough", "Local Authority District name (2019)", "Local Authority Name", "borough"]

# ============================================
# 日志设置
# ============================================

def setup_logging():
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Config.DETAIL_DIR.mkdir(parents=True, exist_ok=True)
    
    log_file = Config.OUTPUT_DIR / f"processing_{datetime.now():%Y%m%d_%H%M}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

# ============================================
# 数据管理器
# ============================================

class DataManager:
    def __init__(self, logger):
        self.logger = logger
        self.all_lsoas = None
        self.results_df = None
        self.processed_codes = set()
        
    def load_existing_results(self):
        """加载已有的LSOA结果"""
        # 优先加载主输出文件（包含所有已处理的）
        if Config.MAIN_OUTPUT_FILE.exists():
            try:
                self.results_df = pd.read_csv(Config.MAIN_OUTPUT_FILE)
                self.processed_codes = set(self.results_df['lsoa_code'].values)
                self.logger.info(f"✅ 加载已处理结果：{len(self.results_df)} 个LSOA")
                return True
            except Exception as e:
                self.logger.warning(f"无法加载主文件：{e}")
        
        # 如果主文件不存在，尝试加载原始171个
        if Config.EXISTING_RESULTS.exists():
            try:
                self.results_df = pd.read_csv(Config.EXISTING_RESULTS)
                self.results_df['status'] = 'completed'  # 标记为已完成
                self.processed_codes = set(self.results_df['lsoa_code'].values)
                self.logger.info(f"✅ 加载原始结果：{len(self.results_df)} 个LSOA")
                
                # 保存到主输出文件
                self.results_df.to_csv(Config.MAIN_OUTPUT_FILE, index=False)
                return True
            except Exception as e:
                self.logger.error(f"加载已有结果失败：{e}")
        
        self.logger.info("未找到已有结果文件，从头开始")
        self.results_df = pd.DataFrame()
        return False
    
    @staticmethod
    def _pick_col(df, cands):
        """从候选列名中选择存在的列"""
        for c in cands:
            if c in df.columns:
                return c
        low = [c.lower() for c in df.columns]
        for c in cands:
            for i, name in enumerate(low):
                if c.lower() == name:
                    return df.columns[i]
        return None
    
    def load_all_lsoas(self):
        """加载所有Inner London LSOAs"""
        self.logger.info("加载所有Inner London LSOAs...")
        
        if not Config.IMD_FILE.exists():
            raise FileNotFoundError(f"未找到IMD文件：{Config.IMD_FILE}")
        
        imd = pd.read_excel(Config.IMD_FILE)
        
        # 找到正确的列名
        code_col = self._pick_col(imd, Config.IMD_LSOA_CODE_CANDS)
        name_col = self._pick_col(imd, Config.IMD_LSOA_NAME_CANDS)
        score_col = self._pick_col(imd, Config.IMD_SCORE_CANDS)
        borough_col = self._pick_col(imd, Config.IMD_BOROUGH_CANDS)
        
        if not all([code_col, name_col, score_col, borough_col]):
            raise KeyError("IMD表缺少必要列")
        
        # 筛选Inner London
        inner = imd[imd[borough_col].isin(Config.INNER_LONDON_BOROUGHS)].copy()
        inner = inner.rename(columns={
            code_col: "lsoa_code",
            name_col: "lsoa_name",
            score_col: "imd_score",
            borough_col: "borough"
        })
        
        # 计算IMD五分位
        inner["imd_quintile"] = pd.qcut(
            inner["imd_score"], q=5, 
            labels=["Q1_Least", "Q2", "Q3", "Q4", "Q5_Most"]
        )
        
        self.all_lsoas = inner[["lsoa_code", "lsoa_name", "borough", "imd_score", "imd_quintile"]]
        self.logger.info(f"找到 {len(self.all_lsoas)} 个Inner London LSOAs")
        
        # 计算待处理数量
        remaining = len(self.all_lsoas) - len(self.processed_codes)
        self.logger.info(f"已处理：{len(self.processed_codes)}，待处理：{remaining}")
        
        return self.all_lsoas
    
    def get_unprocessed_lsoas(self):
        """获取未处理的LSOAs"""
        unprocessed = self.all_lsoas[~self.all_lsoas['lsoa_code'].isin(self.processed_codes)]
        return unprocessed
    
    def save_result(self, result):
        """保存单个结果"""
        # 添加到DataFrame
        new_row = pd.DataFrame([result])
        
        if self.results_df is None or self.results_df.empty:
            self.results_df = new_row
        else:
            self.results_df = pd.concat([self.results_df, new_row], ignore_index=True)
        
        # 保存到文件
        self.results_df.to_csv(Config.MAIN_OUTPUT_FILE, index=False)
        self.processed_codes.add(result['lsoa_code'])
        
    def save_checkpoint(self):
        """保存检查点"""
        checkpoint = {
            'processed': list(self.processed_codes),
            'timestamp': datetime.now().isoformat(),
            'total_processed': len(self.processed_codes)
        }
        with open(Config.CHECKPOINT_FILE, 'w') as f:
            json.dump(checkpoint, f, indent=2)

# ============================================
# Mapillary API
# ============================================

class MapillaryAPI:
    def __init__(self, token, logger):
        self.token = token
        self.logger = logger
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"OAuth {token}"})
        
    def search_images(self, bbox, min_date=None, max_date=None, limit=500):
        """搜索指定区域的图片"""
        url = f"{Config.MAPILLARY_API}/images"
        params = {
            "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
            "fields": "id,captured_at,computed_geometry,thumb_2048_url,thumb_1024_url,sequence",
            "limit": limit,
            "is_pano": "false"
        }
        
        if min_date:
            params["min_captured_at"] = f"{min_date}T00:00:00Z"
        if max_date:
            params["max_captured_at"] = f"{max_date}T23:59:59Z"
        
        try:
            resp = self.session.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                return data.get('data', [])
        except Exception as e:
            self.logger.error(f"Mapillary API错误：{e}")
        
        return []
    
    def search_with_strategy(self, bbox, target=80):
        """使用多种策略搜索图片"""
        all_images = []
        seen_ids = set()
        
        # 尝试不同的时间范围和空间扩展
        for time_range in Config.TIME_RANGES:
            for expansion in Config.SEARCH_RADIUS_LEVELS:
                # 扩展边界
                expanded_bbox = self._expand_bbox(bbox, expansion)
                
                # 搜索
                images = self.search_images(
                    expanded_bbox,
                    min_date=time_range[0] if time_range else None,
                    max_date=time_range[1] if time_range else None
                )
                
                # 去重添加
                for img in images:
                    if img.get('id') not in seen_ids:
                        all_images.append(img)
                        seen_ids.add(img.get('id'))
                
                # 如果找到足够图片，返回
                if len(all_images) >= target:
                    return all_images[:target]
        
        return all_images
    
    def _expand_bbox(self, bbox, factor):
        """扩展边界框"""
        if factor == 0:
            return bbox
        
        center_lon = (bbox[0] + bbox[2]) / 2
        center_lat = (bbox[1] + bbox[3]) / 2
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        new_width = width * (1 + factor)
        new_height = height * (1 + factor)
        
        return [
            center_lon - new_width / 2,
            center_lat - new_height / 2,
            center_lon + new_width / 2,
            center_lat + new_height / 2
        ]
    
    def download_image(self, image_meta):
        """下载单张图片"""
        url = image_meta.get('thumb_2048_url') or image_meta.get('thumb_1024_url')
        if not url:
            return None
        
        try:
            resp = self.session.get(url, timeout=30)
            if resp.status_code == 200:
                return Image.open(BytesIO(resp.content)).convert('RGB')
        except Exception:
            pass
        
        return None

# ============================================
# GVI计算器
# ============================================

class GVICalculator:
    def __init__(self, model_size="b0", device=None):
        if torch is None:
            raise ImportError("需要安装PyTorch")
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        
        # 加载模型
        model_name = f"nvidia/segformer-{model_size}-finetuned-ade-512-512"
        
        try:
            from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModelForSemanticSegmentation.from_pretrained(model_name)
        except Exception as e:
            raise ImportError(f"无法加载模型：{e}")
        
        self.model.to(self.device)
        self.model.eval()
        
        # 植被类别ID
        self.vegetation_ids = {4, 9, 17, 29, 46, 66, 87, 90}
        print(f"使用设备：{self.device}")
        print(f"植被类别IDs：{sorted(self.vegetation_ids)}")
    
    @torch.no_grad()
    def calculate_gvi(self, pil_image):
        """计算单张图片的GVI"""
        try:
            # 调整大小
            if max(pil_image.size) > 512:
                pil_image.thumbnail((512, 512), Image.Resampling.LANCZOS)
            
            # 处理图片
            inputs = self.processor(images=pil_image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 预测
            outputs = self.model(**inputs)
            pred = outputs.logits.argmax(dim=1).cpu().numpy()[0]
            
            # 计算GVI
            veg_mask = np.isin(pred, list(self.vegetation_ids))
            total_pixels = pred.size
            veg_pixels = veg_mask.sum()
            
            gvi = (veg_pixels / total_pixels) * 100.0
            
            return gvi
            
        except Exception:
            return None
        finally:
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

# ============================================
# LSOA处理器
# ============================================

class LSOAProcessor:
    def __init__(self, mapillary_api, gvi_calculator, logger):
        self.mapillary = mapillary_api
        self.calculator = gvi_calculator
        self.logger = logger
    
    def process_lsoa(self, lsoa_info):
        """处理单个LSOA"""
        lsoa_code = lsoa_info['lsoa_code']
        t0 = time.time()
        
        self.logger.info(f"处理 {lsoa_code} - {lsoa_info['borough']}")
        
        # 获取边界
        bbox = self._get_bbox(lsoa_code)
        
        # 搜索图片
        images = self.mapillary.search_with_strategy(bbox, target=Config.TARGET_IMAGES)
        n_images = len(images)
        
        self.logger.info(f"  找到 {n_images} 张图片")
        
        # 根据图片数量决定处理方式
        if n_images == 0:
            # 无数据
            result = {
                'lsoa_code': lsoa_code,
                'lsoa_name': lsoa_info.get('lsoa_name', ''),
                'borough': lsoa_info['borough'],
                'imd_score': lsoa_info['imd_score'],
                'imd_quintile': str(lsoa_info['imd_quintile']),
                'status': 'no_data',
                'n_images': 0,
                'mean_gvi': None,
                'median_gvi': None,
                'std_gvi': None,
                'min_gvi': None,
                'max_gvi': None,
                'q25_gvi': None,
                'q75_gvi': None,
                'processing_time': time.time() - t0,
                'timestamp': datetime.now().isoformat()
            }
            
        elif n_images < Config.MIN_IMAGES_FOR_GVI:
            # 图片不足，不计算GVI
            result = {
                'lsoa_code': lsoa_code,
                'lsoa_name': lsoa_info.get('lsoa_name', ''),
                'borough': lsoa_info['borough'],
                'imd_score': lsoa_info['imd_score'],
                'imd_quintile': str(lsoa_info['imd_quintile']),
                'status': 'insufficient',
                'n_images': n_images,
                'mean_gvi': None,
                'median_gvi': None,
                'std_gvi': None,
                'min_gvi': None,
                'max_gvi': None,
                'q25_gvi': None,
                'q75_gvi': None,
                'processing_time': time.time() - t0,
                'timestamp': datetime.now().isoformat()
            }
            
        else:
            # 计算GVI
            gvi_values = []
            processed = 0
            
            # 使用线程池并行处理
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                for img_meta in images[:Config.MAX_IMAGES]:
                    future = executor.submit(self._process_single_image, img_meta)
                    futures.append(future)
                
                # 收集结果
                for future in as_completed(futures):
                    gvi = future.result()
                    if gvi is not None:
                        gvi_values.append(gvi)
                        processed += 1
                    
                    # 达到目标数量即可
                    if processed >= Config.TARGET_IMAGES:
                        break
            
            # 计算统计值
            if len(gvi_values) >= Config.MIN_IMAGES_FOR_GVI:
                gvi_array = np.array(gvi_values)
                result = {
                    'lsoa_code': lsoa_code,
                    'lsoa_name': lsoa_info.get('lsoa_name', ''),
                    'borough': lsoa_info['borough'],
                    'imd_score': lsoa_info['imd_score'],
                    'imd_quintile': str(lsoa_info['imd_quintile']),
                    'status': 'completed',
                    'n_images': len(gvi_values),
                    'mean_gvi': float(np.mean(gvi_array)),
                    'median_gvi': float(np.median(gvi_array)),
                    'std_gvi': float(np.std(gvi_array)),
                    'min_gvi': float(np.min(gvi_array)),
                    'max_gvi': float(np.max(gvi_array)),
                    'q25_gvi': float(np.percentile(gvi_array, 25)),
                    'q75_gvi': float(np.percentile(gvi_array, 75)),
                    'processing_time': time.time() - t0,
                    'timestamp': datetime.now().isoformat()
                }
                self.logger.info(f"  ✅ GVI = {result['mean_gvi']:.2f}%")
            else:
                # 处理失败
                result = {
                    'lsoa_code': lsoa_code,
                    'lsoa_name': lsoa_info.get('lsoa_name', ''),
                    'borough': lsoa_info['borough'],
                    'imd_score': lsoa_info['imd_score'],
                    'imd_quintile': str(lsoa_info['imd_quintile']),
                    'status': 'insufficient',
                    'n_images': n_images,
                    'mean_gvi': None,
                    'median_gvi': None,
                    'std_gvi': None,
                    'min_gvi': None,
                    'max_gvi': None,
                    'q25_gvi': None,
                    'q75_gvi': None,
                    'processing_time': time.time() - t0,
                    'timestamp': datetime.now().isoformat()
                }
        
        return result
    
    def _process_single_image(self, img_meta):
        """处理单张图片"""
        try:
            # 下载图片
            img = self.mapillary.download_image(img_meta)
            if img is None:
                return None
            
            # 计算GVI
            gvi = self.calculator.calculate_gvi(img)
            
            # 清理
            img.close()
            
            return gvi
            
        except Exception:
            return None
    
    def _get_bbox(self, lsoa_code):
        """获取LSOA边界框"""
        try:
            if gpd and Config.LSOA_BOUNDARIES.exists():
                gdf = gpd.read_file(Config.LSOA_BOUNDARIES)
                gdf = gdf.to_crs(epsg=4326)
                
                lsoa_row = gdf[gdf['LSOA11CD'] == lsoa_code]
                if not lsoa_row.empty:
                    bounds = lsoa_row.total_bounds
                    return [bounds[0], bounds[1], bounds[2], bounds[3]]
        except Exception:
            pass
        
        # 默认边界（伦敦中心）
        return [-0.15, 51.48, -0.05, 51.54]

# ============================================
# 主处理管道
# ============================================

class MainPipeline:
    def __init__(self):
        self.logger = setup_logging()
        self.data_manager = DataManager(self.logger)
        self.mapillary = MapillaryAPI(Config.MAPILLARY_TOKEN, self.logger)
        
        # 初始化GVI计算器（如果可用）
        try:
            self.calculator = GVICalculator(model_size=Config.MODEL_SIZE)
            self.gvi_enabled = True
        except Exception as e:
            self.logger.warning(f"GVI计算器初始化失败：{e}")
            self.logger.warning("将只记录图片数量，不计算GVI")
            self.calculator = None
            self.gvi_enabled = False
        
        self.processor = LSOAProcessor(self.mapillary, self.calculator, self.logger)
    
    def run(self, batch_size=50, max_lsoas=None):
        """运行处理流程"""
        self.logger.info("=" * 70)
        self.logger.info("Inner London LSOA 数据收集")
        self.logger.info("=" * 70)
        
        # 加载已有结果
        self.data_manager.load_existing_results()
        
        # 加载所有LSOA
        self.data_manager.load_all_lsoas()
        
        # 获取未处理的LSOA
        unprocessed = self.data_manager.get_unprocessed_lsoas()
        
        if max_lsoas:
            unprocessed = unprocessed.head(max_lsoas)
            self.logger.info(f"限制处理数量：{max_lsoas}")
        
        total_to_process = len(unprocessed)
        
        if total_to_process == 0:
            self.logger.info("所有LSOA已处理完成！")
            return
        
        self.logger.info(f"待处理：{total_to_process} 个LSOA")
        self.logger.info(f"批次大小：{batch_size}")
        
        # 分批处理
        processed_count = 0
        
        for batch_start in range(0, total_to_process, batch_size):
            batch_end = min(batch_start + batch_size, total_to_process)
            batch = unprocessed.iloc[batch_start:batch_end]
            
            self.logger.info(f"\n批次 {batch_start//batch_size + 1}：处理 {batch_start+1}-{batch_end}/{total_to_process}")
            
            for _, lsoa_info in batch.iterrows():
                processed_count += 1
                
                self.logger.info(f"\n[{processed_count}/{total_to_process}] {lsoa_info['lsoa_code']}")
                
                try:
                    # 处理LSOA
                    result = self.processor.process_lsoa(lsoa_info)
                    
                    # 保存结果
                    self.data_manager.save_result(result)
                    
                    # 定期保存检查点
                    if processed_count % 10 == 0:
                        self.data_manager.save_checkpoint()
                        self._report_progress()
                        
                except Exception as e:
                    self.logger.error(f"处理失败 {lsoa_info['lsoa_code']}: {e}")
                    # 记录失败
                    failed_result = {
                        'lsoa_code': lsoa_info['lsoa_code'],
                        'lsoa_name': lsoa_info.get('lsoa_name', ''),
                        'borough': lsoa_info['borough'],
                        'imd_score': lsoa_info['imd_score'],
                        'imd_quintile': str(lsoa_info['imd_quintile']),
                        'status': 'error',
                        'n_images': 0,
                        'mean_gvi': None,
                        'median_gvi': None,
                        'std_gvi': None,
                        'min_gvi': None,
                        'max_gvi': None,
                        'q25_gvi': None,
                        'q75_gvi': None,
                        'processing_time': 0,
                        'timestamp': datetime.now().isoformat()
                    }
                    self.data_manager.save_result(failed_result)
        
        # 最终报告
        self._final_report()
    
    def _report_progress(self):
        """报告进度"""
        df = self.data_manager.results_df
        if df is not None and len(df) > 0:
            completed = len(df[df['status'] == 'completed'])
            insufficient = len(df[df['status'] == 'insufficient'])
            no_data = len(df[df['status'] == 'no_data'])
            
            self.logger.info("\n--- 进度报告 ---")
            self.logger.info(f"已完成(GVI计算): {completed}")
            self.logger.info(f"数据不足(<30): {insufficient}")
            self.logger.info(f"无数据: {no_data}")
            self.logger.info(f"总计: {len(df)}")
    
    def _final_report(self):
        """生成最终报告"""
        self.logger.info("\n" + "=" * 70)
        self.logger.info("处理完成！")
        
        df = self.data_manager.results_df
        if df is not None and len(df) > 0:
            # 统计
            stats = {
                'total': len(df),
                'completed': len(df[df['status'] == 'completed']),
                'insufficient': len(df[df['status'] == 'insufficient']),
                'no_data': len(df[df['status'] == 'no_data']),
                'error': len(df[df['status'] == 'error']) if 'error' in df['status'].values else 0
            }
            
            self.logger.info(f"\n最终统计：")
            self.logger.info(f"  总LSOA数: {stats['total']}")
            self.logger.info(f"  已计算GVI: {stats['completed']} ({stats['completed']/stats['total']*100:.1f}%)")
            self.logger.info(f"  数据不足: {stats['insufficient']} ({stats['insufficient']/stats['total']*100:.1f}%)")
            self.logger.info(f"  无数据: {stats['no_data']} ({stats['no_data']/stats['total']*100:.1f}%)")
            
            # 如果有GVI数据，计算平均值
            completed_df = df[df['status'] == 'completed']
            if len(completed_df) > 0:
                mean_gvi = completed_df['mean_gvi'].mean()
                self.logger.info(f"\n平均GVI: {mean_gvi:.2f}%")
            
            # 保存统计报告
            report_file = Config.OUTPUT_DIR / f"final_report_{datetime.now():%Y%m%d_%H%M}.json"
            with open(report_file, 'w') as f:
                json.dump(stats, f, indent=2)
            
            self.logger.info(f"\n输出文件：")
            self.logger.info(f"  主数据文件: {Config.MAIN_OUTPUT_FILE}")
            self.logger.info(f"  统计报告: {report_file}")
        
        self.logger.info("=" * 70)

# ============================================
# 主程序
# ============================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Inner London LSOA GVI数据收集")
    parser.add_argument("--test", action="store_true", help="测试模式（5个LSOA）")
    parser.add_argument("--batch-size", type=int, default=50, help="每批处理的LSOA数量")
    parser.add_argument("--max-lsoas", type=int, default=None, help="最大处理LSOA数")
    
    args = parser.parse_args()
    
    if args.test:
        print("🧪 测试模式：处理5个LSOA")
        args.max_lsoas = 5
        args.batch_size = 5
    
    print(f"配置：")
    print(f"  - 批次大小: {args.batch_size}")
    print(f"  - 最大处理数: {args.max_lsoas if args.max_lsoas else '全部'}")
    print(f"  - GVI计算阈值: ≥{Config.MIN_IMAGES_FOR_GVI}张图片")
    print()
    
    # 运行
    pipeline = MainPipeline()
    pipeline.run(batch_size=args.batch_size, max_lsoas=args.max_lsoas)

if __name__ == "__main__":
    main()
