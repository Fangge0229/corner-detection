#!/usr/bin/env python3
"""
训练数据效果可视化脚本
加载训练好的模型，在训练数据上运行推理并可视化结果
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import os
import sys
import argparse
import json
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from corner_detection import CornerDetectionModel
from train_loader_bop import BOPCornerDataset

def load_model(model_path):
    """加载训练好的模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建模型（使用训练时的图像尺寸 256x256）
    H, W = 256, 256
    model = CornerDetectionModel(H, W)
    model = model.to(device)

    # 加载权重
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"成功加载模型: {model_path}")
    else:
        print(f"警告: 模型文件不存在: {model_path}")
        print("将使用随机初始化的模型进行演示")

    model.eval()
    return model, device

def visualize_predictions(model, device, dataset, num_samples=5, save_dir=None):
    """可视化模型预测结果"""

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    if num_samples == 1:
        axes = [axes]

    for i in range(min(num_samples, len(dataset))):
        # 获取数据
        image, heatmap_gt, corners_gt = dataset[i]

        # 模型推理
        with torch.no_grad():
            image_input = image.unsqueeze(0).to(device)
            heatmap_pred = model(image_input)
            heatmap_pred = torch.sigmoid(heatmap_pred).squeeze(0).squeeze(0).cpu().numpy()

        # 转换为numpy数组用于显示
        image_np = image.permute(1, 2, 0).numpy()
        image_np = (image_np * 255).astype(np.uint8)

        # 找到预测的角点（heatmap中的局部最大值）
        pred_corners = extract_corners_from_heatmap(heatmap_pred, threshold=0.5)

        # 显示结果
        row = axes[i]

        # 原始图像
        row[0].imshow(image_np)
        row[0].set_title(f'原始图像 {i+1}')
        row[0].axis('off')

        # Ground Truth角点
        gt_image = image_np.copy()
        if len(corners_gt) > 0:
            for corner in corners_gt:
                x, y = corner
                cv2.circle(gt_image, (int(x), int(y)), 3, (0, 255, 0), -1)  # 绿色圆圈
        row[1].imshow(gt_image)
        row[1].set_title(f'Ground Truth角点 ({len(corners_gt)}个)')
        row[1].axis('off')

        # 预测角点
        pred_image = image_np.copy()
        if len(pred_corners) > 0:
            for corner in pred_corners:
                x, y = corner
                cv2.circle(pred_image, (int(x), int(y)), 3, (255, 0, 0), -1)  # 蓝色圆圈
        row[2].imshow(pred_image)
        row[2].set_title(f'预测角点 ({len(pred_corners)}个)')
        row[2].axis('off')

        # 打印统计信息
        print(f"样本 {i+1}:")
        print(f"  Ground Truth角点: {len(corners_gt)}")
        print(f"  预测角点: {len(pred_corners)}")
        print(f"  图像尺寸: {image_np.shape}")
        print()

    plt.tight_layout()

    if save_dir:
        save_path = os.path.join(save_dir, 'training_data_visualization.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"可视化结果已保存到: {save_path}")

    plt.show()

def extract_corners_from_heatmap(heatmap, threshold=0.5, min_distance=10):
    """从heatmap中提取角点坐标"""
    # 应用阈值
    heatmap_binary = (heatmap > threshold).astype(np.uint8)

    # 找到局部最大值
    corners = []
    h, w = heatmap.shape

    for y in range(min_distance, h - min_distance):
        for x in range(min_distance, w - min_distance):
            if heatmap_binary[y, x] == 1:
                # 检查是否是局部最大值
                local_region = heatmap[y-min_distance:y+min_distance+1,
                                      x-min_distance:x+min_distance+1]
                if heatmap[y, x] == np.max(local_region):
                    corners.append((x, y))

    return corners

def main():
    parser = argparse.ArgumentParser(description='训练数据效果可视化')
    parser.add_argument('--scene_dir', type=str,
                       default='/nas2/home/qianqian/projects/corner_detection/demo-bin-picking/train_pbr/000000',
                       help='训练数据目录路径')
    parser.add_argument('--model_path', type=str,
                       default='./corner_detection_model.pth',
                       help='模型权重文件路径')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='可视化的样本数量')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='保存可视化结果的目录')

    args = parser.parse_args()

    print("=== 训练数据效果可视化 ===")
    print(f"数据目录: {args.scene_dir}")
    print(f"模型路径: {args.model_path}")
    print(f"样本数量: {args.num_samples}")
    print()

    # 检查数据目录
    if not os.path.exists(args.scene_dir):
        print(f"错误: 数据目录不存在: {args.scene_dir}")
        return

    # 加载模型
    model, device = load_model(args.model_path)

    # 创建数据集
    try:
        dataset = BOPCornerDataset(args.scene_dir)
        print(f"数据集大小: {len(dataset)} 样本")
        print()
    except Exception as e:
        print(f"错误: 无法创建数据集: {e}")
        return

    # 可视化预测结果
    visualize_predictions(model, device, dataset, args.num_samples, args.save_dir)

if __name__ == "__main__":
    main()