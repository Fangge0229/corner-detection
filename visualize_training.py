#!/usr/bin/env python3
"""
训练数据可视化脚本
显示原图像、Ground Truth热图以及模型预测热图
只显示 heatmap 不为零的有效样本
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
from torchvision import transforms

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from corner_detection import CornerDetectionModel
from train_loader_bop import BOPCornerDataset

def load_model(model_path, device):
    """加载训练好的模型"""
    H, W = 256, 256
    model = CornerDetectionModel(H, W)
    model = model.to(device)

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"✓ 成功加载模型: {model_path}")
    else:
        print(f"⚠ 模型文件不存在: {model_path}")
        return None

    model.eval()
    return model

def visualize_heatmaps(model, device, dataset, num_samples=5, save_dir=None):
    """
    可视化原图像和热图
    三列显示：原图像 | Ground Truth热图 | 预测热图
    只显示 heatmap 不为零的样本
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # 收集有效样本
    valid_samples = []
    print("正在扫描数据集，寻找有效样本...")
    
    for idx in range(len(dataset)):
        data_dict = dataset[idx]
        heatmap_gt = data_dict['heatmap'].squeeze(0).numpy()
        
        if heatmap_gt.max() > 0:
            valid_samples.append(idx)
            if len(valid_samples) >= num_samples:
                break
    
    print(f"✓ 找到 {len(valid_samples)} 个有效样本\n")
    
    if len(valid_samples) == 0:
        print("错误：没有找到有效样本（heatmap 不为零）")
        return

    # 创建图表
    fig, axes = plt.subplots(len(valid_samples), 3, figsize=(15, 5*len(valid_samples)))
    if len(valid_samples) == 1:
        axes = [axes]

    for row_idx, sample_idx in enumerate(valid_samples):
        data_dict = dataset[sample_idx]
        image = data_dict['image']
        heatmap_gt = data_dict['heatmap'].squeeze(0).numpy()

        # 反标准化图像
        image_np = image.permute(1, 2, 0).numpy()
        image_np = image_np * std + mean
        image_np = np.clip(image_np, 0, 1)
        image_np = (image_np * 255).astype(np.uint8)

        # 模型推理
        with torch.no_grad():
            image_input = image.unsqueeze(0).to(device)
            heatmap_pred = model(image_input)
            heatmap_pred = torch.sigmoid(heatmap_pred).squeeze(0).squeeze(0).cpu().numpy()

        row = axes[row_idx]

        # 列1：原始图像
        row[0].imshow(image_np, cmap='gray')
        row[0].set_title(f'原始图像 (样本 {sample_idx})', fontsize=12, fontweight='bold')
        row[0].axis('off')

        # 列2：Ground Truth 热图
        # 使用jet colormap更能突出峰值位置
        im1 = row[1].imshow(heatmap_gt, cmap='jet', vmin=0, vmax=1)
        row[1].set_title(f'Ground Truth 热图\n(Max: {heatmap_gt.max():.4f}, Sum: {heatmap_gt.sum():.2f})', 
                         fontsize=12, fontweight='bold')
        row[1].axis('off')
        plt.colorbar(im1, ax=row[1], fraction=0.046, pad=0.04)

        # 列3：预测热图
        im2 = row[2].imshow(heatmap_pred, cmap='jet', vmin=0, vmax=1)
        row[2].set_title(f'预测热图\n(Max: {heatmap_pred.max():.4f}, Sum: {heatmap_pred.sum():.2f})', 
                         fontsize=12, fontweight='bold')
        row[2].axis('off')
        plt.colorbar(im2, ax=row[2], fraction=0.046, pad=0.04)

        # 统计信息
        gt_sum = heatmap_gt.sum()
        pred_sum = heatmap_pred.sum()
        print(f"样本 {sample_idx}:")
        print(f"  GT热图：最大值={heatmap_gt.max():.4f}, 总和={gt_sum:.2f}, 非零像素={np.count_nonzero(heatmap_gt > 0.1)}")
        print(f"  预测：最大值={heatmap_pred.max():.4f}, 总和={pred_sum:.2f}, 非零像素={np.count_nonzero(heatmap_pred > 0.1)}")
        print()

    plt.tight_layout()

    if save_dir:
        save_path = os.path.join(save_dir, 'heatmap_visualization.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 可视化结果已保存到: {save_path}")

    plt.show()

def main():
    parser = argparse.ArgumentParser(description='训练数据热图可视化')
    parser.add_argument('--scene_dir', type=str,
                       default='/nas2/home/qianqian/projects/HCCEPose/demo-bin-pick-back/train_pbr/000000',
                       help='训练数据目录路径')
    parser.add_argument('--model_path', type=str,
                       default='./corner_detection_model_retrained.pth',
                       help='模型权重文件路径')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='可视化的有效样本数量')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='保存可视化结果的目录')

    args = parser.parse_args()

    print("=" * 60)
    print("训练数据热图可视化")
    print("=" * 60)
    print(f"数据目录: {args.scene_dir}")
    print(f"模型路径: {args.model_path}")
    print(f"样本数量: {args.num_samples}")
    print()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}\n")

    if not os.path.exists(args.scene_dir):
        print(f"❌ 错误: 数据目录不存在: {args.scene_dir}")
        return

    model = load_model(args.model_path, device)
    if model is None:
        return

    try:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        dataset = BOPCornerDataset(args.scene_dir, transform=transform)
        print(f"✓ 数据集大小: {len(dataset)} 样本\n")
    except Exception as e:
        print(f"❌ 错误: 无法创建数据集: {e}")
        return

    visualize_heatmaps(model, device, dataset, args.num_samples, args.save_dir)
    
    print("=" * 60)
    print("可视化完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()
