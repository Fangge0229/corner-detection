#!/usr/bin/env python3
"""
检查数据加载器实际加载的角点数量
"""
import sys
import os
sys.path.insert(0, '/Users/qianqian/Desktop/corner-detection')

from train_loader_bop import BOPCornerDataset
from torchvision import transforms
import numpy as np

def check_loader_corners(scene_dir, num_samples=10):
    """检查数据加载器加载的角点数量"""
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = BOPCornerDataset(scene_dir, transform=transform, phase='train')
    
    print(f"\n{'='*60}")
    print(f"检查数据加载器 - 共{len(dataset)}个样本")
    print('='*60)
    
    corner_counts = []
    
    for i in range(min(num_samples, len(dataset))):
        data = dataset[i]
        corners = data['corners']
        heatmap = data['heatmap'].squeeze(0).numpy()
        
        num_corners = len(corners)
        corner_counts.append(num_corners)
        
        # 从heatmap检测峰值数量
        from scipy.ndimage import maximum_filter
        local_max = maximum_filter(heatmap, size=10) == heatmap
        peaks = np.argwhere((heatmap > 0.3) & local_max)
        num_peaks = len(peaks)
        
        print(f"\n样本 {i}:")
        print(f"  - 角点数量: {num_corners}")
        print(f"  - Heatmap峰值数: {num_peaks}")
        print(f"  - Heatmap最大值: {heatmap.max():.4f}")
        
        if num_corners > 0:
            print(f"  - 角点坐标 (前5个):")
            for j, (x, y) in enumerate(corners[:5]):
                print(f"    角点{j+1}: ({x:.1f}, {y:.1f})")
    
    print(f"\n{'='*60}")
    print(f"统计:")
    print(f"  - 最少角点: {min(corner_counts)}")
    print(f"  - 最多角点: {max(corner_counts)}")
    print(f"  - 平均角点: {sum(corner_counts)/len(corner_counts):.1f}")
    print('='*60)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        scene_dir = sys.argv[1]
    else:
        scene_dir = '/nas2/home/qianqian/projects/HCCEPose/demo-bin-picking/train_pbr/000000'
    
    check_loader_corners(scene_dir)
