#!/usr/bin/env python3
"""
分析生成的COCO标注文件，检查8角点是否正确
"""
import json
import numpy as np
import argparse


def analyze_coco(coco_path, num_samples=5):
    """分析COCO文件中的角点分布"""
    
    with open(coco_path, 'r') as f:
        coco_data = json.load(f)
    
    print("="*70)
    print(f"COCO文件分析: {coco_path}")
    print("="*70)
    print(f"总图像数: {len(coco_data['images'])}")
    print(f"总标注数: {len(coco_data['annotations'])}")
    
    # 统计每个图像的角点数量
    corners_per_image = []
    visible_corners_per_image = []
    
    # 收集所有角点坐标用于分析分布
    all_corners_x = []
    all_corners_y = []
    
    print(f"\n前{num_samples}个标注的详细信息:")
    print("-"*70)
    
    for idx, ann in enumerate(coco_data['annotations'][:num_samples]):
        img_id = ann['image_id']
        keypoints = ann['keypoints']
        
        # 解析keypoints
        corners = []
        visible_corners = []
        
        for i in range(0, len(keypoints), 3):
            if i+2 < len(keypoints):
                x, y, v = keypoints[i], keypoints[i+1], keypoints[i+2]
                corners.append((x, y, v))
                all_corners_x.append(x)
                all_corners_y.append(y)
                
                if v == 2:
                    visible_corners.append((x, y))
        
        corners_per_image.append(len(corners))
        visible_corners_per_image.append(len(visible_corners))
        
        print(f"\n标注 {idx} (图像 {img_id}):")
        print(f"  总角点数: {len(corners)}")
        print(f"  可见角点数: {len(visible_corners)}")
        print(f"  bbox: {ann['bbox']}")
        
        # 检查角点分布
        if visible_corners:
            xs = [c[0] for c in visible_corners]
            ys = [c[1] for c in visible_corners]
            
            x_range = max(xs) - min(xs)
            y_range = max(ys) - min(ys)
            
            print(f"  X坐标范围: [{min(xs):.1f}, {max(xs):.1f}], 跨度: {x_range:.1f}")
            print(f"  Y坐标范围: [{min(ys):.1f}, {max(ys):.1f}], 跨度: {y_range:.1f}")
            
            # 如果跨度太大，说明有问题
            if x_range > 1000 or y_range > 1000:
                print(f"  ⚠️ 警告: 角点分布跨度太大，可能存在单位转换问题!")
        
        # 打印所有角点
        print(f"  角点坐标:")
        for i, (x, y, v) in enumerate(corners):
            status = "可见" if v == 2 else "不可见"
            print(f"    {i}: ({x:10.2f}, {y:10.2f}) - {status}")
    
    # 全局统计
    print("\n" + "="*70)
    print("全局统计:")
    print("="*70)
    
    if all_corners_x and all_corners_y:
        print(f"所有角点X范围: [{min(all_corners_x):.1f}, {max(all_corners_x):.1f}]")
        print(f"所有角点Y范围: [{min(all_corners_y):.1f}, {max(all_corners_y):.1f}]")
        
        # 检查是否有异常值
        x_outliers = [x for x in all_corners_x if x < -10000 or x > 10000]
        y_outliers = [y for y in all_corners_y if y < -10000 or y > 10000]
        
        if x_outliers or y_outliers:
            print(f"\n⚠️ 警告: 发现异常值!")
            print(f"  X异常值数量: {len(x_outliers)}")
            print(f"  Y异常值数量: {len(y_outliers)}")
    
    print(f"\n每张图像的平均角点数: {np.mean(corners_per_image):.1f}")
    print(f"每张图像的平均可见角点数: {np.mean(visible_corners_per_image):.1f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='分析生成的COCO标注')
    parser.add_argument('--coco_path', type=str, required=True,
                        help='COCO标注文件路径')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='分析的样本数量')
    
    args = parser.parse_args()
    
    analyze_coco(args.coco_path, args.num_samples)
