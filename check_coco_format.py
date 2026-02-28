#!/usr/bin/env python3
"""
检查COCO标注格式，诊断角点数量问题
"""
import json
import sys
import os

def check_coco_format(coco_path):
    """检查COCO标注文件的格式和内容"""
    print(f"\n{'='*60}")
    print(f"检查COCO标注文件: {coco_path}")
    print('='*60)
    
    if not os.path.exists(coco_path):
        print(f"❌ 文件不存在: {coco_path}")
        return
    
    with open(coco_path, 'r') as f:
        data = json.load(f)
    
    print(f"\n基本统计:")
    print(f"  - 图像数量: {len(data.get('images', []))}")
    print(f"  - 标注数量: {len(data.get('annotations', []))}")
    
    # 检查前5个标注的详细信息
    print(f"\n前5个标注详情:")
    for i, ann in enumerate(data.get('annotations', [])[:5]):
        print(f"\n  Annotation {i+1} (ID: {ann.get('id')}):")
        print(f"    - image_id: {ann.get('image_id')}")
        print(f"    - num_keypoints: {ann.get('num_keypoints')}")
        
        keypoints = ann.get('keypoints', [])
        print(f"    - keypoints长度: {len(keypoints)}")
        
        if keypoints:
            # 解析keypoints
            corners = []
            for j in range(0, len(keypoints), 3):
                if j+2 < len(keypoints):
                    x, y, v = keypoints[j], keypoints[j+1], keypoints[j+2]
                    corners.append((x, y, v))
            
            print(f"    - 解析出的角点数量: {len(corners)}")
            for k, (x, y, v) in enumerate(corners[:10]):  # 只显示前10个
                print(f"      角点{k+1}: ({x:.1f}, {y:.1f}), v={v}")
        
        bbox = ann.get('bbox', [])
        print(f"    - bbox: {bbox}")
    
    # 统计所有标注的角点数量分布
    print(f"\n角点数量分布:")
    corner_counts = {}
    for ann in data.get('annotations', []):
        num = ann.get('num_keypoints', 0)
        corner_counts[num] = corner_counts.get(num, 0) + 1
    
    for num in sorted(corner_counts.keys()):
        count = corner_counts[num]
        pct = count * 100 / len(data.get('annotations', [1]))
        print(f"  {num}个角点: {count}个标注 ({pct:.1f}%)")
    
    # 检查是否有8个角点的标注
    if 8 in corner_counts:
        print(f"\n✅ 找到{corner_counts[8]}个有8个角点的标注")
    else:
        print(f"\n⚠️ 没有找到8个角点的标注！")
    
    print(f"\n{'='*60}\n")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        coco_path = sys.argv[1]
    else:
        coco_path = '/nas2/home/qianqian/projects/HCCEPose/demo-bin-pick-back/train_pbr/000000/scene_gt_coco.json'
    
    check_coco_format(coco_path)
