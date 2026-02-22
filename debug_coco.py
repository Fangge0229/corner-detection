#!/usr/bin/env python3
"""
调试COCO标注文件内容
检查角点标注是否正确
"""

import json
import os
import sys

def debug_coco_annotations(scene_dir):
    """调试COCO标注文件"""

    coco_path = os.path.join(scene_dir, 'scene_gt_coco.json')

    if not os.path.exists(coco_path):
        print(f"错误: COCO文件不存在: {coco_path}")
        return

    print(f"加载COCO文件: {coco_path}")

    with open(coco_path, 'r') as f:
        coco_data = json.load(f)

    print(f"图像数量: {len(coco_data.get('images', []))}")
    print(f"标注数量: {len(coco_data.get('annotations', []))}")

    # 检查前5个图像
    images = coco_data.get('images', [])[:5]
    annotations = coco_data.get('annotations', [])

    # 创建img_id到标注的映射
    img_id_to_anns = {}
    for ann in annotations:
        img_id = ann['image_id']
        if img_id not in img_id_to_anns:
            img_id_to_anns[img_id] = []
        img_id_to_anns[img_id].append(ann)

    print("\n=== 检查前5个图像的标注 ===")
    for i, img in enumerate(images):
        img_id = img['id']
        img_filename = img['file_name']
        anns = img_id_to_anns.get(img_id, [])

        print(f"\n图像 {i+1}:")
        print(f"  ID: {img_id}")
        print(f"  文件名: {img_filename}")
        print(f"  标注数量: {len(anns)}")

        if anns:
            for j, ann in enumerate(anns):
                print(f"  标注 {j+1}:")
                if 'keypoints' in ann:
                    keypoints = ann['keypoints']
                    print(f"    keypoints: {len(keypoints)//3} 个点")
                    # 显示前几个点
                    for k in range(min(3, len(keypoints)//3)):
                        x, y, v = keypoints[k*3:(k+1)*3]
                        print(f"      点{k+1}: ({x}, {y}, v={v})")
                else:
                    print("    无keypoints字段")

                if 'bbox' in ann:
                    bbox = ann['bbox']
                    print(f"    bbox: {bbox}")
                else:
                    print("    无bbox字段")

                # 检查其他可能包含角点信息的字段
                for key, value in ann.items():
                    if key not in ['id', 'image_id', 'category_id', 'keypoints', 'bbox', 'area', 'iscrowd', 'segmentation']:
                        print(f"    {key}: {value}")
        else:
            print("  无标注")

    # 检查是否有任何标注包含角点信息
    total_keypoints = 0
    total_bbox = 0
    for ann in annotations:
        if 'keypoints' in ann and ann['keypoints']:
            total_keypoints += 1
        if 'bbox' in ann:
            total_bbox += 1

    print("
=== 统计信息 ==="    print(f"总标注数: {len(annotations)}")
    print(f"有keypoints的标注: {total_keypoints}")
    print(f"有bbox的标注: {total_bbox}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python3 debug_coco.py <scene_dir>")
        sys.exit(1)

    scene_dir = sys.argv[1]
    debug_coco_annotations(scene_dir)