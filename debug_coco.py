#!/usr/bin/env python3
"""
调试COCO标注文件内容
检查角点标注是否正确
支持使用示例数据进行演示
"""

import json
import os
import sys

def create_sample_coco_data():
    """创建示例COCO数据用于演示"""
    return {
        "images": [
            {"id": 1, "file_name": "rgb/000001.png", "width": 640, "height": 480},
            {"id": 2, "file_name": "rgb/000002.png", "width": 640, "height": 480},
            {"id": 3, "file_name": "rgb/000003.png", "width": 640, "height": 480}
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [100, 100, 200, 150],
                "keypoints": [100, 100, 2, 300, 100, 2, 300, 250, 2, 100, 250, 2],  # 4个角点
                "ignore": False
            },
            {
                "id": 2,
                "image_id": 2,
                "category_id": 1,
                "bbox": [150, 120, 180, 140],
                "keypoints": [150, 120, 2, 330, 120, 2, 330, 260, 2, 150, 260, 2],  # 4个角点
                "ignore": False
            },
            {
                "id": 3,
                "image_id": 3,
                "category_id": 1,
                "bbox": [80, 90, 220, 160],
                "keypoints": [],  # 空keypoints
                "ignore": True  # 被忽略的标注
            }
        ]
    }

def debug_coco_annotations(scene_dir=None):
    """调试COCO标注文件"""

    if scene_dir:
        coco_path = os.path.join(scene_dir, 'scene_gt_coco.json')

        if not os.path.exists(coco_path):
            print(f"错误: COCO文件不存在: {coco_path}")
            print("将使用示例数据进行演示...")
            coco_data = create_sample_coco_data()
        else:
            print(f"加载COCO文件: {coco_path}")
            with open(coco_path, 'r') as f:
                coco_data = json.load(f)
    else:
        print("未提供scene_dir，将使用示例数据进行演示...")
        coco_data = create_sample_coco_data()

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
                print(f"    ignore: {ann.get('ignore', False)}")  # 显示ignore字段
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
                    if key not in ['id', 'image_id', 'category_id', 'keypoints', 'bbox', 'area', 'iscrowd', 'segmentation', 'ignore']:
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

    print("=== 统计信息 ===")
    print(f"总标注数: {len(annotations)}")
    print(f"有keypoints的标注: {total_keypoints}")
    print(f"有bbox的标注: {total_bbox}")

    # 检查有多少标注没有被忽略
    non_ignored_annotations = 0
    for ann in annotations:
        if not ann.get('ignore', False):
            non_ignored_annotations += 1

    print(f"没有被忽略的标注: {non_ignored_annotations}")
    print(f"被忽略的标注: {len(annotations) - non_ignored_annotations}")

    # 检查前几个没有被忽略的标注
    print("\n=== 检查没有被忽略的标注 ===")
    count = 0
    for ann in annotations:
        if not ann.get('ignore', False):
            img_id = ann['image_id']
            img_info = next((img for img in coco_data['images'] if img['id'] == img_id), None)
            if img_info:
                print(f"标注 ID {ann['id']} (图像 {img_info['file_name']}):")
                if 'bbox' in ann:
                    print(f"  bbox: {ann['bbox']}")
                if 'keypoints' in ann and ann['keypoints']:
                    print(f"  keypoints: {len(ann['keypoints'])//3} 个点")
                count += 1
                if count >= 5:  # 只显示前5个
                    break

    if count == 0:
        print("没有找到没有被忽略的标注！")

if __name__ == "__main__":
    if len(sys.argv) == 2:
        scene_dir = sys.argv[1]
        debug_coco_annotations(scene_dir)
    else:
        print("用法: python3 debug_coco.py <scene_dir>")
        print("或者不提供参数使用示例数据演示")
        debug_coco_annotations()