#!/usr/bin/env python3
"""
测试角点提取逻辑
验证ignore字段处理是否正确
"""

import json
import numpy as np

def test_corner_extraction():
    """测试角点提取逻辑"""

    # 创建测试COCO数据
    test_coco_data = {
        "images": [
            {"id": 1, "file_name": "rgb/test001.png", "width": 640, "height": 480},
            {"id": 2, "file_name": "rgb/test002.png", "width": 640, "height": 480},
            {"id": 3, "file_name": "rgb/test003.png", "width": 640, "height": 480}
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
            },
            {
                "id": 4,
                "image_id": 3,
                "category_id": 1,
                "bbox": [200, 200, 100, 80],
                "keypoints": [200, 200, 2, 300, 200, 2, 300, 280, 2, 200, 280, 2],  # 4个角点
                "ignore": False  # 没有被忽略
            }
        ]
    }

    # 创建img_id到标注的映射
    img_id_to_anns = {}
    for ann in test_coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_id_to_anns:
            img_id_to_anns[img_id] = []
        img_id_to_anns[img_id].append(ann)

    print("=== 测试角点提取逻辑 ===")

    # 测试每个图像的角点提取
    for img_info in test_coco_data['images']:
        img_id = img_info['id']
        img_filename = img_info['file_name']
        anns = img_id_to_anns.get(img_id, [])

        print(f"\n图像 {img_id} ({img_filename}):")
        print(f"  标注数量: {len(anns)}")

        # 提取角点坐标 (使用与train_loader_bop.py相同的逻辑)
        corners = []
        for ann in anns:
            # 跳过被忽略的标注
            if ann.get('ignore', False):
                print(f"  跳过被忽略的标注 ID {ann['id']}")
                continue

            if 'keypoints' in ann:
                # keypoints格式: [x1,y1,v1, x2,y2,v2, ...]
                keypoints = ann['keypoints']
                # 提取可见的角点 (v=2表示可见)
                visible_corners = []
                for i in range(0, len(keypoints), 3):
                    x, y, v = keypoints[i:i+3]
                    if v > 0:  # 可见或不可见但标注的点
                        visible_corners.append([x, y])
                corners.extend(visible_corners)
                print(f"  从标注 ID {ann['id']} 提取了 {len(visible_corners)} 个角点")

        # 如果没有keypoints，尝试从bbox计算角点
        if not corners and anns:
            for ann in anns:
                # 跳过被忽略的标注
                if ann.get('ignore', False):
                    continue

                if 'bbox' in ann:
                    x, y, w, h = ann['bbox']
                    # 计算bbox的四个角点
                    bbox_corners = [
                        [x, y],          # 左上
                        [x + w, y],      # 右上
                        [x + w, y + h],  # 右下
                        [x, y + h]       # 左下
                    ]
                    corners.extend(bbox_corners)
                    print(f"  从标注 ID {ann['id']} 的bbox计算了 {len(bbox_corners)} 个角点")

        print(f"  总角点数量: {len(corners)}")
        if corners:
            print(f"  角点坐标: {corners}")

if __name__ == "__main__":
    test_corner_extraction()