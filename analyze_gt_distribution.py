#!/usr/bin/env python3
"""
检查服务器上的角点标注分布情况
帮助诊断为什么很多样本只有少量角点
"""
import json
import os
import argparse
from collections import Counter

def analyze_coco_annotations(coco_path):
    """分析COCO标注文件中角点的分布情况"""

    if not os.path.exists(coco_path):
        print(f"错误: 文件不存在: {coco_path}")
        return

    with open(coco_path, 'r') as f:
        coco_data = json.load(f)

    print("=" * 60)
    print("COCO标注文件分析")
    print("=" * 60)

    # 基本统计
    num_images = len(coco_data.get('images', []))
    num_annotations = len(coco_data.get('annotations', []))

    print(f"\n基本统计:")
    print(f"  - 图像数量: {num_images}")
    print(f"  - 标注数量: {num_annotations}")

    # 分析每个图像的annotation数量
    img_id_to_anns = {}
    for ann in coco_data.get('annotations', []):
        img_id = ann['image_id']
        if img_id not in img_id_to_anns:
            img_id_to_anns[img_id] = []
        img_id_to_anns[img_id].append(ann)

    ann_per_image = [len(anns) for anns in img_id_to_anns.values()]
    print(f"\n每个图像的annotation数量分布:")
    print(f"  - 最少: {min(ann_per_image)}")
    print(f"  - 最多: {max(ann_per_image)}")
    print(f"  - 平均: {sum(ann_per_image)/len(ann_per_image):.2f}")

    # 分析每个annotation中的keypoints数量
    keypoints_count_per_ann = []
    keypoints_visibility = []

    for ann in coco_data.get('annotations', []):
        if 'keypoints' in ann:
            keypoints = ann['keypoints']
            # keypoints格式: [x1, y1, v1, x2, y2, v2, ...]
            num_keypoints = len(keypoints) // 3
            keypoints_count_per_ann.append(num_keypoints)

            # 统计可见性
            for i in range(2, len(keypoints), 3):
                keypoints_visibility.append(keypoints[i])

    if keypoints_count_per_ann:
        print(f"\n每个annotation的keypoints数量分布:")
        print(f"  - 最少: {min(keypoints_count_per_ann)}")
        print(f"  - 最多: {max(keypoints_count_per_ann)}")
        print(f"  - 平均: {sum(keypoints_count_per_ann)/len(keypoints_count_per_ann):.2f}")

        # 统计分布
        count_dist = Counter(keypoints_count_per_ann)
        print(f"\n  Keypoints数量分布:")
        for num, cnt in sorted(count_dist.items()):
            print(f"    {num}个角点: {cnt}个annotation ({cnt*100/len(keypoints_count_per_ann):.1f}%)")

    if keypoints_visibility:
        print(f"\n角点可见性分布 (v=0:未标记, v=1:不可见, v=2=可见):")
        visibility_dist = Counter(keypoints_visibility)
        for v, cnt in sorted(visibility_dist.items()):
            label = {0: "未标记", 1: "不可见", 2: "可见"}.get(v, f"未知({v})")
            print(f"  - {label} (v={v}): {cnt}个 ({cnt*100/len(keypoints_visibility):.1f}%)")

    # 分析合并后的结果（模拟Dataset的逻辑）
    print("\n" + "=" * 60)
    print("模拟Dataset加载逻辑（合并所有annotation）")
    print("=" * 60)

    corners_per_image = []
    for img_id, anns in img_id_to_anns.items():
        all_corners = []
        for ann in anns:
            if 'keypoints' in ann:
                keypoints = ann['keypoints']
                # v > 0 才提取（当前代码逻辑）
                for i in range(0, len(keypoints), 3):
                    x, y, v = keypoints[i:i+3]
                    if v > 0:
                        all_corners.append([x, y])

        corners_per_image.append(len(all_corners))

    if corners_per_image:
        print(f"\n每个图像合并后的角点数量:")
        print(f"  - 最少: {min(corners_per_image)}")
        print(f"  - 最多: {max(corners_per_image)}")
        print(f"  - 平均: {sum(corners_per_image)/len(corners_per_image):.2f}")

        # 分布统计
        count_dist = Counter(corners_per_image)
        print(f"\n  角点数量分布:")
        for num, cnt in sorted(count_dist.items()):
            print(f"    {num}个角点: {cnt}个图像 ({cnt*100/len(corners_per_image):.1f}%)")

        # 统计8个角点的情况
        num_with_8 = sum(1 for c in corners_per_image if c == 8)
        num_with_4 = sum(1 for c in corners_per_image if c == 4)
        num_with_less_than_4 = sum(1 for c in corners_per_image if c < 4)
        print(f"\n  关键统计:")
        print(f"    - 有8个角点: {num_with_8}个图像 ({num_with_8*100/len(corners_per_image):.1f}%)")
        print(f"    - 有4个角点: {num_with_4}个图像 ({num_with_4*100/len(corners_per_image):.1f}%)")
        print(f"    - 少于4个角点: {num_with_less_than_4}个图像 ({num_with_less_than_4*100/len(corners_per_image):.1f}%)")

    # 显示几个示例
    print("\n" + "=" * 60)
    print("示例图像分析")
    print("=" * 60)

    # 找几个典型例子
    img_id_list = list(img_id_to_anns.keys())[:5]
    for img_id in img_id_list:
        anns = img_id_to_anns[img_id]

        # 找对应的图像信息
        img_info = None
        for img in coco_data.get('images', []):
            if img['id'] == img_id:
                img_info = img
                break

        print(f"\n图像 ID {img_id} ({img_info['file_name'] if img_info else '未知'}):")
        print(f"  - 包含 {len(anns)} 个annotation")

        total_corners = 0
        for i, ann in enumerate(anns):
            if 'keypoints' in ann:
                kps = ann['keypoints']
                visible = sum(1 for j in range(2, len(kps), 3) if kps[j] > 0)
                total_corners += visible
                print(f"    Annotation {i}: {len(kps)//3} 个keypoints, {visible} 个可见")
            else:
                print(f"    Annotation {i}: 无keypoints")

        print(f"  - 合并后共 {total_corners} 个角点")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='分析COCO角点标注')
    parser.add_argument('--coco_path', type=str,
                        default='/nas2/home/qianqian/projects/corner_detection/demo-bin-picking/train_pbr/000000/scene_gt_coco.json',
                        help='COCO JSON文件路径')
    args = parser.parse_args()

    analyze_coco_annotations(args.coco_path)
