"""
BOP数据集角点检测训练指南

本文件展示了如何使用为BOP数据集设计的角点检测训练loader。

BOP数据集结构：
scene_dir/
├── rgb/                    # RGB图像
├── depth/                  # 深度图
├── mask/                   # 对象mask
├── mask_visib/             # 可见mask
├── scene_camera.json       # 相机参数
├── scene_gt.json           # ground truth poses
├── scene_gt_coco.json      # COCO格式标注 (bboxes, keypoints等)
├── scene_gt_info.json      # pose信息
└── scene_gt.json

基本使用模式：
1. 准备BOP数据集
2. 创建数据加载器
3. 训练模型

注意事项：
- scene_gt_coco.json中的keypoints字段用于角点标注
- 如果没有keypoints，将使用bbox的四个角点作为角点
- 角点坐标会转换为heatmap用于训练
"""

import torch
import os
from train_loader_bop import (
    create_bop_data_loader,
    train_bop_model,
    BOPCornerDataset
)
from corner_detection import criterion

def example_basic_usage():
    """基本使用示例"""

    # 1. 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 2. 数据集路径
    scene_dir = "/nas2/home/qianqian/projects/corner_detection/demo-bin-picking/train_pbr/000000/rgb"

    # 检查数据集是否存在
    if not os.path.exists(scene_dir):
        print(f"数据集路径不存在: {scene_dir}")
        print("请确保数据路径正确，或将数据复制到本地")
        return

    # 3. 创建数据加载器
    try:
        train_loader = create_bop_data_loader(
            scene_dir=scene_dir,
            batch_size=4,
            num_workers=0,  # 在macOS上设为0避免问题
            phase='train'
        )

        if train_loader is None:
            print(f"无法创建数据加载器，请检查路径: {scene_dir}")
            print("确保scene_gt_coco.json文件存在")
            return

        # 4. 测试数据加载
        print("测试数据加载...")
        for batch in train_loader:
            print(f"批次图像形状: {batch['images'].shape}")
            print(f"Heatmap形状: {batch['heatmaps'].shape}")
            print(f"角点数量: {[len(c) for c in batch['corners']]}")
            break

        print("数据加载测试完成！")

    except Exception as e:
        print(f"创建数据加载器时出错: {e}")
        print("请检查数据集结构和文件是否存在")

def example_training():
    """训练示例"""

    # 数据集路径
    scene_dir = "/nas2/home/qianqian/projects/corner_detection/demo-bin-picking/train_pbr/000000/rgb"

    # 检查数据集是否存在
    if not os.path.exists(scene_dir):
        print(f"数据集路径不存在: {scene_dir}")
        print("请先准备数据集")
        return

    # 训练参数
    num_epochs = 10
    batch_size = 4
    learning_rate = 1e-4

    # 开始训练
    train_bop_model(
        scene_dir=scene_dir,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )

def check_dataset():
    """检查数据集结构"""

    import json

    scene_dir = "/nas2/home/qianqian/projects/corner_detection/demo-bin-picking/train_pbr/000000/rgb"

    # 检查必要文件
    required_files = [
        'scene_gt_coco.json',
        'rgb',
        'scene_camera.json'
    ]

    print(f"检查数据集: {scene_dir}")
    for file in required_files:
        path = os.path.join(scene_dir, file)
        if os.path.exists(path):
            print(f"✓ {file} 存在")
        else:
            print(f"✗ {file} 不存在")

    # 检查COCO标注
    coco_path = os.path.join(scene_dir, 'scene_gt_coco.json')
    if os.path.exists(coco_path):
        try:
            with open(coco_path, 'r') as f:
                coco_data = json.load(f)

            print(f"图像数量: {len(coco_data['images'])}")
            print(f"标注数量: {len(coco_data['annotations'])}")

            # 检查是否有keypoints
            has_keypoints = any('keypoints' in ann for ann in coco_data['annotations'])
            print(f"是否有keypoints: {has_keypoints}")

            if has_keypoints:
                keypoints_count = sum(len(ann.get('keypoints', [])) // 3 for ann in coco_data['annotations'])
                print(f"总keypoints数量: {keypoints_count}")
        except Exception as e:
            print(f"读取COCO文件出错: {e}")

if __name__ == "__main__":
    print("BOP数据集角点检测训练指南")
    print("=" * 40)

    # 检查数据集
    check_dataset()
    print()

    # 基本使用示例
    example_basic_usage()
    print()

    # 可选：开始训练
    # print("开始训练...")
    # example_training()