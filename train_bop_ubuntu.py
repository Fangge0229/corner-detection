#!/usr/bin/env python3
"""
BOP数据集角点检测训练脚本 - Linux Ubuntu版本
在Ubuntu 18.04上运行的专用训练脚本
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import sys
import argparse
import json
from PIL import Image
import numpy as np
from datetime import datetime

# 导入本地模块
from corner_detection import CornerDetectionModel, criterion

def corners_to_heatmap(corners, height, width, sigma=2.0):
    """
    将角点坐标转换为heatmap
    """
    heatmap = np.zeros((height, width), dtype=np.float32)

    if len(corners) == 0:
        return heatmap

    for corner in corners:
        x, y = corner
        # 确保坐标在图像范围内
        x = np.clip(x, 0, width - 1)
        y = np.clip(y, 0, height - 1)

        # 创建高斯核
        xx, yy = np.meshgrid(np.arange(width), np.arange(height))
        gaussian = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))

        # 累加到heatmap
        heatmap += gaussian

    # 归一化到[0, 1]
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    return heatmap

class BOPCornerDataset(torch.utils.data.Dataset):
    """
    BOP数据集的角点检测Dataset类 - Ubuntu优化版本
    """
    def __init__(self, scene_dir, transform=None, phase='train'):
        self.scene_dir = scene_dir
        self.transform = transform
        self.phase = phase

        # 加载COCO格式的标注
        coco_path = os.path.join(scene_dir, 'scene_gt_coco.json')
        if not os.path.exists(coco_path):
            raise FileNotFoundError(f"COCO标注文件不存在: {coco_path}")

        with open(coco_path, 'r') as f:
            self.coco_data = json.load(f)

        # 获取图像列表
        self.images = self.coco_data['images']
        self.annotations = self.coco_data['annotations']

        # 创建图像ID到标注的映射
        self.img_id_to_anns = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.img_id_to_anns:
                self.img_id_to_anns[img_id] = []
            self.img_id_to_anns[img_id].append(ann)

        print(f"找到 {len(self.images)} 个训练样本 ({phase}阶段)")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 获取图像信息
        img_info = self.images[idx]
        img_id = img_info['id']
        img_filename = img_info['file_name']

        # 处理file_name中的rgb/前缀
        if img_filename.startswith('rgb/'):
            img_filename = img_filename[4:]  # 移除 'rgb/' 前缀

        # 加载图像 (16-bit灰度PNG)
        rgb_dir = os.path.join(self.scene_dir, 'rgb')
        img_path = os.path.join(rgb_dir, img_filename)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"图像文件不存在: {img_path}")

        # 处理16-bit PNG图像
        image = Image.open(img_path)

        # 如果是16-bit灰度图，转换为8-bit并复制到3通道
        if image.mode == 'I' or image.mode == 'L':  # 16-bit or 8-bit grayscale
            # 转换为numpy数组进行处理
            img_array = np.array(image)

            # 归一化到0-255范围 (16-bit -> 8-bit)
            if img_array.dtype == np.uint16:
                img_array = (img_array / 65535.0 * 255).astype(np.uint8)
            elif img_array.dtype == np.uint8:
                pass  # 已经是8-bit
            else:
                img_array = (img_array * 255).astype(np.uint8)

            # 创建PIL图像
            image = Image.fromarray(img_array, mode='L')

            # 转换为RGB (复制灰度到3通道)
            image = image.convert('RGB')

        # 获取该图像的标注
        anns = self.img_id_to_anns.get(img_id, [])

        # 提取角点坐标
        corners = []
        for ann in anns:
            # 跳过被忽略的标注
            if ann.get('ignore', False):
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

        # 如果没有keypoints，尝试从bbox计算角点
        if not corners and anns:
            for ann in anns:
                # 跳过被忽略的标注
                if ann.get('ignore', False):
                    continue

                if 'bbox' in ann:
                    x, y, w, h = ann['bbox']
                    # 计算bbox的四个角点
                    corners.extend([
                        [x, y],          # 左上
                        [x + w, y],      # 右上
                        [x + w, y + h],  # 右下
                        [x, y + h]       # 左下
                    ])

        # 转换为numpy数组
        if corners:
            corners = np.array(corners, dtype=np.float32)
        else:
            corners = np.array([], dtype=np.float32).reshape(0, 2)

        # 数据预处理
        if self.transform:
            try:
                image = self.transform(image)
            except Exception as e:
                print(f"Transform error for image {img_path}: {e}")
                print(f"Image mode: {Image.open(img_path).mode}")
                print(f"Image size: {Image.open(img_path).size}")
                raise e

        # 创建角点heatmap (使用模型输入尺寸256x256)
        try:
            heatmap = corners_to_heatmap(corners, 256, 256)
        except Exception as e:
            print(f"Heatmap creation error for image {img_path}: {e}")
            print(f"Corners shape: {corners.shape if hasattr(corners, 'shape') else 'No shape'}")
            raise e

        # 验证输出格式
        if not isinstance(image, torch.Tensor):
            raise ValueError(f"Image is not a tensor: {type(image)}")
        if image.shape[0] != 3:
            raise ValueError(f"Image has wrong number of channels: {image.shape}")
        if heatmap.shape != (256, 256):
            raise ValueError(f"Heatmap has wrong shape: {heatmap.shape}")

        return {
            'image': image,
            'heatmap': torch.from_numpy(heatmap).unsqueeze(0),  # 添加通道维度
            'corners': corners,  # 保留原始角点坐标用于调试
            'image_id': img_id,
            'image_path': img_path
        }

def create_bop_data_loader(scene_dir, batch_size=4, num_workers=4, phase='train'):
    """
    创建BOP数据集的DataLoader - Ubuntu优化版本

    Args:
        scene_dir: 场景目录路径
        batch_size: 批次大小
        num_workers: 数据加载线程数 (Ubuntu上可以使用多线程)
        phase: 阶段 ('train', 'val', 'test')
    """

    # 数据预处理 (针对640x480灰度图像，resize到256x256)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 从640x480 resize到256x256
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # 创建数据集
    dataset = BOPCornerDataset(scene_dir, transform=transform, phase=phase)

    # 创建DataLoader
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(phase == 'train'),
        num_workers=0,  # 暂时禁用多线程以避免存储重置错误
        pin_memory=False,  # 禁用pin_memory
        persistent_workers=False,
        collate_fn=collate_fn  # 使用自定义collate函数
    )

    return data_loader

def collate_fn(batch):
    """
    自定义collate函数处理变长角点列表
    """
    images = []
    heatmaps = []
    corners_list = []
    image_ids = []
    image_paths = []

    for item in batch:
        try:
            images.append(item['image'])
            heatmaps.append(item['heatmap'])
            corners_list.append(item['corners'])
            image_ids.append(item['image_id'])
            image_paths.append(item['image_path'])
        except Exception as e:
            print(f"Error processing batch item: {e}")
            print(f"Item keys: {item.keys() if isinstance(item, dict) else 'Not a dict'}")
            raise e

    try:
        return {
            'images': torch.stack(images),
            'heatmaps': torch.stack(heatmaps),
            'corners': corners_list,
            'image_ids': image_ids,
            'image_paths': image_paths
        }
    except Exception as e:
        print(f"Error stacking batch: {e}")
        print(f"Images shapes: {[img.shape for img in images]}")
        print(f"Heatmaps shapes: {[hmap.shape for hmap in heatmaps]}")
        raise e

def train_bop_model(scene_dir, num_epochs=100, batch_size=8, learning_rate=1e-4,
                   save_path='./corner_detection_model.pth', log_interval=10):
    """
    使用BOP数据集训练角点检测模型 - Ubuntu优化版本

    Args:
        scene_dir: 场景目录路径
        num_epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
        save_path: 模型保存路径
        log_interval: 日志打印间隔
    """

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    if torch.cuda.is_available():
        print(f"GPU数量: {torch.cuda.device_count()}")
        print(f"当前GPU: {torch.cuda.get_device_name()}")

    # 创建模型
    model = CornerDetectionModel(256, 256).to(device)

    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 创建学习率调度器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # 创建数据加载器
    train_loader = create_bop_data_loader(scene_dir, batch_size=batch_size, phase='train')

    if not train_loader:
        print(f"无法创建数据加载器，请检查路径: {scene_dir}")
        return

    # 创建损失函数
    loss_fn = criterion()

    # 训练循环
    model.train()
    best_loss = float('inf')

    print(f"开始训练 {num_epochs} 个epoch...")
    print(f"数据集大小: {len(train_loader.dataset)}")
    print(f"批次大小: {batch_size}")
    print(f"总批次数: {len(train_loader)}")

    for epoch in range(num_epochs):
        total_loss = 0.0
        epoch_start_time = datetime.now()

        for batch_idx, batch in enumerate(train_loader):
            images = batch['images'].to(device)
            heatmaps = batch['heatmaps'].to(device)

            # 前向传播
            outputs = model(images)

            # 计算损失
            loss = loss_fn(outputs, heatmaps)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # 打印训练进度
            if batch_idx % log_interval == 0:
                current_time = datetime.now()
                elapsed = (current_time - epoch_start_time).total_seconds()
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}, Time: {elapsed:.1f}s")

        # 每个epoch结束
        avg_loss = total_loss / len(train_loader)
        scheduler.step()

        epoch_time = (datetime.now() - epoch_start_time).total_seconds()
        print(f"Epoch {epoch+1}/{num_epochs} 完成，平均损失: {avg_loss:.4f}, "
              f"用时: {epoch_time:.1f}s, 学习率: {scheduler.get_last_lr()[0]:.6f}")

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': best_loss,
            }, save_path)
            print(f"保存最佳模型到: {save_path}")

    print("训练完成！")
    print(f"最佳损失: {best_loss:.4f}")

def main():
    parser = argparse.ArgumentParser(description='BOP数据集角点检测训练')
    parser.add_argument('--scene_dir', type=str, required=True,
                       help='BOP场景目录路径')
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='学习率')
    parser.add_argument('--save_path', type=str, default='./corner_detection_model.pth',
                       help='模型保存路径')
    parser.add_argument('--log_interval', type=int, default=10,
                       help='日志打印间隔')

    args = parser.parse_args()

    # 检查目录是否存在
    if not os.path.exists(args.scene_dir):
        print(f"错误: 场景目录不存在: {args.scene_dir}")
        sys.exit(1)

    # 开始训练
    train_bop_model(
        scene_dir=args.scene_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_path=args.save_path,
        log_interval=args.log_interval
    )

if __name__ == "__main__":
    main()