import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
import numpy as np
import json
from corner_detection import CornerDetectionModel, criterion

def corners_to_heatmap(corners, height, width, sigma=2.0):
    """
    将角点坐标转换为heatmap

    Args:
        corners: 角点坐标列表, shape (N, 2)
        height: heatmap高度
        width: heatmap宽度
        sigma: 高斯核标准差

    Returns:
        heatmap: shape (height, width)
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

class BOPCornerDataset(Dataset):
    """
    BOP数据集的角点检测Dataset类
    加载RGB图像和从scene_gt_coco.json中提取的角点标注
    """
    def __init__(self, scene_dir, transform=None, phase='train'):
        self.scene_dir = scene_dir
        self.transform = transform
        self.phase = phase

        # 加载COCO格式的标注
        coco_path = os.path.join(scene_dir, 'scene_gt_coco.json')
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

        # 加载图像 (16-bit灰度图)
        rgb_dir = os.path.join(self.scene_dir, 'rgb')
        img_path = os.path.join(rgb_dir, img_filename)
        
        # 处理16-bit图像（PNG/JPG等格式）
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
        
        # 获取图像尺寸
        width, height = image.size

        # 获取该图像的标注
        anns = self.img_id_to_anns.get(img_id, [])

        # 提取角点坐标 (假设keypoints字段存在)
        corners = []
        for ann in anns:
            # 注意: 忽略ignore字段的检查，假设所有标注都是有效的
            # if ann.get('ignore', False):
            #     continue

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
                # 注意: 忽略ignore字段的检查，假设所有标注都是有效的
                # if ann.get('ignore', False):
                #     continue

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
            # 如果没有角点，创建一个空的数组
            corners = np.array([], dtype=np.float32).reshape(0, 2)

        # 数据增强和预处理
        if self.transform:
            image = self.transform(image)

        # 创建角点heatmap (使用模型输入尺寸256x256)
        heatmap = corners_to_heatmap(corners, 256, 256)

        return {
            'image': image,
            'heatmap': torch.from_numpy(heatmap).unsqueeze(0),  # 添加通道维度
            'corners': corners,  # 保留原始角点坐标用于调试
            'image_id': img_id,
            'image_path': img_path
        }

def create_bop_data_loader(scene_dir, batch_size=4, num_workers=0, phase='train'):
    """
    创建BOP数据集的DataLoader

    Args:
        scene_dir: 场景目录路径 (包含rgb/, scene_gt_coco.json等)
        batch_size: 批次大小
        num_workers: 数据加载线程数
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
        num_workers=num_workers,
        collate_fn=collate_fn
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
        images.append(item['image'])
        heatmaps.append(item['heatmap'])
        corners_list.append(item['corners'])
        image_ids.append(item['image_id'])
        image_paths.append(item['image_path'])

    return {
        'images': torch.stack(images),
        'heatmaps': torch.stack(heatmaps),
        'corners': corners_list,  # 保持为列表，因为长度不同
        'image_ids': image_ids,
        'image_paths': image_paths
    }

def train_bop_model(scene_dir, num_epochs=10, batch_size=4, learning_rate=1e-4):
    """
    使用BOP数据集训练角点检测模型

    Args:
        scene_dir: 场景目录路径
        num_epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建模型
    model = CornerDetectionModel(256, 256).to(device)

    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 创建数据加载器
    train_loader = create_bop_data_loader(scene_dir, batch_size=batch_size, phase='train')

    if not train_loader:
        print(f"无法创建数据加载器，请检查路径: {scene_dir}")
        return

    # 训练循环
    model.train()
    loss_fn = criterion()

    for epoch in range(num_epochs):
        total_loss = 0.0

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

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs} 完成，平均损失: {avg_loss:.4f}")

    print("训练完成！")

if __name__ == "__main__":
    # 示例用法
    scene_dir = "/nas2/home/qianqian/projects/corner_detection/demo-bin-picking/train_pbr/000000"

    # 创建数据加载器
    train_loader = create_bop_data_loader(scene_dir, batch_size=2)

    # 测试数据加载
    if train_loader:
        for batch in train_loader:
            print(f"批次图像形状: {batch['images'].shape}")
            print(f"Heatmap形状: {batch['heatmaps'].shape}")
            print(f"角点数量: {[len(c) for c in batch['corners']]}")
            break

    # 训练模型 (可选)
    # train_bop_model(scene_dir)