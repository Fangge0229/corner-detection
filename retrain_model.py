#!/usr/bin/env python3
"""
重新训练角点检测模型，使用修复后的数据加载器
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import time
from corner_detection import CornerDetectionModel
from train_loader_bop import BOPCornerDataset

def train_model(num_epochs=50, batch_size=8, learning_rate=1e-4):
    """训练模型"""

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载数据集
    train_dataset = BOPCornerDataset(
        '/Users/qianqian/Desktop/corner-detection/demo-bin-picking/train_pbr/000000',
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # 创建模型
    model = CornerDetectionModel(256, 256)

    # 检查是否有GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # 损失函数和优化器
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"开始训练，使用设备: {device}")
    print(f"训练样本数量: {len(train_dataset)}")
    print(f"批次大小: {batch_size}, 学习率: {learning_rate}")
    print("=" * 50)

    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        batch_count = 0

        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            targets = batch['heatmap'].to(device)

            # 前向传播
            optimizer.zero_grad()
            outputs = model(images)

            # 计算损失
            loss = criterion(outputs, targets)

            # 反向传播
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}, Loss: {loss.item():.4f}")

        avg_epoch_loss = epoch_loss / batch_count
        print(f"Epoch {epoch+1}/{num_epochs} 完成，平均损失: {avg_epoch_loss:.4f}")

        # 保存最佳模型
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, 'corner_detection_model_retrained.pth')
            print(f"保存最佳模型，损失: {best_loss:.4f}")

        # 每10个epoch保存一次检查点
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
            }, f'checkpoint_epoch_{epoch+1}.pth')

    print("训练完成！")
    return model

if __name__ == "__main__":
    # 设置随机种子保证可重复性
    torch.manual_seed(42)

    # 开始训练
    trained_model = train_model(
        num_epochs=50,
        batch_size=8,
        learning_rate=1e-4
    )

    print("模型重新训练完成！")