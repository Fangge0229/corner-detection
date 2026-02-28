#!/usr/bin/env python3
"""
继续训练角点检测模型，使用之前保存的最佳检查点
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from corner_detection import CornerDetectionModel
from train_loader_bop import BOPCornerDataset, collate_fn

def continue_training(checkpoint_path, num_epochs=100, batch_size=8, learning_rate=1e-4):
    """继续训练模型"""

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载数据集 - 8角点数据
    train_dataset = BOPCornerDataset(
        '/nas2/home/qianqian/projects/HCCEPose/demo-bin-picking/train_pbr/000000',
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)

    # 创建模型
    model = CornerDetectionModel(256, 256)

    # 检查是否有GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint.get('epoch', 0)

    # 损失函数和优化器
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 加载优化器状态
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f"从第 {start_epoch} 个epoch继续训练")
    print(f"使用设备: {device}")
    print(f"训练样本数量: {len(train_dataset)}")
    print(f"批次大小: {batch_size}, 学习率: {learning_rate}")
    print("=" * 50)

    best_loss = checkpoint.get('loss', float('inf'))

    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()
        epoch_loss = 0.0
        batch_count = 0

        for batch_idx, batch in enumerate(train_loader):
            images = batch['images'].to(device)
            targets = batch['heatmaps'].to(device)

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
                print(f"Epoch {epoch+1}/{start_epoch + num_epochs}, Batch {batch_idx+1}, Loss: {loss.item():.4f}")

        avg_epoch_loss = epoch_loss / batch_count
        print(f"Epoch {epoch+1}/{start_epoch + num_epochs} 完成，平均损失: {avg_epoch_loss:.4f}")

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

        # 每5个epoch保存一次检查点
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
            }, f'checkpoint_epoch_{epoch+1}.pth')
            print(f"保存检查点: checkpoint_epoch_{epoch+1}.pth")

    print("训练完成！")
    return model

if __name__ == "__main__":
    # 设置随机种子保证可重复性
    torch.manual_seed(42)

    # 从保存的检查点继续训练
    # 如果检查点不存在，则从头开始训练
    checkpoint_path = 'corner_detection_model_retrained.pth'

    if torch.cuda.is_available():
        num_epochs = 100  # GPU上训练100个epoch
    else:
        num_epochs = 50   # CPU上训练50个epoch

    trained_model = continue_training(
        checkpoint_path,
        num_epochs=num_epochs,
        batch_size=8,
        learning_rate=5e-5  # 降低学习率，因为模型已经有一定的训练基础
    )

    print("模型训练完成！")