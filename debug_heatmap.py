import torch
from train_loader_bop import BOPCornerDataset
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

# 加载数据
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = BOPCornerDataset('/Users/qianqian/Desktop/corner-detection/demo-bin-picking/train_pbr/000000', transform=transform)

# 检查几个有效样本的热图
valid_indices = [10, 11, 12]

for idx in valid_indices:
    data = dataset[idx]
    heatmap = data['heatmap'].squeeze(0).numpy()  # 移除通道维度

    print(f"样本 {idx}:")
    print(f"  热图形状: {heatmap.shape}")
    print(f"  最大值: {heatmap.max():.4f}")
    print(f"  最小值: {heatmap.min():.4f}")
    print(f"  非零像素数量: {(heatmap > 0.1).sum()}")
    print(f"  高于0.5的像素数量: {(heatmap > 0.5).sum()}")
    print()

# 可视化第一个样本的热图
plt.figure(figsize=(8, 6))
plt.imshow(heatmap, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title(f'样本 {valid_indices[0]} 的角点热图')
plt.savefig('heatmap_visualization.png')
plt.show()
print("热图已保存为 heatmap_visualization.png")