import torch
from corner_detection import CornerDetectionModel
from train_loader_bop import BOPCornerDataset
from torchvision import transforms

# 加载模型
model = CornerDetectionModel(256, 256)
checkpoint = torch.load('corner_detection_model.pth', map_location='cpu')
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 加载数据
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = BOPCornerDataset('/nas2/home/qianqian/projects/corner_detection/demo-bin-picking/train_pbr/000000', transform=transform)

# 测试有效样本
valid_indices = [10, 11, 12, 13, 14]

print("测试当前模型在有效样本上的预测:")
print("=" * 50)

for idx in valid_indices:
    data = dataset[idx]
    image = data['image']
    corners_gt = data['corners']

    with torch.no_grad():
        image_input = image.unsqueeze(0)
        heatmap_pred = model(image_input)
        heatmap_pred = torch.sigmoid(heatmap_pred).squeeze(0).squeeze(0).cpu()

    max_val = heatmap_pred.max().item()
    print(f'样本 {idx}: GT={len(corners_gt)}角点, 最大预测值={max_val:.4f}')

print("\n结论: 如果最大预测值接近0.5，说明模型已经学习了一些特征")
print("如果仍然接近0，说明需要重新训练")