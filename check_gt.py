from train_loader_bop import BOPCornerDataset
from torchvision import transforms
import numpy as np

transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
])

dataset = BOPCornerDataset('/Users/qianqian/Desktop/corner-detection/demo-bin-picking/train_pbr/000000', transform=transform)

print('dataset size', len(dataset))

found = 0
i = 0
while found < 10 and i < len(dataset):
    d = dataset[i]
    hm = d['heatmap'].squeeze(0).numpy()
    corners = d.get('corners', None)
    if corners is None:
        i += 1
        continue
    if corners.size>0 and hm.max() > 0:
        mins = corners.min(axis=0)
        maxs = corners.max(axis=0)
        print(f'i={i}, heatmap_max={hm.max():.4f}, nonzero={(hm>0.1).sum()}, corners_count={corners.shape[0]}, corners_min={mins}, corners_max={maxs}')
        found += 1
    i += 1
print(f'找到 {found} 个有效样本（前 {i} 个样本中）')
