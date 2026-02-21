# Corner Detection Model

角点检测模型训练项目，支持BOP数据集格式。专为Ubuntu 18.04环境优化，支持16-bit灰度PNG图像。

## 🚀 快速开始

### Ubuntu环境训练（推荐）

```bash
# 1. 验证环境
python3 validate_ubuntu.py

# 2. 开始训练（使用默认参数）
./train_ubuntu.sh

# 3. 或自定义参数训练
./train_ubuntu.sh --epochs 200 --batch_size 16 --lr 0.0001
```

### Python脚本训练

```python
from train_bop_ubuntu import train_bop_model

# 训练模型
train_bop_model(
    scene_dir="/path/to/bop/scene",
    num_epochs=100,
    batch_size=8,
    learning_rate=1e-4
)
```

## 📁 项目结构

```
├── corner_detection.py           # 核心模型定义
├── corner_detection_model.pth    # 训练好的模型权重
├── train_bop_ubuntu.py          # Ubuntu专用训练脚本
├── train_ubuntu.sh              # 自动化训练脚本
├── validate_ubuntu.py           # 环境验证工具
├── train_loader_bop.py          # BOP数据集加载器
├── train_loader_bop_usage.py    # BOP loader使用指南
└── README.md                    # 本文档
```

## 🔧 环境要求

### 系统要求
- **操作系统**: Ubuntu 18.04+ / macOS / Windows
- **Python**: 3.6+
- **内存**: 8GB+ (推荐16GB+)
- **存储**: 10GB+ 可用空间

### 依赖包
```bash
pip install torch torchvision pillow numpy
```

### 可选依赖（GPU训练）
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 📊 数据格式

### 支持的图像格式
- **格式**: 16-bit灰度PNG / 8-bit灰度PNG / RGB PNG
- **尺寸**: 任意尺寸（自动调整为256x256）
- **示例**: `000000.png: PNG image data, 640 x 480, 16-bit grayscale`

### BOP数据集结构
```
scene_dir/
├── rgb/                    # RGB/灰度图像目录
│   ├── 000000.png         # 16-bit灰度PNG (640x480)
│   ├── 000001.png
│   └── ...
├── scene_gt_coco.json      # COCO格式标注（必需）
├── scene_camera.json       # 相机参数
├── scene_gt.json          # ground truth poses
└── scene_gt_info.json     # pose信息
```

### COCO标注格式
```json
{
  "images": [{"id": 0, "file_name": "000000.png", "width": 640, "height": 480}],
  "annotations": [
    {
      "image_id": 0,
      "keypoints": [x1, y1, v1, x2, y2, v2, ...],  // 角点坐标
      "bbox": [x, y, w, h]  // 边界框（备选）
    }
  ]
}
```

## 🏗️ 模型架构

- **Backbone**: ResNet18 (ImageNet预训练)
- **输入尺寸**: 256×256×3
- **输出**: 256×256角点heatmap
- **损失函数**: BCEWithLogitsLoss
- **优化器**: Adam + 学习率调度

## 🎯 训练指南

### 1. 环境验证

首先验证您的环境是否准备就绪：

```bash
python3 validate_ubuntu.py
```

成功输出示例：
```
✅ 所有测试通过！可以开始训练。
运行训练命令: ./train_ubuntu.sh
```

### 2. 数据准备

确保数据目录结构正确：

```bash
# 检查数据目录
ls -la /path/to/scene_dir/
# 应该包含: rgb/ scene_gt_coco.json scene_camera.json

# 检查图像格式
file /path/to/scene_dir/rgb/000000.png
# 输出: 000000.png: PNG image data, 640 x 480, 16-bit grayscale, non-interlaced
```

### 3. 开始训练

#### 方法一：使用自动化脚本（推荐）

```bash
# 基本训练
./train_ubuntu.sh

# 高级训练（自定义参数）
./train_ubuntu.sh \
    --scene_dir "/data/bop_scene" \
    --epochs 200 \
    --batch_size 16 \
    --lr 0.00005 \
    --save_path "./my_model.pth"
```

#### 方法二：使用Python脚本

```python
from train_bop_ubuntu import train_bop_model

train_bop_model(
    scene_dir="/data/bop_scene",
    num_epochs=100,
    batch_size=8,
    learning_rate=1e-4,
    save_path="./model.pth",
    log_interval=10
)
```

### 4. 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--scene_dir` | `/nas2/home/qianqian/...` | 数据集路径 |
| `--epochs` | `100` | 训练轮数 |
| `--batch_size` | `8` | 批次大小 |
| `--lr` | `0.0001` | 初始学习率 |
| `--save_path` | `./corner_detection_model.pth` | 模型保存路径 |
| `--log_interval` | `10` | 日志打印间隔 |

### 5. 监控训练

训练过程中会显示：
- 当前epoch和总epoch
- 批次损失
- 学习率
- 训练时间
- 最佳模型保存提示

示例输出：
```
使用设备: cuda
开始训练 100 个epoch...
Epoch 1/100, Batch 10/150, Loss: 0.5432, Time: 15.2s
Epoch 1/100 完成，平均损失: 0.6789，用时: 45.6s，学习率: 0.000100
保存最佳模型到: ./corner_detection_model.pth
```

## 🔍 推理使用

### 加载训练好的模型

```python
import torch
from corner_detection import CornerDetectionModel
from torchvision import transforms

# 加载模型
model = CornerDetectionModel(256, 256)
model.load_state_dict(torch.load('corner_detection_model.pth'))
model.eval()

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 处理单张图像
def detect_corners(image_path):
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        heatmap = model(input_tensor).squeeze()

    # 应用sigmoid获取概率
    prob_map = torch.sigmoid(heatmap)

    # 阈值处理获取角点
    corners = (prob_map > 0.5).nonzero(as_tuple=True)

    return corners
```

## 🛠️ 故障排除

### 常见问题

#### 1. CUDA不可用
```
警告: 未检测到CUDA，将使用CPU训练
```
**解决**: 安装CUDA版本的PyTorch
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 2. 内存不足
```
CUDA out of memory
```
**解决**: 减小批次大小
```bash
./train_ubuntu.sh --batch_size 4
```

#### 3. 数据路径不存在
```
错误: 场景目录不存在
```
**解决**: 检查并修正数据路径
```bash
ls -la /path/to/your/scene_dir/
```

#### 4. COCO文件格式错误
```
JSON解析错误
```
**解决**: 验证JSON格式
```bash
python3 -c "import json; json.load(open('scene_gt_coco.json'))"
```

### 性能优化

#### GPU优化
- 使用 `--batch_size 16-32` 充分利用GPU
- 确保CUDA版本与PyTorch兼容

#### CPU优化
- 设置 `--num_workers 4` 使用多线程
- 使用 `--batch_size 8` 避免内存溢出

## 📈 训练技巧

### 学习率调整
- 初始学习率: 1e-4
- 每30个epoch衰减为原来的0.1倍
- 监控损失曲线，避免过拟合

### 数据增强
- 自动尺寸调整: 任意尺寸 → 256×256
- 标准化: ImageNet均值和方差

### 模型保存
- 自动保存最佳模型（最低验证损失）
- 支持断点续训（加载checkpoint）

## 📝 更新日志

### v1.0.0 (2026-02-21)
- 支持16-bit灰度PNG图像
- Ubuntu 18.04环境优化
- 自动化训练脚本
- 环境验证工具
- 详细的错误处理和日志

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

本项目仅用于学术和研究目的。