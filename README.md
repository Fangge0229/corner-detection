# Corner Detection Model

角点检测模型训练项目，支持BOP数据集格式。专为Ubuntu 18.04环境优化，支持16-bit灰度图像（PNG/JPG格式）。

## 🚀 快速开始

### 训练模型

```bash
# 1. 基本训练（从头开始）
python3 retrain_model.py

# 2. 继续训练（从检查点继续）
python3 continue_training.py

# 3. 可视化训练结果
python3 visualize_training.py
```

### ⚠️ 关键修复说明

**问题**：原始模型总是预测 0 个角点

**根本原因**：BOP数据集中的 `ignore=True` 标注（表示对象可见度低）没有被正确处理，导致模型在大量无效数据上训练。

**解决方案**：
- ✅ `train_loader_bop.py` 现已正确跳过 `ignore=True` 标注
- ✅ 只使用 641 个有效样本进行训练（共 660 个样本）
- ✅ 损失从 0.68 降至 0.004，模型正在快速学习

**所有新脚本都已使用修复后的数据加载器**

## 📁 项目结构

```
├── corner_detection.py                    # 核心模型定义
├── corner_detection_model_retrained.pth   # 训练好的模型权重（35MB）
├── train_loader_bop.py                   # BOP数据集加载器（已修复 ignore 检查）
├── retrain_model.py                      # 从头开始训练脚本
├── continue_training.py                  # 从检查点继续训练脚本
├── visualize_training.py                 # 训练数据效果可视化脚本
├── generate_8corner_coco.py              # 从BOP格式生成8角点COCO标注
├── check_coco_format.py                  # 验证COCO格式标注
├── check_loader_corners.py               # 检查数据加载器加载的角点
├── test_model_performance.py             # 模型性能评估脚本
├── test_retrained_model.py               # 重新训练模型测试脚本
├── debug_heatmap.py                      # 热图调试脚本
├── SOLUTION_SUMMARY.md                   # 详细的修复总结
└── README.md                             # 本文档
```

## 📥 模型文件

最新的重新训练模型已包含在仓库中（`corner_detection_model_retrained.pth`，35MB）。如需从头开始训练新模型：

```bash
python3 retrain_model.py  # 生成 corner_detection_model_retrained.pth
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

# 可视化依赖（可选）
pip install matplotlib opencv-python
```

### 可选依赖（GPU训练）
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 📊 数据格式

### 支持的图像格式
- **格式**: 16-bit灰度PNG/JPG / 8-bit灰度PNG/JPG / RGB PNG/JPG
- **尺寸**: 任意尺寸（自动调整为256x256）
- **示例**: `000000.png: PNG image data, 640 x 480, 16-bit grayscale`
- **示例**: `000000.jpg: JPEG image data, JFIF standard 1.01, 640 x 480`

### BOP数据集结构
```
scene_dir/                  # <-- 这个目录路径作为 --scene_dir 参数
├── rgb/                    # RGB/灰度图像目录
│   ├── 000000.png         # 16-bit灰度PNG/JPG (640x480)
│   ├── 000001.png
│   └── ...
├── scene_gt_coco.json      # COCO格式标注（必需）
├── scene_camera.json       # 相机参数（可选）
├── scene_gt.json          # ground truth poses（可选）
└── scene_gt_info.json     # pose信息（可选）
```

**重要**: `--scene_dir` 参数应该指向包含 `rgb/` 子目录的场景目录，而不是 `rgb/` 目录本身。

## 🎯 生成Ground Truth数据

### 从BOP格式生成8角点标注

如果你的数据是BOP格式（包含 `scene_gt.json` 和3D模型），可以使用以下脚本生成8角点COCO标注：

#### 1. 基本用法（处理所有物体）

```bash
python3 generate_8corner_coco.py \
    --scene_dir /path/to/train_pbr/000000 \
    --models_dir /path/to/models \
    --output /path/to/output/scene_gt_coco.json
```

#### 2. 只处理特定物体

```bash
python3 generate_8corner_coco.py \
    --scene_dir /path/to/train_pbr/000000 \
    --models_dir /path/to/models \
    --obj_id 2 \
    --output /path/to/output/scene_gt_coco.json
```

#### 3. 单位转换（如果平移向量是米）

```bash
python3 generate_8corner_coco.py \
    --scene_dir /path/to/train_pbr/000000 \
    --models_dir /path/to/models \
    --t_scale 1000 \
    --output /path/to/output/scene_gt_coco.json
```

**参数说明**:
- `--scene_dir`: BOP场景目录，包含 `rgb/`, `scene_gt.json`, `scene_camera.json`
- `--models_dir`: 3D模型目录，包含 `obj_000001.ply`, `obj_000002.ply` 等
- `--obj_id`: 可选，只处理特定obj_id的物体（默认处理所有）
- `--t_scale`: 可选，平移向量缩放因子（默认1.0，如果是米则设为1000）
- `--output`: 输出COCO JSON文件路径（默认覆盖 `scene_dir/scene_gt_coco.json`）

#### 4. 验证生成的标注

```bash
# 检查COCO格式
python3 check_coco_format.py --scene_dir /path/to/train_pbr/000000

# 检查数据加载器加载的角点
python3 check_loader_corners.py --scene_dir /path/to/train_pbr/000000

# 可视化检查
python3 visualize_training.py --scene_dir /path/to/train_pbr/000000
```

### 数据准备流程

```
1. 准备3D模型 (PLY格式)
   └── models/
       ├── obj_000001.ply
       ├── obj_000002.ply
       └── ...

2. 准备BOP场景数据
   └── train_pbr/000000/
       ├── rgb/                    # 渲染的图像
       ├── scene_gt.json          # 物体姿态
       ├── scene_camera.json      # 相机参数
       └── scene_gt_info.json     # 物体信息

3. 生成8角点标注
   └── python3 generate_8corner_coco.py

4. 验证数据
   └── python3 check_coco_format.py

5. 开始训练
   └── python3 retrain_model.py
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

### 1. 数据准备

确保数据目录结构正确：

```bash
# 检查数据目录
ls -la /path/to/scene_dir/
# 应该包含: rgb/ scene_gt_coco.json scene_camera.json

# 检查图像格式
file /path/to/scene_dir/rgb/000000.png
# 输出: 000000.png: PNG image data, 640 x 480, 16-bit grayscale, non-interlaced
```

### 2. 开始训练

#### 从头开始训练

```bash
python3 retrain_model.py
```

此脚本会：
- 从头开始训练（使用 ImageNet 预训练的 ResNet18）
- 自动保存最佳模型到 `corner_detection_model_retrained.pth`
- 在 CPU 上训练 50 个 epoch（GPU 可训练更多）

#### 继续训练

```bash
python3 continue_training.py
```

此脚本会：
- 从最新的检查点继续训练
- 使用较低的学习率（5e-5）避免发散
- 支持断点续训

### 3. 训练参数

在脚本中修改以下参数自定义训练：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_epochs` | `50` | 训练轮数 |
| `batch_size` | `8` | 批次大小 |
| `learning_rate` | `1e-4` (retrain) / `5e-5` (continue) | 学习率 |
| `num_workers` | `0` | 数据加载线程数 |

### 4. 监控训练

训练过程中会显示：
- 当前 epoch 和总 epoch
- 批次损失（每 10 个 batch）
- 平均损失（每个 epoch）
- 最佳模型保存提示

### 5. 查看训练效果

训练完成后，可以使用可视化脚本查看模型在训练数据上的表现：

```bash
# 安装可视化依赖
pip install matplotlib opencv-python

# 基本可视化（显示5个样本）
python3 visualize_training.py --scene_dir "/path/to/training/data"

# 高级可视化（指定模型和保存结果）
python3 visualize_training.py \
    --scene_dir "/path/to/training/data" \
    --model_path "./corner_detection_model.pth" \
    --num_samples 10 \
    --save_dir "./visualization_results"
```

可视化结果将显示：
- **左侧**: 原始训练图像
- **中间**: Ground Truth角点（绿色圆圈）
- **右侧**: 模型预测的角点（蓝色圆圈）

每个子图标题会显示角点数量统计。

示例输出：
```
找到 660 个训练样本 (train阶段)
开始训练，使用设备: cpu
训练样本数量: 660
批次大小: 8, 学习率: 0.0001
==================================================
Epoch 1/50, Batch 1, Loss: 0.6813
Epoch 1/50, Batch 11, Loss: 0.3828
Epoch 1/50 完成，平均损失: 0.1068
保存最佳模型，损失: 0.1068
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

### v2.0.0 (2026-02-22)
- ✅ **关键修复**: 数据加载器现正确跳过 `ignore=True` 标注
- ✅ **改进训练**: 损失从 0.68 降至 0.004，模型正快速学习
- ✅ **新脚本**: 替换 shell 脚本为 Python 脚本（`retrain_model.py`、`continue_training.py`）
- ✅ **清理项目**: 删除过时脚本，保留核心代码
- 验证在 641 个有效样本上的正确训练（共 660 个）

### v1.0.0 (2026-02-21)
- 支持16-bit灰度PNG图像
- 自动化训练脚本
- 环境验证工具
- 详细的错误处理和日志

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

本项目仅用于学术和研究目的。