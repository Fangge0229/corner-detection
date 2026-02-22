#!/bin/bash
# BOP数据集角点检测训练脚本 - Ubuntu运行脚本
# 用于在Ubuntu 18.04系统上运行训练

# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 检查CUDA是否可用
if command -v nvidia-smi &> /dev/null; then
    echo "检测到NVIDIA GPU:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
    echo ""
else
    echo "未检测到NVIDIA GPU，将使用CPU训练"
fi

# 默认参数
SCENE_DIR="/nas2/home/qianqian/projects/corner_detection/demo-bin-picking/train_pbr/000000"
EPOCHS=100
BATCH_SIZE=8
LEARNING_RATE=0.0001
SAVE_PATH="./corner_detection_model.pth"
LOG_INTERVAL=10

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --scene_dir)
      SCENE_DIR="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --lr)
      LEARNING_RATE="$2"
      shift 2
      ;;
    --save_path)
      SAVE_PATH="$2"
      shift 2
      ;;
    --log_interval)
      LOG_INTERVAL="$2"
      shift 2
      ;;
    *)
      echo "未知参数: $1"
      echo "使用方法: $0 [--scene_dir PATH] [--epochs N] [--batch_size N] [--lr FLOAT] [--save_path PATH] [--log_interval N]"
      exit 1
      ;;
  esac
done

# 检查必要文件
echo "检查训练环境..."
echo "场景目录: $SCENE_DIR"

if [ ! -d "$SCENE_DIR" ]; then
    echo "错误: 场景目录不存在: $SCENE_DIR"
    exit 1
fi

if [ ! -f "$SCENE_DIR/scene_gt_coco.json" ]; then
    echo "错误: COCO标注文件不存在: $SCENE_DIR/scene_gt_coco.json"
    exit 1
fi

# SCENE_DIR 指向 rgb 目录，检查图像文件是否存在
IMAGE_COUNT=$(ls "$SCENE_DIR"/*.png 2>/dev/null | wc -l)
if [ "$IMAGE_COUNT" -eq 0 ]; then
    echo "错误: RGB图像目录中没有找到PNG文件: $SCENE_DIR"
    exit 1
fi
echo "数据集信息:"
echo "  图像数量: $IMAGE_COUNT"

# 检查Python环境
echo "检查Python环境..."
python3 --version
pip3 list | grep -E "(torch|torchvision|numpy|PIL)"

# 开始训练
echo ""
echo "开始训练..."
echo "参数:"
echo "  场景目录: $SCENE_DIR"
echo "  训练轮数: $EPOCHS"
echo "  批次大小: $BATCH_SIZE"
echo "  学习率: $LEARNING_RATE"
echo "  保存路径: $SAVE_PATH"
echo "  日志间隔: $LOG_INTERVAL"
echo ""

python3 train_bop_ubuntu.py \
    --scene_dir "$SCENE_DIR" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --save_path "$SAVE_PATH" \
    --log_interval $LOG_INTERVAL

# 检查训练结果
if [ $? -eq 0 ]; then
    echo ""
    echo "训练完成！"
    if [ -f "$SAVE_PATH" ]; then
        echo "模型已保存到: $SAVE_PATH"
        ls -lh "$SAVE_PATH"
    fi
else
    echo ""
    echo "训练失败！"
    exit 1
fi