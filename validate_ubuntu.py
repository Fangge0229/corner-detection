#!/usr/bin/env python3
"""
Ubuntu环境验证脚本
验证训练环境和数据处理是否正常
"""

import torch
import numpy as np
from torchvision import transforms
import sys

def test_environment():
    """测试训练环境"""

    print("=== Ubuntu训练环境检查 ===")

    # 检查PyTorch
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("警告: 未检测到CUDA，将使用CPU训练")

    # 检查numpy
    print(f"NumPy版本: {np.__version__}")

    print("环境检查完成！\n")

def test_data_processing():
    """测试数据处理流程"""

    print("=== 数据处理测试 ===")

    # 创建测试图像 (模拟640x480的处理后图像)
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # 转换为PIL图像
    pil_image = transforms.ToPILImage()(test_image)

    # 处理图像
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    processed_tensor = transform(pil_image)

    print(f"原始图像形状: {test_image.shape}")
    print(f"处理后张量形状: {processed_tensor.shape}")
    print(f"值范围: {processed_tensor.min():.3f} - {processed_tensor.max():.3f}")

    print("数据处理测试完成！\n")

def test_model():
    """测试模型加载"""

    print("=== 模型测试 ===")

    try:
        from corner_detection import CornerDetectionModel

        model = CornerDetectionModel(256, 256)
        print("模型创建成功！")

        # 测试前向传播
        test_input = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            output = model(test_input)

        print(f"输入形状: {test_input.shape}")
        print(f"输出形状: {output.shape}")

        print("模型测试完成！\n")

    except Exception as e:
        print(f"模型测试失败: {e}")
        return False

    return True

def main():
    """主函数"""

    print("Ubuntu角点检测训练环境验证")
    print("=" * 40)

    try:
        test_environment()
        test_data_processing()

        if test_model():
            print("✅ 所有测试通过！可以开始训练。")
            print("\n运行训练命令:")
            print("./train_ubuntu.sh")
        else:
            print("❌ 模型测试失败，请检查依赖项。")
            sys.exit(1)

    except Exception as e:
        print(f"❌ 测试过程中出错: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()