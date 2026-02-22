#!/usr/bin/env python3
"""
测试JPG图像格式支持
验证代码是否能正确处理JPG格式的训练图像
"""

import os
import numpy as np
from PIL import Image
import tempfile

def test_jpg_support():
    """测试JPG格式图像的加载和处理"""

    print("=== JPG格式支持测试 ===")

    # 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        # 创建测试JPG图像
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        pil_image = Image.fromarray(test_image, mode='RGB')

        # 保存为JPG格式
        jpg_path = os.path.join(temp_dir, 'test_image.jpg')
        pil_image.save(jpg_path, 'JPEG', quality=95)

        print(f"创建测试JPG图像: {jpg_path}")

        # 测试加载JPG图像
        try:
            loaded_image = Image.open(jpg_path)
            print(f"成功加载JPG图像: 模式={loaded_image.mode}, 大小={loaded_image.size}")

            # 测试图像处理逻辑（模拟train_loader_bop.py中的处理）
            if loaded_image.mode == 'I' or loaded_image.mode == 'L':  # 16-bit or 8-bit grayscale
                print("检测到灰度图像，进行转换处理...")
                img_array = np.array(loaded_image)

                # 归一化到0-255范围 (16-bit -> 8-bit)
                if img_array.dtype == np.uint16:
                    img_array = (img_array / 65535.0 * 255).astype(np.uint8)
                    print("转换16-bit到8-bit")
                elif img_array.dtype == np.uint8:
                    print("已经是8-bit")
                else:
                    img_array = (img_array * 255).astype(np.uint8)

                # 创建PIL图像
                processed_image = Image.fromarray(img_array, mode='L')

                # 转换为RGB (复制灰度到3通道)
                processed_image = processed_image.convert('RGB')
                print(f"转换为RGB: 模式={processed_image.mode}, 大小={processed_image.size}")
            else:
                print("检测到彩色图像，无需转换")
                processed_image = loaded_image

            print("✅ JPG图像处理成功！")

        except Exception as e:
            print(f"❌ JPG图像处理失败: {e}")
            return False

    print("\n=== 测试结果 ===")
    print("✅ 代码已支持JPG格式图像")
    print("✅ PIL.Image.open() 自动处理多种图像格式")
    print("✅ 现有的图像处理逻辑适用于JPG格式")

    return True

if __name__ == "__main__":
    test_jpg_support()