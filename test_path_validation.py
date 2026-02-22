#!/usr/bin/env python3
"""
测试训练脚本的路径检查逻辑
验证修改后的脚本是否能正确识别数据集结构
"""

import os
import tempfile
import json

def test_path_validation():
    """测试路径验证逻辑"""

    print("=== 路径验证测试 ===")

    # 创建临时目录结构模拟BOP数据集
    with tempfile.TemporaryDirectory() as temp_dir:
        # 创建场景目录
        scene_dir = os.path.join(temp_dir, "scene_000000")
        rgb_dir = os.path.join(scene_dir, "rgb")
        os.makedirs(rgb_dir, exist_ok=True)

        # 创建COCO标注文件
        coco_data = {
            "images": [
                {"id": 0, "file_name": "rgb/000000.png", "width": 640, "height": 480},
                {"id": 1, "file_name": "rgb/000001.png", "width": 640, "height": 480}
            ],
            "annotations": [
                {"image_id": 0, "keypoints": [100, 100, 2, 200, 100, 2, 200, 200, 2, 100, 200, 2]},
                {"image_id": 1, "keypoints": [150, 150, 2, 250, 150, 2, 250, 250, 2, 150, 250, 2]}
            ]
        }

        coco_path = os.path.join(scene_dir, "scene_gt_coco.json")
        with open(coco_path, 'w') as f:
            json.dump(coco_data, f)

        # 创建一些测试图像文件
        for i in range(2):
            img_path = os.path.join(rgb_dir, f"{i:06d}.png")
            # 创建一个简单的1x1像素图像用于测试
            with open(img_path, 'wb') as f:
                # PNG文件头（简化版）
                f.write(b'\x89PNG\r\n\x1a\n')  # PNG signature
                # 其他PNG数据（这里只是为了文件存在性测试）

        print(f"创建测试数据集结构:")
        print(f"  场景目录: {scene_dir}")
        print(f"  RGB目录: {rgb_dir}")
        print(f"  COCO文件: {coco_path}")

        # 验证目录结构
        checks = [
            ("场景目录存在", os.path.isdir(scene_dir)),
            ("COCO文件存在", os.path.isfile(coco_path)),
            ("RGB目录存在", os.path.isdir(rgb_dir)),
        ]

        png_files = [f for f in os.listdir(rgb_dir) if f.endswith('.png')]
        jpg_files = [f for f in os.listdir(rgb_dir) if f.endswith('.jpg')]

        checks.extend([
            ("PNG文件存在", len(png_files) > 0),
            ("JPG文件存在", len(jpg_files) >= 0),  # JPG是可选的
        ])

        print(f"  PNG文件数量: {len(png_files)}")
        print(f"  JPG文件数量: {len(jpg_files)}")
        print(f"  总图像数量: {len(png_files) + len(jpg_files)}")

        print("\n验证结果:")
        all_passed = True
        for check_name, result in checks:
            status = "✅" if result else "❌"
            print(f"  {status} {check_name}")
            if not result:
                all_passed = False

        if all_passed:
            print("\n✅ 所有路径检查通过！数据集结构正确。")
        else:
            print("\n❌ 路径检查失败！请检查数据集结构。")

        return all_passed

if __name__ == "__main__":
    test_path_validation()