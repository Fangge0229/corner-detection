#!/usr/bin/env python3
"""
演示正确的BOP数据集路径设置
展示如何避免"RGB图像目录中没有找到PNG文件"错误
"""

def demonstrate_correct_paths():
    """演示正确的路径设置"""

    print("=== BOP数据集路径设置指南 ===\n")

    print("❌ 错误的路径设置:")
    print("  train_ubuntu.sh --scene_dir /path/to/dataset/rgb/")
    print("  错误: 直接指向rgb目录，缺少scene_gt_coco.json\n")

    print("✅ 正确的路径设置:")
    print("  train_ubuntu.sh --scene_dir /path/to/dataset/scene_000000/")
    print("  正确: 指向包含rgb/子目录的场景目录\n")

    print("📁 正确的BOP数据集结构:")
    print("""
    /path/to/bop/dataset/
    └── lm/                    # 对象类型目录
        └── train_pbr/         # 训练数据类型
            └── 000000/       # 场景目录 (这个作为--scene_dir)
                ├── rgb/      # 图像目录
                │   ├── 000000.png
                │   ├── 000001.png
                │   └── ...
                ├── scene_gt_coco.json    # COCO标注
                ├── scene_camera.json     # 相机参数
                ├── scene_gt.json         # GT poses
                └── scene_gt_info.json    # pose信息
    """)

    print("🔍 路径检查逻辑:")
    print("1. 检查场景目录是否存在")
    print("2. 检查scene_gt_coco.json是否存在")
    print("3. 检查rgb/子目录是否存在")
    print("4. 检查rgb/目录中的PNG/JPG文件")
    print("5. 统计并显示数据集信息\n")

    print("💡 提示:")
    print("- 确保--scene_dir指向包含rgb/的目录")
    print("- 支持PNG和JPG格式的图像")
    print("- COCO标注文件是必需的")
    print("- 其他文件(scene_camera.json等)是可选的\n")

    print("🚀 使用示例:")
    print("  ./train_ubuntu.sh --scene_dir /data/bop/lm/train_pbr/000000")
    print("  python3 train_bop_ubuntu.py --scene_dir /data/bop/lm/train_pbr/000000")

if __name__ == "__main__":
    demonstrate_correct_paths()