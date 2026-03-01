#!/usr/bin/env python3
"""
详细调试3D到2D投影过程
"""
import os
import json
import numpy as np
import struct
from PIL import Image
import matplotlib.pyplot as plt


def load_ply_vertices(model_path):
    """加载PLY文件的所有顶点"""
    vertices = []
    with open(model_path, 'rb') as f:
        header_lines = []
        vertex_count = 0
        
        for line in f:
            line = line.decode('utf-8').strip()
            header_lines.append(line)
            
            if line.startswith('element vertex'):
                vertex_count = int(line.split()[-1])
            elif line == 'end_header':
                break
        
        is_ascii = any('ascii' in line.lower() for line in header_lines)
        
        if is_ascii:
            content = f.read().decode('utf-8')
            for line in content.strip().split('\n')[:vertex_count]:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
                    except:
                        pass
        else:
            for _ in range(vertex_count):
                data = f.read(12)
                if len(data) == 12:
                    x, y, z = struct.unpack('fff', data)
                    vertices.append([x, y, z])
    
    return np.array(vertices, dtype=np.float32)


def debug_projection(scene_dir, models_dir, img_id=0, model_scale=10.0, t_scale=1000.0):
    """详细调试投影过程"""
    
    scene_gt_path = os.path.join(scene_dir, 'scene_gt.json')
    scene_camera_path = os.path.join(scene_dir, 'scene_camera.json')
    rgb_dir = os.path.join(scene_dir, 'rgb')
    
    # 加载数据
    with open(scene_gt_path, 'r') as f:
        scene_gt = json.load(f)
    
    with open(scene_camera_path, 'r') as f:
        scene_camera = json.load(f)
    
    # 获取相机参数
    cam_data = scene_camera[str(img_id)]
    K = np.array(cam_data['cam_K'], dtype=np.float32).reshape(3, 3)
    
    print("="*70)
    print(f"调试图像 {img_id}")
    print("="*70)
    print(f"\n相机内参 K:\n{K}")
    
    # 加载图像 - 尝试不同的文件名格式
    img_filename = f"{img_id:06d}.png"
    img_path = os.path.join(rgb_dir, img_filename)
    
    # 如果找不到，尝试查找实际文件名
    if not os.path.exists(img_path):
        rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.png')])
        if rgb_files:
            img_filename = rgb_files[0]  # 使用第一个找到的图像
            img_path = os.path.join(rgb_dir, img_filename)
            print(f"  注意: 使用图像 {img_filename} 代替 {img_id:06d}.png")
    
    image = Image.open(img_path).convert('RGB')
    width, height = image.size
    print(f"\n图像尺寸: {width} x {height}")
    
    # 获取该图像的标注
    anns = scene_gt.get(str(img_id), [])
    print(f"\n该图像有 {len(anns)} 个物体")
    
    fig, axes = plt.subplots(len(anns), 3, figsize=(15, 5*len(anns)))
    if len(anns) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, ann in enumerate(anns):
        obj_id = ann['obj_id']
        
        print(f"\n{'='*70}")
        print(f"物体 {idx}: obj_id={obj_id}")
        print(f"{'='*70}")
        
        # 加载模型
        model_path = os.path.join(models_dir, f'obj_{obj_id:06d}.ply')
        if not os.path.exists(model_path):
            print(f"  错误: 找不到模型 {model_path}")
            continue
        
        vertices = load_ply_vertices(model_path) * model_scale
        
        # 计算bounding box
        min_coords = vertices.min(axis=0)
        max_coords = vertices.max(axis=0)
        
        print(f"\n  3D模型信息 (缩放后):")
        print(f"    顶点数: {len(vertices)}")
        print(f"    Bounding box min: {min_coords}")
        print(f"    Bounding box max: {max_coords}")
        print(f"    尺寸: {max_coords - min_coords}")
        
        # 生成8角点
        corners_3d = np.array([
            [min_coords[0], min_coords[1], min_coords[2]],
            [max_coords[0], min_coords[1], min_coords[2]],
            [min_coords[0], max_coords[1], min_coords[2]],
            [max_coords[0], max_coords[1], min_coords[2]],
            [min_coords[0], min_coords[1], max_coords[2]],
            [max_coords[0], min_coords[1], max_coords[2]],
            [min_coords[0], max_coords[1], max_coords[2]],
            [max_coords[0], max_coords[1], max_coords[2]],
        ])
        
        print(f"\n  8个3D角点 (模型坐标系):")
        for i, c in enumerate(corners_3d):
            print(f"    角点 {i}: {c}")
        
        # 获取位姿
        R = np.array(ann['cam_R_m2c'], dtype=np.float32).reshape(3, 3)
        t = np.array(ann['cam_t_m2c'], dtype=np.float32) * t_scale
        
        print(f"\n  位姿:")
        print(f"    旋转矩阵 R:\n{R}")
        print(f"    平移向量 t: {t}")
        print(f"    t的模长: {np.linalg.norm(t):.2f}")
        
        # 投影到相机坐标系
        corners_cam = (R @ corners_3d.T).T + t
        
        print(f"\n  相机坐标系中的8个角点:")
        for i, c in enumerate(corners_cam):
            print(f"    角点 {i}: {c} (深度={c[2]:.2f})")
        
        # 投影到图像
        corners_2d_hom = (K @ corners_cam.T).T
        corners_2d = corners_2d_hom[:, :2] / corners_2d_hom[:, 2:3]
        
        print(f"\n  2D投影结果:")
        for i, (x, y) in enumerate(corners_2d):
            in_image = 0 <= x < width and 0 <= y < height
            status = "✓ 在图像内" if in_image else "✗ 在图像外"
            print(f"    角点 {i}: ({x:8.2f}, {y:8.2f}) {status}")
        
        # 可视化
        ax1 = axes[idx, 0]
        ax1.imshow(image)
        ax1.set_title(f'物体 {idx} - 原始图像')
        
        # 绘制所有顶点（半透明）
        # 投影所有顶点
        vertices_cam = (R @ vertices.T).T + t
        vertices_2d_hom = (K @ vertices_cam.T).T
        vertices_2d = vertices_2d_hom[:, :2] / vertices_2d_hom[:, 2:3]
        
        valid_vertices = [(x, y) for x, y in vertices_2d if 0 <= x < width and 0 <= y < height]
        if valid_vertices:
            xs, ys = zip(*valid_vertices)
            ax1.scatter(xs, ys, c='blue', s=1, alpha=0.3, label='顶点')
        
        # 绘制8角点
        for i, (x, y) in enumerate(corners_2d):
            if 0 <= x < width and 0 <= y < height:
                ax1.plot(x, y, 'r+', markersize=15, markeredgewidth=2)
                ax1.text(x+5, y-5, str(i), color='red', fontsize=10)
        
        ax1.legend()
        ax1.axis('off')
        
        # 角点分布图
        ax2 = axes[idx, 1]
        valid_corners = [(x, y) for x, y in corners_2d if 0 <= x < width and 0 <= y < height]
        if valid_corners:
            xs, ys = zip(*valid_corners)
            ax2.scatter(xs, ys, c='red', s=200)
            for i, (x, y) in enumerate(corners_2d):
                if 0 <= x < width and 0 <= y < height:
                    ax2.text(x, y, str(i), fontsize=12, ha='center', va='center')
        
        ax2.set_xlim(0, width)
        ax2.set_ylim(height, 0)
        ax2.set_aspect('equal')
        ax2.set_title('8角点分布')
        ax2.grid(True, alpha=0.3)
        
        # 3D可视化
        ax3 = axes[idx, 2]
        ax3.remove()
        ax3 = fig.add_subplot(len(anns), 3, idx*3+3, projection='3d')
        
        # 绘制3D角点
        ax3.scatter(corners_3d[:, 0], corners_3d[:, 1], corners_3d[:, 2], c='red', s=100)
        for i, c in enumerate(corners_3d):
            ax3.text(c[0], c[1], c[2], str(i), fontsize=10)
        
        # 绘制相机位置
        ax3.scatter([t[0]], [t[1]], [t[2]], c='blue', s=200, marker='^', label='相机')
        
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        ax3.set_title('3D视图')
        ax3.legend()
    
    plt.tight_layout()
    plt.savefig(f'debug_projection_img{img_id}.png', dpi=150, bbox_inches='tight')
    print(f"\n可视化结果已保存到 debug_projection_img{img_id}.png")
    plt.show()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='调试3D到2D投影')
    parser.add_argument('--scene_dir', type=str, required=True,
                        help='场景目录路径')
    parser.add_argument('--models_dir', type=str, required=True,
                        help='模型目录路径')
    parser.add_argument('--img_id', type=int, default=0,
                        help='要调试的图像ID')
    parser.add_argument('--model_scale', type=float, default=10.0,
                        help='模型缩放因子')
    parser.add_argument('--t_scale', type=float, default=1000.0,
                        help='平移向量缩放因子')
    
    args = parser.parse_args()
    
    debug_projection(args.scene_dir, args.models_dir, args.img_id, 
                     args.model_scale, args.t_scale)
