#!/usr/bin/env python3
"""
从3D模型和pose信息生成8角点标注
用于BOP格式的角点检测训练数据

修复版：正确处理单位转换和多物体场景
"""
import os
import json
import numpy as np
import argparse
from PIL import Image


def load_ply_corners(model_path):
    """
    从PLY模型文件加载并计算其8个角点（axis-aligned bounding box）
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
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
            for line in content.strip().split('\n'):
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
                    except:
                        pass
        else:
            import struct
            for _ in range(vertex_count):
                data = f.read(12)
                if len(data) == 12:
                    x, y, z = struct.unpack('fff', data)
                    vertices.append([x, y, z])
    
    if not vertices:
        raise ValueError(f"无法从模型中读取顶点: {model_path}")
    
    vertices = np.array(vertices, dtype=np.float32)
    
    # 计算bounding box
    min_coords = vertices.min(axis=0)
    max_coords = vertices.max(axis=0)
    
    # 8个角点
    corners_3d = np.array([
        [min_coords[0], min_coords[1], min_coords[2]],
        [max_coords[0], min_coords[1], min_coords[2]],
        [min_coords[0], max_coords[1], min_coords[2]],
        [max_coords[0], max_coords[1], min_coords[2]],
        [min_coords[0], min_coords[1], max_coords[2]],
        [max_coords[0], min_coords[1], max_coords[2]],
        [min_coords[0], max_coords[1], max_coords[2]],
        [max_coords[0], max_coords[1], max_coords[2]],
    ], dtype=np.float32)
    
    size = max_coords - min_coords
    print(f"  模型尺寸: [{size[0]:.2f}, {size[1]:.2f}, {size[2]:.2f}] mm")
    
    return corners_3d


def project_corners_to_2d(corners_3d, R, t, K, width, height):
    """
    将3D角点投影到2D图像平面
    
    Args:
        corners_3d: (8, 3) 3D角点坐标
        R: (3, 3) 旋转矩阵
        t: (3,) 平移向量（已转换为单位）
        K: (3, 3) 相机内参
        width, height: 图像尺寸
    
    返回：
        corners_2d: (8, 2) 2D角点坐标
        visibility: (8,) 可见性标志 (0=不可见, 2=可见)
    """
    # 转换到相机坐标系
    corners_cam = (R @ corners_3d.T).T + t
    
    # 投影到图像平面
    corners_2d_hom = (K @ corners_cam.T).T
    corners_2d = corners_2d_hom[:, :2] / corners_2d_hom[:, 2:3]
    
    # 检查可见性
    visibility = []
    for i, (x, y) in enumerate(corners_2d):
        # 深度必须为正且在图像范围内
        if corners_cam[i, 2] > 0 and 0 <= x < width and 0 <= y < height:
            visibility.append(2)  # 可见
        else:
            visibility.append(0)  # 不可见
    
    return corners_2d.astype(np.float32), visibility


def load_all_models(models_dir):
    """
    加载所有模型文件
    
    返回：
        dict: {obj_id: corners_3d}
    """
    models = {}
    
    for filename in sorted(os.listdir(models_dir)):
        if filename.endswith('.ply'):
            try:
                # 从文件名提取obj_id (e.g., obj_000001.ply -> 1)
                obj_id = int(filename.replace('obj_', '').replace('.ply', ''))
                model_path = os.path.join(models_dir, filename)
                
                print(f"\n加载模型: {filename}")
                corners_3d = load_ply_corners(model_path)
                models[obj_id] = corners_3d
                
            except Exception as e:
                print(f"  警告: 无法加载 {filename}: {e}")
    
    return models


def generate_8corner_coco(scene_dir, models_dir, output_path=None, t_scale=1.0, target_obj_id=None):
    """
    为场景生成8角点COCO标注
    
    Args:
        scene_dir: 场景目录（包含scene_gt.json, scene_camera.json, rgb/）
        models_dir: 模型目录（包含obj_xxxxxx.ply）
        output_path: 输出JSON路径（默认保存到scene_dir/scene_gt_coco.json）
        t_scale: 平移向量缩放因子（如果数据是米，需要乘以1000转换为毫米）
        target_obj_id: 如果只处理特定物体，指定obj_id；None表示处理所有物体
    """
    scene_gt_path = os.path.join(scene_dir, 'scene_gt.json')
    scene_camera_path = os.path.join(scene_dir, 'scene_camera.json')
    rgb_dir = os.path.join(scene_dir, 'rgb')
    
    # 检查文件是否存在
    if not os.path.exists(scene_gt_path):
        raise FileNotFoundError(f"找不到场景标注文件: {scene_gt_path}")
    if not os.path.exists(scene_camera_path):
        raise FileNotFoundError(f"找不到相机参数文件: {scene_camera_path}")
    if not os.path.exists(rgb_dir):
        raise FileNotFoundError(f"找不到图像目录: {rgb_dir}")
    
    # 加载场景数据
    print(f"\n加载场景数据...")
    with open(scene_gt_path, 'r') as f:
        scene_gt = json.load(f)
    
    with open(scene_camera_path, 'r') as f:
        scene_camera = json.load(f)
    
    # 加载所有模型
    print(f"\n加载3D模型...")
    models = load_all_models(models_dir)
    print(f"\n共加载 {len(models)} 个模型")
    
    # 获取相机内参（假设所有图像使用相同相机）
    first_cam = list(scene_camera.values())[0]
    K = np.array(first_cam['cam_K'], dtype=np.float32).reshape(3, 3)
    print(f"\n相机内参:\n{K}")
    
    # 获取图像尺寸
    image_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith(('.jpg', '.png', '.jpeg', '.bmp'))])
    if not image_files:
        raise ValueError(f"没有找到图像文件: {rgb_dir}")
    
    with Image.open(os.path.join(rgb_dir, image_files[0])) as img:
        width, height = img.width, img.height
    print(f"图像尺寸: {width} x {height}")
    print(f"图像数量: {len(image_files)}")
    
    # 初始化COCO格式
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{
            "id": 1,
            "name": "object",
            "keypoints": [f"corner_{i}" for i in range(8)],
            "skeleton": []
        }]
    }
    
    annotation_id = 0
    valid_images = 0
    skipped_no_model = 0
    skipped_no_visible = 0
    
    print(f"\n处理图像...")
    
    # 处理每个图像
    for img_filename in image_files:
        img_id = int(os.path.splitext(img_filename)[0])
        
        # 添加图像信息
        coco_data["images"].append({
            "id": img_id,
            "file_name": f"rgb/{img_filename}",
            "width": width,
            "height": height
        })
        
        # 检查该图像是否有标注
        if str(img_id) not in scene_gt:
            continue
        
        anns = scene_gt[str(img_id)]
        
        # 处理每个物体实例
        for ann in anns:
            obj_id = ann['obj_id']
            
            # 如果指定了目标物体，只处理该物体
            if target_obj_id is not None and obj_id != target_obj_id:
                continue
            
            # 检查是否有对应的模型
            if obj_id not in models:
                skipped_no_model += 1
                continue
            
            # 获取位姿
            R = np.array(ann['cam_R_m2c'], dtype=np.float32).reshape(3, 3)
            t = np.array(ann['cam_t_m2c'], dtype=np.float32) * t_scale  # 应用单位转换
            
            # 投影8角点到2D
            corners_2d, visibility = project_corners_to_2d(
                models[obj_id], R, t, K, width, height
            )
            
            # 检查是否有可见角点
            visible_count = sum(1 for v in visibility if v == 2)
            if visible_count == 0:
                skipped_no_visible += 1
                continue
            
            valid_images += 1
            
            # 构建keypoints格式 [x1,y1,v1, x2,y2,v2, ...]
            keypoints = []
            for (x, y), v in zip(corners_2d, visibility):
                keypoints.extend([float(x), float(y), v])
            
            # 计算bbox（只使用可见角点）
            visible_corners = [c for c, v in zip(corners_2d, visibility) if v == 2]
            if visible_corners:
                x_coords = [c[0] for c in visible_corners]
                y_coords = [c[1] for c in visible_corners]
                bbox_x = min(x_coords)
                bbox_y = min(y_coords)
                bbox_w = max(x_coords) - bbox_x
                bbox_h = max(y_coords) - bbox_y
            else:
                bbox_x = bbox_y = bbox_w = bbox_h = 0
            
            coco_data["annotations"].append({
                "id": annotation_id,
                "image_id": img_id,
                "category_id": 1,
                "obj_id": obj_id,
                "bbox": [bbox_x, bbox_y, bbox_w, bbox_h],
                "keypoints": keypoints,
                "num_keypoints": 8,
                "area": bbox_w * bbox_h,
                "iscrowd": 0
            })
            
            annotation_id += 1
    
    # 保存结果
    if output_path is None:
        output_path = os.path.join(scene_dir, 'scene_gt_coco.json')
    
    with open(output_path, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"\n{'='*50}")
    print(f"生成完成!")
    print(f"  - 输出文件: {output_path}")
    print(f"  - 总图像数量: {len(image_files)}")
    print(f"  - 有效标注数量: {len(coco_data['annotations'])}")
    print(f"  - 跳过（无模型）: {skipped_no_model}")
    print(f"  - 跳过（无可见角点）: {skipped_no_visible}")
    print(f"{'='*50}\n")
    
    return output_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='从BOP格式生成8角点COCO标注')
    parser.add_argument('--scene_dir', type=str, required=True,
                        help='场景目录路径（包含scene_gt.json, scene_camera.json, rgb/）')
    parser.add_argument('--models_dir', type=str, required=True,
                        help='模型目录路径（包含obj_xxxxxx.ply）')
    parser.add_argument('--output_path', type=str, default=None,
                        help='输出JSON文件路径（默认: scene_dir/scene_gt_coco.json）')
    parser.add_argument('--t_scale', type=float, default=1.0,
                        help='平移向量缩放因子（如果数据是米，设置为1000）')
    parser.add_argument('--target_obj_id', type=int, default=None,
                        help='只处理特定物体ID（默认处理所有物体）')
    
    args = parser.parse_args()
    
    generate_8corner_coco(
        scene_dir=args.scene_dir,
        models_dir=args.models_dir,
        output_path=args.output_path,
        t_scale=args.t_scale,
        target_obj_id=args.target_obj_id
    )
