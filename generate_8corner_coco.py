#!/usr/bin/env python3
"""
从3D模型和pose信息生成8角点标注
用于BOP格式的角点检测训练数据

修复版：处理各种边界情况
"""
import os
import json
import numpy as np
import argparse
from collections import Counter
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
    
    min_coords = vertices.min(axis=0)
    max_coords = vertices.max(axis=0)
    
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
    
    print(f"3D模型角点 (mm):")
    print(f"  顶点数: {len(vertices)}")
    print(f"  Bounding box: min={min_coords}, max={max_coords}")
    
    return corners_3d


def project_corners_to_2d(corners_3d, R, t, K, width, height):
    """
    将3D角点投影到2D图像平面，并检查可见性
    返回：角点坐标列表，可见性列表
    """
    corners_3d_hom = np.hstack([corners_3d, np.ones((corners_3d.shape[0], 1))])
    
    transform = np.eye(4)
    transform[:3, :3] = R
    transform[:3, 3] = t
    
    corners_cam = (transform @ corners_3d_hom.T).T
    
    corners_2d_hom = K @ corners_cam[:, :3].T
    
    # 检查深度是否有效（必须在相机前面）
    valid_depth = corners_cam[:, 2] > 0
    
    corners_2d = corners_2d_hom[:2, :] / corners_2d_hom[2:, :]
    corners_2d = corners_2d.T
    
    valid_corners = []
    visibility = []
    
    for i, (x, y) in enumerate(corners_2d):
        # 角点必须在图像范围内且深度有效
        if valid_depth[i] and 0 <= x < width and 0 <= y < height:
            valid_corners.append([float(x), float(y)])
            visibility.append(2)  # 可见
        else:
            # 角点在图像外，不添加（这是关键修复）
            pass
    
    return valid_corners, visibility


def get_image_files(rgb_dir):
    """
    获取所有图像文件，按文件名排序
    """
    files = []
    for f in os.listdir(rgb_dir):
        if f.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            files.append(f)
    return sorted(files)


def generate_8corner_coco(scene_dir, models_dir, output_path=None):
    """
    为场景生成8角点COCO标注
    """
    scene_gt_path = os.path.join(scene_dir, 'scene_gt.json')
    scene_coco_path = os.path.join(scene_dir, 'scene_gt_coco.json')
    scene_camera_path = os.path.join(scene_dir, 'scene_camera.json')
    rgb_dir = os.path.join(scene_dir, 'rgb')
    
    with open(scene_gt_path, 'r') as f:
        scene_gt = json.load(f)
    
    with open(scene_camera_path, 'r') as f:
        scene_camera = json.load(f)
    
    first_cam = list(scene_camera.values())[0]
    K = np.array(first_cam['cam_K'], dtype=np.float32).reshape(3, 3)
    print(f"相机内参:\n{K}")
    
    # 获取图像尺寸
    image_files = get_image_files(rgb_dir)
    if image_files:
        with Image.open(os.path.join(rgb_dir, image_files[0])) as img:
            width, height = img.width, img.height
        print(f"图像尺寸: {width} x {height}")
        print(f"图像数量: {len(image_files)}")
    else:
        raise ValueError(f"没有找到图像文件: {rgb_dir}")
    
    model_path = os.path.join(models_dir, 'obj_000001.ply')
    model_corners_3d = load_ply_corners(model_path)
    
    coco_data = {
        "images": [],
        "annotations": []
    }
    
    annotation_id = 0
    
    # 用于统计
    valid_images = 0
    skipped_no_corners = 0
    skipped_outside = 0
    
    # 遍历所有图像文件
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
            skipped_no_corners += 1
            continue
        
        anns = scene_gt[str(img_id)]
        
        all_corners = []
        all_visibility = []
        
        for ann in anns:
            obj_id = ann['obj_id']
            R = np.array(ann['cam_R_m2c'], dtype=np.float32).reshape(3, 3)
            t = np.array(ann['cam_t_m2c'], dtype=np.float32)
            
            corners_2d, visibility = project_corners_to_2d(
                model_corners_3d, R, t, K, width, height
            )
            
            if corners_2d:  # 只添加有效的（可见的）角点
                all_corners.extend(corners_2d)
                all_visibility.extend(visibility)
        
        # 只有当有至少1个有效角点时才创建标注
        if len(all_corners) >= 1:
            valid_images += 1
            
            # 转换为keypoints格式
            keypoints = []
            for (x, y), v in zip(all_corners, all_visibility):
                keypoints.extend([x, y, v])
            
            # 计算bbox
            x_coords = [c[0] for c in all_corners]
            y_coords = [c[1] for c in all_corners]
            bbox_x = min(x_coords)
            bbox_y = min(y_coords)
            bbox_w = max(x_coords) - bbox_x
            bbox_h = max(y_coords) - bbox_y
            
            coco_data["annotations"].append({
                "id": annotation_id,
                "image_id": img_id,
                "category_id": 1,
                "obj_id": obj_id,
                "bbox": [bbox_x, bbox_y, bbox_w, bbox_h],
                "keypoints": keypoints,
                "num_keypoints": len(all_corners),
                "area": bbox_w * bbox_h,
                "iscrowd": 0
            })
            
            annotation_id += 1
        else:
            skipped_outside += 1
    
    if output_path is None:
        output_path = scene_coco_path
    
    with open(output_path, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"\n生成完成!")
    print(f"  - 总图像数量: {len(image_files)}")
    print(f"  - 有效标注数量: {len(coco_data['annotations'])}")
    print(f"  - 跳过（无标注）: {skipped_no_corners}")
    print(f"  - 跳过（无可见角点）: {skipped_outside}")
    print(f"  - 输出文件: {output_path}")
    
    if coco_data["annotations"]:
        corner_counts = [ann['num_keypoints'] for ann in coco_data['annotations']]
        print(f"\n角点数量统计:")
        print(f"  - 最少: {min(corner_counts)}")
        print(f"  - 最多: {max(corner_counts)}")
        print(f"  - 平均: {np.mean(corner_counts):.2f}")
        
        count_dist = Counter(corner_counts)
        for num, cnt in sorted(count_dist.items()):
            print(f"    {num}个角点: {cnt}个标注 ({cnt*100/len(corner_counts):.1f}%)")
    else:
        print("\n警告: 没有生成任何有效标注!")
    
    return coco_data


def main():
    parser = argparse.ArgumentParser(description='生成8角点COCO标注')
    parser.add_argument('--scene_dir', type=str,
                        default='/nas2/home/qianqian/projects/HCCEPose/demo-bin-pick-back/train_pbr/000000',
                        help='场景目录')
    parser.add_argument('--models_dir', type=str,
                        default='/nas2/home/qianqian/projects/HCCEPose/demo-bin-pick-back/models',
                        help='models目录')
    parser.add_argument('--output', type=str, default=None,
                        help='输出COCO JSON路径')
    args = parser.parse_args()
    
    generate_8corner_coco(args.scene_dir, args.models_dir, args.output)


if __name__ == '__main__':
    main()
