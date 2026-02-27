#!/usr/bin/env python3
"""
从3D模型和pose信息生成8角点标注
用于BOP格式的角点检测训练数据
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
    with open(model_path, 'r') as f:
        in_vertex_section = False
        for line in f:
            line = line.strip()
            if line == 'end_header':
                in_vertex_section = False
                break
            if in_vertex_section:
                parts = line.split()
                vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
            if line == 'vertex':
                in_vertex_section = True
    
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
    print(f"  Bounding box: min={min_coords}, max={max_coords}")
    
    return corners_3d


def project_corners_to_2d(corners_3d, R, t, K):
    """
    将3D角点投影到2D图像平面
    """
    corners_3d_hom = np.hstack([corners_3d, np.ones((corners_3d.shape[0], 1))])
    
    transform = np.eye(4)
    transform[:3, :3] = R
    transform[:3, 3] = t
    
    corners_cam = (transform @ corners_3d_hom.T).T
    
    corners_2d_hom = K @ corners_cam[:, :3].T
    corners_2d = corners_2d_hom[:2, :] / corners_2d_hom[2, :]
    corners_2d = corners_2d.T
    
    return corners_2d


def get_image_size(rgb_dir, img_id):
    """
    从图像文件获取尺寸
    """
    files = sorted(os.listdir(rgb_dir))
    if img_id < len(files):
        img_path = os.path.join(rgb_dir, files[img_id])
        with Image.open(img_path) as img:
            return img.width, img.height
    return None, None


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
    
    width, height = get_image_size(rgb_dir, 0)
    if width and height:
        print(f"图像尺寸: {width} x {height}")
    else:
        print("警告: 无法获取图像尺寸")
    
    model_path = os.path.join(models_dir, 'obj_000001.ply')
    model_corners_3d = load_ply_corners(model_path)
    
    coco_data = {
        "images": [],
        "annotations": []
    }
    
    annotation_id = 0
    
    for img_id, anns in scene_gt.items():
        img_id = int(img_id)
        
        files = sorted(os.listdir(rgb_dir))
        if img_id < len(files):
            file_name = files[img_id]
            img_path = os.path.join(rgb_dir, file_name)
            with Image.open(img_path) as img:
                width, height = img.width, img.height
        else:
            file_name = f"{img_id:06d}.jpg"
        
        coco_data["images"].append({
            "id": img_id,
            "file_name": f"rgb/{file_name}",
            "width": width,
            "height": height
        })
        
        for ann in anns:
            obj_id = ann['obj_id']
            R = np.array(ann['cam_R_m2c'], dtype=np.float32).reshape(3, 3)
            t = np.array(ann['cam_t_m2c'], dtype=np.float32)
            
            corners_2d = project_corners_to_2d(model_corners_3d, R, t, K)
            
            valid_corners = []
            visibility = []
            for i, (x, y) in enumerate(corners_2d):
                if 0 <= x < width and 0 <= y < height:
                    valid_corners.append([float(x), float(y)])
                    visibility.append(2)
                else:
                    valid_corners.append([float(np.clip(x, 0, width-1)), float(np.clip(y, 0, height-1))])
                    visibility.append(1)
            
            keypoints = []
            for (x, y), v in zip(valid_corners, visibility):
                keypoints.extend([x, y, v])
            
            x_coords = [c[0] for c in valid_corners]
            y_coords = [c[1] for c in valid_corners]
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
                "num_keypoints": len(valid_corners),
                "area": bbox_w * bbox_h,
                "iscrowd": 0
            })
            
            annotation_id += 1
    
    if output_path is None:
        output_path = scene_coco_path
    
    with open(output_path, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"\n生成完成!")
    print(f"  - 图像数量: {len(coco_data['images'])}")
    print(f"  - 标注数量: {len(coco_data['annotations'])}")
    print(f"  - 输出文件: {output_path}")
    
    corner_counts = [ann['num_keypoints'] for ann in coco_data['annotations']]
    print(f"\n角点数量统计:")
    print(f"  - 最少: {min(corner_counts)}")
    print(f"  - 最多: {max(corner_counts)}")
    print(f"  - 平均: {np.mean(corner_counts):.2f}")
    
    count_dist = Counter(corner_counts)
    for num, cnt in sorted(count_dist.items()):
        print(f"    {num}个角点: {cnt}个标注 ({cnt*100/len(corner_counts):.1f}%)")
    
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
