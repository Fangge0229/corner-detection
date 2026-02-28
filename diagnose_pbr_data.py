#!/usr/bin/env python3
"""
诊断PBR渲染数据问题
检查scene_gt.json、scene_camera.json和PLY模型的一致性
"""
import json
import os
import sys
import numpy as np
from PIL import Image

def load_ply_vertices(model_path):
    """加载PLY模型的顶点"""
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
    
    return np.array(vertices, dtype=np.float32)

def check_pbr_data(scene_dir, models_dir):
    """检查PBR渲染数据的完整性"""
    
    print(f"\n{'='*70}")
    print(f"诊断PBR数据: {scene_dir}")
    print('='*70)
    
    # 加载文件
    scene_gt_path = os.path.join(scene_dir, 'scene_gt.json')
    scene_camera_path = os.path.join(scene_dir, 'scene_camera.json')
    rgb_dir = os.path.join(scene_dir, 'rgb')
    
    with open(scene_gt_path, 'r') as f:
        scene_gt = json.load(f)
    
    with open(scene_camera_path, 'r') as f:
        scene_camera = json.load(f)
    
    # 检查模型
    model_path = os.path.join(models_dir, 'obj_000001.ply')
    if os.path.exists(model_path):
        vertices = load_ply_vertices(model_path)
        print(f"\n3D模型信息:")
        print(f"  - 顶点数: {len(vertices)}")
        print(f"  - Bounding Box:")
        print(f"    min: [{vertices[:,0].min():.2f}, {vertices[:,1].min():.2f}, {vertices[:,2].min():.2f}]")
        print(f"    max: [{vertices[:,0].max():.2f}, {vertices[:,1].max():.2f}, {vertices[:,2].max():.2f}]")
        print(f"    size: [{vertices[:,0].max()-vertices[:,0].min():.2f}, {vertices[:,1].max()-vertices[:,1].min():.2f}, {vertices[:,2].max()-vertices[:,2].min():.2f}]")
    
    # 检查前5个图像
    print(f"\n前5个图像的相机和姿态信息:")
    for i, img_id in enumerate(list(scene_gt.keys())[:5]):
        print(f"\n  图像 {img_id}:")
        
        # 相机参数
        cam = scene_camera.get(img_id, {})
        K = cam.get('cam_K', [])
        if K:
            print(f"    相机内参 K:")
            print(f"      [{K[0]:.2f}, {K[1]:.2f}, {K[2]:.2f}]")
            print(f"      [{K[3]:.2f}, {K[4]:.2f}, {K[5]:.2f}]")
            print(f"      [{K[6]:.2f}, {K[7]:.2f}, {K[8]:.2f}]")
        
        # 图像尺寸
        img_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith(('.jpg', '.png'))])
        if int(img_id) < len(img_files):
            img_path = os.path.join(rgb_dir, img_files[int(img_id)])
            with Image.open(img_path) as img:
                print(f"    图像尺寸: {img.width} x {img.height}")
        
        # 姿态信息
        gt_list = scene_gt.get(img_id, [])
        for j, gt in enumerate(gt_list):
            R = gt.get('cam_R_m2c', [])
            t = gt.get('cam_t_m2c', [])
            obj_id = gt.get('obj_id', -1)
            
            print(f"    物体 {j} (obj_id={obj_id}):")
            print(f"      旋转矩阵 R:")
            print(f"        [{R[0]:.4f}, {R[1]:.4f}, {R[2]:.4f}]")
            print(f"        [{R[3]:.4f}, {R[4]:.4f}, {R[5]:.4f}]")
            print(f"        [{R[6]:.4f}, {R[7]:.4f}, {R[8]:.4f}]")
            print(f"      平移向量 t: [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}]")
            print(f"      平移向量长度: {np.linalg.norm(t):.2f}")
            
            # 检查平移向量的单位（应该是毫米）
            if np.linalg.norm(t) < 10:
                print(f"      ⚠️ 警告: 平移向量太小，可能是米而不是毫米！")
            elif np.linalg.norm(t) > 10000:
                print(f"      ⚠️ 警告: 平移向量太大，可能单位有误！")
    
    print(f"\n{'='*70}\n")

if __name__ == '__main__':
    if len(sys.argv) > 2:
        scene_dir = sys.argv[1]
        models_dir = sys.argv[2]
    else:
        scene_dir = '/nas2/home/qianqian/projects/HCCEPose/demo-bin-picking/train_pbr/000000'
        models_dir = '/nas2/home/qianqian/projects/HCCEPose/demo-bin-picking/models'
    
    check_pbr_data(scene_dir, models_dir)
