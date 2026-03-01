#!/usr/bin/env python3
"""
检查3D模型的实际尺寸，诊断bounding box问题
"""
import os
import numpy as np
import struct


def load_ply_vertices(model_path):
    """加载PLY文件的所有顶点"""
    vertices = []
    
    with open(model_path, 'rb') as f:
        header_lines = []
        vertex_count = 0
        
        # 读取header
        for line in f:
            line = line.decode('utf-8').strip()
            header_lines.append(line)
            
            if line.startswith('element vertex'):
                vertex_count = int(line.split()[-1])
            elif line == 'end_header':
                break
        
        # 判断格式
        is_ascii = any('ascii' in line.lower() for line in header_lines)
        
        if is_ascii:
            # ASCII格式
            content = f.read().decode('utf-8')
            lines = content.strip().split('\n')
            for line in lines[:vertex_count]:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
                    except:
                        pass
        else:
            # Binary格式
            for _ in range(vertex_count):
                data = f.read(12)
                if len(data) == 12:
                    x, y, z = struct.unpack('fff', data)
                    vertices.append([x, y, z])
    
    return np.array(vertices, dtype=np.float32)


def analyze_model(model_path):
    """分析3D模型的尺寸"""
    print(f"\n{'='*60}")
    print(f"模型: {os.path.basename(model_path)}")
    print(f"{'='*60}")
    
    vertices = load_ply_vertices(model_path)
    print(f"顶点数量: {len(vertices)}")
    
    if len(vertices) == 0:
        print("错误: 没有顶点数据!")
        return
    
    # 计算bounding box
    min_coords = vertices.min(axis=0)
    max_coords = vertices.max(axis=0)
    size = max_coords - min_coords
    center = (min_coords + max_coords) / 2
    
    print(f"\nBounding Box:")
    print(f"  Min: [{min_coords[0]:.6f}, {min_coords[1]:.6f}, {min_coords[2]:.6f}]")
    print(f"  Max: [{max_coords[0]:.6f}, {max_coords[1]:.6f}, {max_coords[2]:.6f}]")
    print(f"  Center: [{center[0]:.6f}, {center[1]:.6f}, {center[2]:.6f}]")
    print(f"  Size: [{size[0]:.6f}, {size[1]:.6f}, {size[2]:.6f}]")
    
    # 判断单位
    max_size = max(size)
    if max_size < 0.1:
        print(f"\n⚠️ 警告: 模型尺寸只有 {max_size:.6f}，可能是以米为单位但数值太小")
        print(f"  建议检查模型是否正确")
    elif max_size < 10:
        print(f"\n⚠️ 模型尺寸 {max_size:.2f}，可能是以厘米或分米为单位")
    elif max_size < 1000:
        print(f"\n✓ 模型尺寸 {max_size:.2f}，可能是以毫米为单位")
    else:
        print(f"\n⚠️ 模型尺寸 {max_size:.2f}，可能是以微米或其他单位")
    
    # 计算8个角点
    corners = np.array([
        [min_coords[0], min_coords[1], min_coords[2]],
        [max_coords[0], min_coords[1], min_coords[2]],
        [min_coords[0], max_coords[1], min_coords[2]],
        [max_coords[0], max_coords[1], min_coords[2]],
        [min_coords[0], min_coords[1], max_coords[2]],
        [max_coords[0], min_coords[1], max_coords[2]],
        [min_coords[0], max_coords[1], max_coords[2]],
        [max_coords[0], max_coords[1], max_coords[2]],
    ])
    
    print(f"\n8个角点 (模型坐标系):")
    for i, c in enumerate(corners):
        print(f"  角点 {i}: [{c[0]:.6f}, {c[1]:.6f}, {c[2]:.6f}]")
    
    # 检查角点之间的距离
    print(f"\n角点之间的距离:")
    for i in range(8):
        for j in range(i+1, 8):
            dist = np.linalg.norm(corners[i] - corners[j])
            if dist > 0.001:  # 只显示有意义的距离
                print(f"  角点{i}-角点{j}: {dist:.6f}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='检查3D模型尺寸')
    parser.add_argument('--models_dir', type=str, required=True,
                        help='模型目录路径')
    
    args = parser.parse_args()
    
    # 分析所有模型
    model_files = sorted([f for f in os.listdir(args.models_dir) if f.endswith('.ply')])
    
    print(f"找到 {len(model_files)} 个模型文件")
    
    for model_file in model_files:
        model_path = os.path.join(args.models_dir, model_file)
        try:
            analyze_model(model_path)
        except Exception as e:
            print(f"\n错误: 无法分析 {model_file}: {e}")
