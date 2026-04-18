#!/usr/bin/env python3
"""
批量生成 3D 分子结构 HTML 可视化

功能：
    遍历指定目录下所有 structures 目录中的 .xyz 文件，
    生成对应的 .html 交互式 3D 分子结构文件。

用法：
    python draw_structure3D.py [path]

参数：
    path: 可选，指定要处理的目录
         - 如果是 structures 目录，直接处理该目录
         - 如果是其他目录，递归查找其下所有 structures 目录
         - 默认为 ./output

示例：
    python draw_structure3D.py                    # 处理 ./output 下所有 structures
    python draw_structure3D.py ./output           # 同上
    python draw_structure3D.py output/CCO_0.01    # 处理指定分子目录
    python draw_structure3D.py path/to/structures # 处理指定 structures 目录

目录结构兼容：
    兼容任意层级的目录结构，只要包含 structures 目录即可：
    - output/{smiles}_{perturb}/{method}/structures/
    - output/{smiles}_{perturb}/structures/
    - output/structures/
    - 或任意其他结构
"""
import os
import sys
from pathlib import Path


def find_xyz_files(structures_dir: str) -> list:
    """
    查找 structures 目录中的所有 XYZ 文件

    Args:
        structures_dir: structures 目录路径

    Returns:
        xyz_files: XYZ 文件路径列表
    """
    xyz_files = []
    if not os.path.isdir(structures_dir):
        return xyz_files

    for filename in sorted(os.listdir(structures_dir)):
        if filename.endswith('.xyz'):
            xyz_files.append(os.path.join(structures_dir, filename))

    return xyz_files


def find_all_structures_dirs(base_path: str) -> list:
    """
    递归查找指定路径下的所有 structures 目录

    Args:
        base_path: 基础路径

    Returns:
        structures_dirs: structures 目录列表
    """
    structures_dirs = []

    # 如果直接指定了 structures 目录
    if os.path.basename(base_path) == 'structures' and os.path.isdir(base_path):
        structures_dirs.append(base_path)
        return structures_dirs

    # 递归查找所有 structures 子目录
    for root, dirs, files in os.walk(base_path):
        if 'structures' in dirs:
            structures_path = os.path.join(root, 'structures')
            if os.path.isdir(structures_path):
                structures_dirs.append(structures_path)

    return structures_dirs


def main():
    """主函数"""
    # 确定基础路径
    if len(sys.argv) > 1:
        base_path = sys.argv[1]
    else:
        base_path = './output'

    if not os.path.exists(base_path):
        print(f"Error: Directory not found: {base_path}")
        print("\nUsage:")
        print("  python draw_structure3D.py [path]")
        print("\nExamples:")
        print("  python draw_structure3D.py                    # 处理 ./output 目录")
        print("  python draw_structure3D.py ./output           # 同上")
        print("  python draw_structure3D.py output/CCO_0.01    # 处理指定分子目录")
        print("  python draw_structure3D.py path/to/structures # 处理指定 structures 目录")
        sys.exit(1)

    # 查找所有 structures 目录
    structures_dirs = find_all_structures_dirs(base_path)

    if not structures_dirs:
        # 检查是否直接指定了 structures 目录
        if os.path.basename(base_path) == 'structures':
            structures_dirs = [base_path]

    if not structures_dirs:
        print(f"No 'structures' directories found in: {base_path}")
        print("\nUsage:")
        print("  python draw_structure3D.py [path]")
        print("\nExamples:")
        print("  python draw_structure3D.py                    # 处理 ./output 目录")
        print("  python draw_structure3D.py ./output           # 同上")
        print("  python draw_structure3D.py output/CCO_0.01    # 处理指定分子目录")
        print("  python draw_structure3D.py path/to/structures # 处理指定 structures 目录")
        sys.exit(1)

    print(f"在 '{base_path}' 下找到 {len(structures_dirs)} 个 structures 目录:")
    for d in structures_dirs:
        print(f"  - {d}")

    # 导入可视化器
    try:
        from visualization.structure3d import MoleculeVisualizer3D
    except ImportError:
        print("Error: Cannot import MoleculeVisualizer3D")
        sys.exit(1)

    # 批量生成 HTML
    total_html_files = 0

    for structures_dir in structures_dirs:
        # 查找所有 XYZ 文件
        xyz_files = find_xyz_files(structures_dir)

        if not xyz_files:
            print(f"\nNo XYZ files found in {structures_dir}")
            continue

        print(f"\n处理目录：{structures_dir}")
        print(f"找到 {len(xyz_files)} 个 XYZ 文件")

        # 创建可视化器
        vis = MoleculeVisualizer3D(
            font_size=14,
            figure_size=(800, 600)
        )

        # 批量生成 HTML
        html_count = 0
        for xyz_file in xyz_files:
            html_file = vis.visualize_from_xyz(xyz_file)
            if html_file:
                html_count += 1

        print(f"生成了 {html_count} 个 HTML 文件")
        total_html_files += html_count

    print(f"\n{'='*60}")
    print(f"总计生成了 {total_html_files} 个 HTML 文件")
    print(f"提示：用浏览器打开 HTML 文件可查看交互式 3D 分子结构。")


if __name__ == '__main__':
    main()
