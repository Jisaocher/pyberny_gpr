#!/usr/bin/env python3
"""
批量生成 3D 分子结构 HTML 可视化

用法：
    python draw_structure3D.py [structures_dir]

参数：
    structures_dir: structures 目录路径（默认为 output 下的所有 method 目录）

示例：
    python draw_structure3D.py
    python draw_structure3D.py /path/to/output

目录结构适配：
    output/
    └── {smiles}_{perturb}/
        ├── pyberny/
        │   └── structures/
        │       ├── pyberny_initial.xyz
        │       └── pyberny_final.xyz
        └── hybrid_{ai_method}/    # 如 hybrid_gpr, hybrid_krr 等
            └── structures/
                ├── hybrid_ggpr_initial.xyz
                └── hybrid_ggpr_final.xyz
"""
import os
import sys
import yaml

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from visualization.structure3d import MoleculeVisualizer3D


def find_structures_dirs():
    """
    查找 structures 目录

    适配新的目录结构：output/{smiles}_{perturb}/{method}/structures/
    
    优先级：
    1. 命令行参数指定的路径
    2. config/default_config.yaml 中定义的 output.save_dir 下的所有目录
    3. ./output/*/structures（旧结构兼容）
    4. ./output/*/{pyberny,hybrid}/structures（新结构）

    返回：
        structures_dirs: structures 目录列表
    """
    structures_dirs = []

    # 1. 检查命令行参数
    if len(sys.argv) > 1:
        arg_path = sys.argv[1]
        if os.path.exists(arg_path):
            # 如果是 structures 目录
            if os.path.basename(arg_path) == 'structures':
                structures_dirs.append(arg_path)
                print(f"使用命令行参数指定的目录：{arg_path}")
                return structures_dirs
            # 如果是 method 目录（pyberny/hybrid），查找其下的 structures
            elif os.path.basename(arg_path) == 'pyberny' or os.path.basename(arg_path).startswith('hybrid'):
                structures_path = os.path.join(arg_path, 'structures')
                if os.path.isdir(structures_path):
                    structures_dirs.append(structures_path)
                    print(f"使用命令行参数指定的目录：{structures_path}")
                    return structures_dirs
            # 如果是 smiles 目录，查找其下的所有 method/structures
            elif os.path.isdir(arg_path):
                for method in os.listdir(arg_path):
                    method_path = os.path.join(arg_path, method)
                    if not os.path.isdir(method_path):
                        continue
                    # 匹配 pyberny 或 hybrid 开头的目录
                    if method == 'pyberny' or method.startswith('hybrid'):
                        structures_path = os.path.join(method_path, 'structures')
                        if os.path.isdir(structures_path):
                            structures_dirs.append(structures_path)
                # 兼容旧结构
                old_structures_path = os.path.join(arg_path, 'structures')
                if os.path.isdir(old_structures_path) and old_structures_path not in structures_dirs:
                    structures_dirs.append(old_structures_path)

                if structures_dirs:
                    print(f"在目录 {arg_path} 下找到 {len(structures_dirs)} 个 structures:")
                    for d in structures_dirs:
                        print(f"  - {d}")
                    return structures_dirs
            # 如果是 output 目录，查找所有子目录
            elif os.path.basename(arg_path) == 'output':
                for smiles_dir in os.listdir(arg_path):
                    smiles_path = os.path.join(arg_path, smiles_dir)
                    if not os.path.isdir(smiles_path):
                        continue
                    # 新结构：output/{smiles}/{method}/structures
                    for method in os.listdir(smiles_path):
                        method_path = os.path.join(smiles_path, method)
                        if not os.path.isdir(method_path):
                            continue
                        # 匹配 pyberny 或 hybrid 开头的目录
                        if method == 'pyberny' or method.startswith('hybrid'):
                            structures_path = os.path.join(method_path, 'structures')
                            if os.path.isdir(structures_path):
                                structures_dirs.append(structures_path)
                    # 兼容旧结构
                    old_structures_path = os.path.join(smiles_path, 'structures')
                    if os.path.isdir(old_structures_path) and old_structures_path not in structures_dirs:
                        structures_dirs.append(old_structures_path)

                if structures_dirs:
                    print(f"在 output 目录下找到 {len(structures_dirs)} 个 structures 目录:")
                    for d in structures_dirs:
                        print(f"  - {d}")
                    return structures_dirs

    # 2. 尝试从配置文件读取
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'default_config.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        base_save_dir = config.get('output', {}).get('save_dir', './output')

        # 查找 base_save_dir 下的所有目录（新结构）
        if os.path.exists(base_save_dir):
            for smiles_item in os.listdir(base_save_dir):
                smiles_path = os.path.join(base_save_dir, smiles_item)
                if not os.path.isdir(smiles_path):
                    continue
                    
                # 新结构：output/{smiles}/{method}/structures
                # method 可以是 pyberny 或 hybrid_{ai_method}（如 hybrid_gpr）
                for method in os.listdir(smiles_path):
                    method_path = os.path.join(smiles_path, method)
                    if not os.path.isdir(method_path):
                        continue
                    # 匹配 pyberny 或 hybrid 开头的目录
                    if method == 'pyberny' or method.startswith('hybrid'):
                        structures_path = os.path.join(method_path, 'structures')
                        if os.path.isdir(structures_path):
                            structures_dirs.append(structures_path)
                
                # 兼容旧结构：output/{smiles}/structures
                old_structures_path = os.path.join(smiles_path, 'structures')
                if os.path.isdir(old_structures_path) and old_structures_path not in structures_dirs:
                    structures_dirs.append(old_structures_path)

            if structures_dirs:
                print(f"使用配置文件定义的目录，找到 {len(structures_dirs)} 个 structures:")
                for d in structures_dirs:
                    print(f"  - {d}")
                return structures_dirs

    # 3. 使用默认路径 ./output
    output_dir = './output'
    if os.path.exists(output_dir):
        for smiles_item in os.listdir(output_dir):
            smiles_path = os.path.join(output_dir, smiles_item)
            if not os.path.isdir(smiles_path):
                continue
                
            # 新结构：output/{smiles}/{method}/structures
            # method 可以是 pyberny 或 hybrid_{ai_method}（如 hybrid_gpr）
            for method in os.listdir(smiles_path):
                method_path = os.path.join(smiles_path, method)
                if not os.path.isdir(method_path):
                    continue
                # 匹配 pyberny 或 hybrid 开头的目录
                if method == 'pyberny' or method.startswith('hybrid'):
                    structures_path = os.path.join(method_path, 'structures')
                    if os.path.isdir(structures_path):
                        structures_dirs.append(structures_path)
            
            # 兼容旧结构：output/{smiles}/structures
            old_structures_path = os.path.join(smiles_path, 'structures')
            if os.path.isdir(old_structures_path) and old_structures_path not in structures_dirs:
                structures_dirs.append(old_structures_path)

        if structures_dirs:
            print(f"使用默认目录，找到 {len(structures_dirs)} 个 structures:")
            for d in structures_dirs:
                print(f"  - {d}")
            return structures_dirs

    # 目录不存在
    print(f"Error: structures directories not found")
    return []


def main():
    """主函数"""
    # 查找 structures 目录
    structures_dirs = find_structures_dirs()
    
    if not structures_dirs:
        print("Please run main.py first to generate XYZ files, or specify the structures directory.")
        print("\nUsage:")
        print("  python draw_structure3D.py [output_dir|smiles_dir|structures_dir]")
        print("\nExamples:")
        print("  python draw_structure3D.py                    # 处理 output 下所有分子")
        print("  python draw_structure3D.py output             # 处理 output 下所有分子")
        print("  python draw_structure3D.py output/CCO         # 处理乙醇分子")
        print("  python draw_structure3D.py output/CCO/structures  # 处理乙醇的 structures")
        sys.exit(1)
    
    # 批量生成 HTML
    total_html_files = []
    
    for structures_dir in structures_dirs:
        # 查找所有 XYZ 文件
        xyz_files = []
        for filename in sorted(os.listdir(structures_dir)):
            if filename.endswith('.xyz'):
                xyz_files.append(os.path.join(structures_dir, filename))
        
        if not xyz_files:
            print(f"No XYZ files found in {structures_dir}")
            continue
        
        print(f"\n处理目录：{structures_dir}")
        print(f"找到 {len(xyz_files)} 个 XYZ 文件:")
        for xyz_file in xyz_files:
            print(f"  - {os.path.basename(xyz_file)}")
        
        # 创建可视化器
        vis = MoleculeVisualizer3D(
            font_size=14,
            figure_size=(800, 600)
        )
        
        # 批量生成 HTML
        print(f"开始生成 HTML 文件...")
        html_files = []
        for xyz_file in xyz_files:
            html_file = vis.visualize_from_xyz(xyz_file)
            if html_file:
                html_files.append(html_file)
        
        # 输出统计
        print(f"完成！生成了 {len(html_files)} 个 HTML 文件:")
        for html_file in html_files:
            print(f"  ✓ {os.path.basename(html_file)}")
        
        total_html_files.extend(html_files)
    
    print(f"\n{'='*60}")
    print(f"总计生成了 {len(total_html_files)} 个 HTML 文件")
    print(f"提示：用浏览器打开 HTML 文件可查看交互式 3D 分子结构。")


if __name__ == '__main__':
    main()
