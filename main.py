#!/usr/bin/env python3
"""
PyBerny-GPR 混合优化项目主程序
分子几何构型优化 - PyBerny 与 GPR 混合策略
"""
import os
import sys
import argparse
import yaml
from typing import Dict, Any, Optional

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.molecule import Molecule
from core.calculator import QuantumCalculator
from optimizers.pyberny_optimizer import PyBernyOptimizer
from optimizers.pyberny_baseline import PyBernyBaselineOptimizer, run_pyberny_baseline_optimization
from optimizers.hybrid import HybridOptimizer, run_hybrid_optimization
from visualization.structure3d import MoleculeVisualizer3D
from visualization.plots import OptimizationPlotter
from utils.io_utils import OutputManager, create_output_manager


def _get_ai_method_suffix(ai_method: str) -> str:
    """获取 AI 方法的简短后缀"""
    # 目前只支持 gradient_predicting 一种 AI 方法
    return 'gpr' if ai_method else ''


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    加载配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        config: 配置字典
    """
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'default_config.yaml')

    if not os.path.exists(config_path):
        print(f"Warning: Config file not found: {config_path}")
        return {}

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


def merge_configs(default: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """合并配置"""
    result = default.copy()
    for key, value in override.items():
        if isinstance(value, dict) and key in result:
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


def run_optimization(method: str, molecule: Molecule, config: Dict[str, Any],
                     output_manager: OutputManager,
                     ai_method: str = None) -> Dict[str, Any]:
    """
    运行优化

    Args:
        method: 优化方法 ('pyberny' 或 'hybrid')
        molecule: 初始分子
        config: 配置字典
        output_manager: 输出管理器
        ai_method: AI 方法类型

    Returns:
        results: 优化结果
    """
    print(f"\n{'='*70}")
    print(f"运行优化：{method.upper()}")
    print(f"{'='*70}")

    # 创建计算器
    calculator = QuantumCalculator(
        basis=config.get('calculation', {}).get('basis', 'cc-pvdz'),
        method=config.get('calculation', {}).get('method', 'RHF'),
        unit=config.get('calculation', {}).get('unit', 'angstrom')
    )

    # 选择优化器
    if method == 'pyberny':
        # 纯 PyBerny 基准方法（一次性运行到底）
        optimizer = PyBernyBaselineOptimizer(config)
    elif method == 'hybrid':
        optimizer = HybridOptimizer(config)
    else:
        raise ValueError(f"Unknown method: {method}")

    # 执行优化
    history = optimizer.optimize(molecule, calculator)

    # 保存结果
    metadata = {
        'method': method,
        'molecule': molecule.name,
        'smiles': molecule.smiles,
        'n_atoms': molecule.n_atoms,
        'config': config
    }

    # 保存历史
    output_manager.save_history(history, method, metadata)

    # 保存轨迹
    output_manager.save_trajectory(history, method, molecule.atom_symbols)

    # 保存详细迭代信息
    if config.get('output', {}).get('save_details', True):
        output_manager.save_iteration_details(history, method, molecule.atom_symbols)

    # 获取最优结构（使用梯度最小的坐标）
    best_iteration = history.get_best_iteration(metric='gradient')
    if best_iteration is not None:
        best_coords = best_iteration.coords
        best_mol = Molecule(
            molecule.atom_symbols,
            best_coords.reshape(-1, 3),
            molecule.smiles,
            f"{molecule.name}_best_{method}"
        )
        output_manager.save_final_structure(best_mol, method, ai_method)

    # 生成图表
    vis_config = config.get('visualization', {})
    plotter = OptimizationPlotter(
        font_size=vis_config.get('font_size', 14),
        figure_size=tuple(vis_config.get('figure_size', [12, 8])),
        dpi=vis_config.get('dpi', 300),
        ai_method=ai_method
    )

    plots_dir = os.path.join(output_manager.method_dir, 'plots')

    # 构建图表标题前缀
    if ai_method:
        ai_suffix = _get_ai_method_suffix(ai_method)
        title_prefix = f"{method}_{ai_suffix} - "
    else:
        title_prefix = f"{method} - "

    plotter.plot_all(history, plots_dir, title_prefix, ai_method=ai_method)

    # 返回结果
    best = history.get_best_iteration('energy')
    results = {
        'method': method,
        'converged': history.converged,
        'initial_energy': history.iterations[0].energy if history.iterations else None,
        'final_energy': best.energy if best else None,
        'final_gradient_norm': best.gradient_norm if best else None,
        'total_iterations': len(history.iterations),
    }

    return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='PyBerny-GPR 混合优化项目')

    parser.add_argument('--method', type=str, default='pyberny',
                       choices=['pyberny', 'hybrid'],
                       help='优化方法：pyberny (纯 PyBerny 基准) 或 hybrid (PyBerny+GPR 混合)')
    parser.add_argument('--molecule', type=str, default='ethanol',
                       help='分子名称 (default: ethanol)')
    parser.add_argument('--smiles', type=str, default=None,
                       help='SMILES 字符串（覆盖 --molecule）')
    parser.add_argument('--perturb', type=float, default=0.0,
                       help='初始扰动强度 (Å) (default: 0.0)')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子 (default: 42)')
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径')
    parser.add_argument('--output', type=str, default=None,
                       help='输出目录')
    parser.add_argument('--max-iter', type=int, default=None,
                       help='最大迭代次数')
    parser.add_argument('--threshold', type=float, default=None,
                       help='收敛阈值')

    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    # 命令行参数覆盖配置
    if args.output:
        if 'output' not in config:
            config['output'] = {}
        config['output']['save_dir'] = args.output

    if args.max_iter:
        if 'optimizer' not in config:
            config['optimizer'] = {}
        config['optimizer']['max_iterations'] = args.max_iter

    if args.threshold:
        if 'optimizer' not in config:
            config['optimizer'] = {}
        config['optimizer']['convergence_threshold'] = args.threshold

    # 分子设置
    if 'molecule' not in config:
        config['molecule'] = {}

    if args.smiles:
        config['molecule']['smiles'] = args.smiles

    if args.seed != 42:
        config['molecule']['seed'] = args.seed

    if args.perturb != 0.0:
        config['molecule']['perturb'] = args.perturb

    # 获取 AI 方法类型（用于输出文件命名）
    ai_method = None
    if args.method == 'hybrid':
        ai_method = config.get('gpr', {}).get('type', 'gradient_predicting')

    # 获取分子信息
    smiles = config['molecule']['smiles']
    seed = config['molecule']['seed']
    perturb = config['molecule']['perturb']

    # 修改输出目录：添加 smiles 和扰动水平
    if 'output' not in config:
        config['output'] = {}

    original_save_dir = config['output'].get('save_dir', './output')

    if perturb == int(perturb):
        perturb_str = str(int(perturb))
    else:
        perturb_str = f"{perturb:.1f}".rstrip('0').rstrip('.')

    dir_name = f"{smiles}_{perturb_str}"
    config['output']['save_dir'] = os.path.join(original_save_dir, dir_name)

    # 创建输出管理器
    output_manager = create_output_manager(config, ai_method=ai_method, method_name=args.method)

    print(f"\n{'='*70}")
    print("PyBerny-GPR 混合优化项目")
    print(f"{'='*70}")
    print(f"分子：{smiles}")
    print(f"扰动：{perturb} Å")
    print(f"种子：{seed}")
    print(f"方法：{args.method}")
    if ai_method:
        print(f"AI 方法：{ai_method}")
    print(f"输出目录：{output_manager.save_dir}")
    print(f"{'='*70}")

    # 创建分子
    molecule = Molecule.from_smiles(smiles, seed=seed, perturb_strength=perturb)

    print(f"\n初始结构:")
    print(f"  原子数：{molecule.n_atoms}")
    print(f"  自由度：{molecule.n_atoms * 3}")

    # 保存初始结构
    output_manager.save_initial_structure(molecule, args.method, ai_method)

    # 运行优化
    results = run_optimization(args.method, molecule, config, output_manager, ai_method)

    # 打印结果
    print(f"\n{'='*70}")
    print("优化结果")
    print(f"{'='*70}")
    print(f"方法：{results['method']}")
    print(f"收敛：{results['converged']}")
    print(f"初始能量：{results['initial_energy']:.10f} Hartree")
    print(f"最终能量：{results['final_energy']:.10f} Hartree")
    print(f"最终梯度：{results['final_gradient_norm']:.6f} Hartree/Å")
    print(f"总迭代次数：{results['total_iterations']}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
