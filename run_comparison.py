#!/usr/bin/env python3
"""
对比运行脚本
运行 PyBerny 和 PyBerny+GPR 混合优化，进行横向对比
"""
import os
import sys
import argparse
import yaml
from datetime import datetime
from typing import Dict, Any, List

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.molecule import Molecule
from optimizers.pyberny_baseline import PyBernyBaselineOptimizer
from optimizers.hybrid import HybridOptimizer
from core.calculator import QuantumCalculator
from visualization.plots import OptimizationPlotter
from visualization.structure3d import MoleculeVisualizer3D
from utils.io_utils import OutputManager


def load_config(config_path: str = None) -> Dict[str, Any]:
    """加载配置文件"""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'default_config.yaml')

    if not os.path.exists(config_path):
        return {}

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


def run_comparison(smiles: str = 'CCO', perturb: float = 0.0, seed: int = 24,
                   config: Dict[str, Any] = None,
                   output_dir: str = None) -> Dict[str, Any]:
    """
    运行对比实验

    Args:
        smiles: 分子 SMILES
        perturb: 扰动强度
        seed: 随机种子
        config: 配置字典
        output_dir: 输出目录

    Returns:
        results: 对比结果
    """
    if config is None:
        config = load_config()

    # 设置输出目录
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"./output/comparison_{timestamp}"

    output_manager = OutputManager(output_dir, format='json')

    print(f"\n{'='*70}")
    print("PyBerny vs PyBerny+GPR 对比实验")
    print(f"{'='*70}")
    print(f"分子：{smiles}")
    print(f"扰动：{perturb} Å")
    print(f"种子：{seed}")
    print(f"输出目录：{output_dir}")
    print(f"{'='*70}")

    # 创建分子（使用相同的初始结构）
    molecule = Molecule.from_smiles(smiles, seed=seed, perturb_strength=perturb)

    print(f"\n初始结构:")
    print(f"  原子数：{molecule.n_atoms}")
    print(f"  自由度：{molecule.n_atoms * 3}")

    # 创建计算器
    calculator = QuantumCalculator(
        basis=config.get('calculation', {}).get('basis', 'cc-pvdz'),
        method=config.get('calculation', {}).get('method', 'RHF'),
        unit=config.get('calculation', {}).get('unit', 'angstrom')
    )

    # 存储结果
    histories = {}
    results = {}

    # ========== 运行 PyBerny 基准方法 ==========
    print(f"\n{'='*70}")
    print("运行 PyBerny 基准方法（完整 BFGS，一次性运行）")
    print(f"{'='*70}")

    pyberny_optimizer = PyBernyBaselineOptimizer(config)
    pyberny_history = pyberny_optimizer.optimize(molecule, calculator)
    histories['pyberny'] = pyberny_history

    # 保存 PyBerny 结果
    pyberny_output = output_manager.create_method_output('pyberny')
    pyberny_output.save_history(pyberny_history, 'pyberny', {'method': 'pyberny'})
    pyberny_output.save_trajectory(pyberny_history, 'pyberny', molecule.atom_symbols)
    pyberny_output.save_final_structure(
        Molecule(molecule.atom_symbols, 
                pyberny_history.get_best_iteration('gradient').coords.reshape(-1, 3),
                molecule.smiles, 'pyberny_final'),
        'pyberny', None
    )

    # ========== 运行 PyBerny+GPR 混合方法 ==========
    print(f"\n{'='*70}")
    print("运行 PyBerny+GPR 混合优化")
    print(f"{'='*70}")

    ai_method = config.get('gpr', {}).get('type', 'gradient_predicting')
    hybrid_optimizer = HybridOptimizer(config)
    hybrid_history = hybrid_optimizer.optimize(molecule, calculator)
    histories['hybrid'] = hybrid_history

    # 保存 Hybrid 结果
    hybrid_output = output_manager.create_method_output('hybrid')
    hybrid_output.save_history(hybrid_history, 'hybrid', {'method': 'hybrid', 'ai_method': ai_method})
    hybrid_output.save_trajectory(hybrid_history, 'hybrid', molecule.atom_symbols)
    hybrid_output.save_final_structure(
        Molecule(molecule.atom_symbols,
                hybrid_history.get_best_iteration('gradient').coords.reshape(-1, 3),
                molecule.smiles, 'hybrid_final'),
        'hybrid', ai_method
    )

    # ========== 生成对比图表 ==========
    print(f"\n{'='*70}")
    print("生成对比图表")
    print(f"{'='*70}")

    # 混合策略模式：只绘制外层迭代数据（内层数据没有实际 PySCF 计算）
    plotter = OptimizationPlotter(
        font_size=config.get('visualization', {}).get('font_size', 14),
        figure_size=tuple(config.get('visualization', {}).get('figure_size', [12, 8])),
        dpi=config.get('visualization', {}).get('dpi', 300),
        hybrid_mode=True  # 混合策略启用外层数据过滤
    )

    plots_dir = os.path.join(output_dir, 'plots')
    plotter.plot_comparison(histories, plots_dir, ['pyberny', 'hybrid'])

    # ========== 总结 ==========
    print(f"\n{'='*70}")
    print("对比实验完成")
    print(f"{'='*70}")
    
    summary = []
    for method_name, history in histories.items():
        best = history.get_best_iteration('gradient')
        summary.append({
            'method': method_name,
            'converged': history.converged,
            'final_energy': best.energy if best else None,
            'final_gradient': best.gradient_norm if best else None,
            'iterations': len(history.iterations)
        })
        print(f"\n{method_name}:")
        print(f"  收敛：{history.converged}")
        print(f"  最终能量：{best.energy:.10f} Hartree" if best else "  最终能量：N/A")
        print(f"  最终梯度：{best.gradient_norm:.6f} Hartree/Å" if best else "  最终梯度：N/A")
        print(f"  迭代次数：{len(history.iterations)}")

    # 保存对比总结
    summary_data = {
        'smiles': smiles,
        'perturb': perturb,
        'seed': seed,
        'results': summary
    }
    
    import json
    summary_file = os.path.join(output_dir, f'comparison_summary_{timestamp}.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n对比总结已保存：{summary_file}")
    print(f"{'='*70}")

    return summary_data


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='PyBerny vs PyBerny+GPR 对比实验')
    parser.add_argument('--smiles', type=str, default='CCO',
                       help='分子 SMILES (default: CCO)')
    parser.add_argument('--perturb', type=float, default=0.0,
                       help='初始扰动强度 (Å) (default: 0.0)')
    parser.add_argument('--seed', type=int, default=24,
                       help='随机种子 (default: 24)')
    parser.add_argument('--output', type=str, default=None,
                       help='输出目录')
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径')

    args = parser.parse_args()

    config = load_config(args.config)

    # 命令行参数覆盖
    if args.perturb != 0.0:
        if 'molecule' not in config:
            config['molecule'] = {}
        config['molecule']['perturb'] = args.perturb

    if args.seed != 24:
        if 'molecule' not in config:
            config['molecule'] = {}
        config['molecule']['seed'] = args.seed

    run_comparison(
        smiles=args.smiles,
        perturb=config.get('molecule', {}).get('perturb', 0.0),
        seed=config.get('molecule', {}).get('seed', 24),
        config=config,
        output_dir=args.output
    )


if __name__ == '__main__':
    main()
