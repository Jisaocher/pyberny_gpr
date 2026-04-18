"""
输入输出工具
处理数据保存和加载
"""
import os
import json
import csv
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime

from core.molecule import Molecule, OptimizationHistory, IterationData


class OutputManager:
    """
    输出管理器
    
    负责保存优化结果、轨迹、图表等
    """
    
    def __init__(self, save_dir: str, format: str = 'json',
                 ai_method: str = None, method_name: str = None):
        """
        初始化输出管理器

        Args:
            save_dir: 保存目录（如 output/CCO_0.1）
            format: 输出格式 ('json', 'csv', 'npy')
            ai_method: AI 方法类型（'gpr' 等）
            method_name: 优化方法名称（'pyberny'/'hybrid'）
        """
        self.save_dir = save_dir
        self.format = format
        self.ai_method = ai_method
        self.method_name = method_name

        # 确定方法专用目录：output/{smiles}_{perturb}/{method}/
        # 混合策略：添加 AI 方法信息，如 output/{smiles}_{perturb}/hybrid_gpr/
        if method_name:
            if method_name == 'hybrid' and ai_method:
                # 混合策略：方法名包含 AI 方法信息
                ai_suffix = self._get_ai_method_suffix() if ai_method else ai_method
                self.method_dir = os.path.join(save_dir, f"{method_name}_{ai_suffix}")
            else:
                self.method_dir = os.path.join(save_dir, method_name)
        else:
            self.method_dir = save_dir

        # 创建目录
        os.makedirs(self.method_dir, exist_ok=True)
        os.makedirs(os.path.join(self.method_dir, 'trajectories'), exist_ok=True)
        os.makedirs(os.path.join(self.method_dir, 'plots'), exist_ok=True)
        os.makedirs(os.path.join(self.method_dir, 'structures'), exist_ok=True)

    def save_history(self, history: OptimizationHistory, method_name: str,
                     metadata: Dict[str, Any] = None) -> str:
        """
        保存优化历史

        Args:
            history: 优化历史
            method_name: 方法名称
            metadata: 额外元数据

        Returns:
            filepath: 保存的文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 构建文件名：如果是 hybrid 方法且指定了 AI 方法，则添加后缀
        if method_name == 'hybrid' and self.ai_method:
            # 将 AI 方法类型转换为简短后缀
            ai_suffix = self._get_ai_method_suffix()
            filename = f"{method_name}_{ai_suffix}_{timestamp}"
        else:
            filename = f"{method_name}_{timestamp}"

        if self.format == 'json':
            return self._save_json(history, filename, metadata)
        elif self.format == 'csv':
            return self._save_csv(history, filename, metadata)
        else:
            return self._save_json(history, filename, metadata)
    
    def _get_ai_method_suffix(self) -> str:
        """获取 AI 方法的简短后缀"""
        suffix_map = {
            'gpr': 'gpr',
            'gradient_predicting': 'gpr',
            'simple': 'gpr',
            'gradient': 'ggpr',
            'random_forest': 'rf',
            'neural_network': 'nn'
        }
        return suffix_map.get(self.ai_method, self.ai_method)
    
    def _save_json(self, history: OptimizationHistory, filename: str,
                   metadata: Dict[str, Any] = None) -> str:
        """保存为 JSON 格式"""
        filepath = os.path.join(self.method_dir, f"{filename}.json")

        data = history.to_dict()

        # 添加元数据
        if metadata:
            data['metadata'] = metadata

        # 添加统计信息
        data['statistics'] = self._compute_statistics(history)

        # 自定义 JSON 编码器，处理 numpy 类型
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

        print(f"优化历史已保存：{filepath}")
        return filepath
    
    def _save_csv(self, history: OptimizationHistory, filename: str,
                  metadata: Dict[str, Any] = None) -> str:
        """保存为 CSV 格式"""
        filepath = os.path.join(self.save_dir, f"{filename}.csv")
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # 写入表头
            writer.writerow([
                'iteration', 'energy', 'gradient_norm', 'displacement_norm',
                'energy_change', 'timestamp'
            ])
            
            # 写入数据
            prev_energy = None
            for it in history.iterations:
                displacement_norm = np.linalg.norm(it.displacement) if it.displacement is not None else 0.0
                energy_change = prev_energy - it.energy if prev_energy is not None else 0.0
                
                writer.writerow([
                    it.iteration,
                    f"{it.energy:.12f}",
                    f"{it.gradient_norm:.10f}",
                    f"{displacement_norm:.10f}",
                    f"{energy_change:.12f}",
                    it.timestamp
                ])
                
                prev_energy = it.energy
        
        # 同时保存元数据为 JSON
        if metadata:
            meta_filepath = os.path.join(self.save_dir, f"{filename}_metadata.json")
            with open(meta_filepath, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
        
        print(f"优化历史已保存：{filepath}")
        return filepath
    
    def _compute_statistics(self, history: OptimizationHistory) -> Dict[str, Any]:
        """计算统计信息"""
        if not history.iterations:
            return {}

        energies = history.get_energies()
        grad_norms = history.get_gradient_norms()

        stats = {
            'total_iterations': len(history),
            'initial_energy': float(energies[0]),
            'final_energy': float(energies[-1]),
            'best_energy': float(np.min(energies)),
            'energy_improvement': float(energies[0] - energies[-1]),
            'initial_gradient_norm': float(grad_norms[0]),
            'final_gradient_norm': float(grad_norms[-1]),
            'best_gradient_norm': float(np.min(grad_norms)),
            'converged': bool(history.converged),
            'convergence_iteration': int(history.convergence_iteration) if history.convergence_iteration is not None else None
        }

        if history.start_time and history.end_time:
            stats['computation_time'] = float(history.end_time - history.start_time)

        return stats
    
    def save_trajectory(self, history: OptimizationHistory, method_name: str,
                        atom_symbols: List[str]) -> str:
        """
        保存优化轨迹为 XYZ 格式

        Args:
            history: 优化历史
            method_name: 方法名称 ('pyberny'/'hybrid')
            atom_symbols: 原子符号列表

        Returns:
            filepath: 保存的文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 构建文件名：如果是 hybrid 方法且指定了 AI 方法，则添加后缀
        if method_name == 'hybrid' and self.ai_method:
            ai_suffix = self._get_ai_method_suffix()
            filename = f"{method_name}_{ai_suffix}_trajectory_{timestamp}.xyz"
        else:
            filename = f"{method_name}_trajectory_{timestamp}.xyz"

        # 轨迹保存目录：{method_dir}/trajectories/
        traj_dir = os.path.join(self.method_dir, 'trajectories')
        os.makedirs(traj_dir, exist_ok=True)
        filepath = os.path.join(traj_dir, filename)

        n_atoms = len(atom_symbols)
        is_hybrid = method_name == 'hybrid'

        with open(filepath, 'w') as f:
            for it in history.iterations:
                # XYZ 格式：原子数
                f.write(f"{n_atoms}\n")
                
                # 注释行：根据方法类型生成不同格式
                if is_hybrid:
                    # 混合策略：区分轮次和内外层
                    # 第 1 轮的外层迭代也标记为"轮次 1"，不再使用"初始采样"标签
                    round_label = f"轮次{it.round_num}"

                    if it.stage == 'outer':
                        # 外层：真实能量和梯度
                        comment = f"{round_label}  Outer Iteration {it.iteration}, Energy={it.energy:.10f}, |grad|={it.gradient_norm:.6f}"
                    elif it.stage == 'inner':
                        # 内层：使用预测梯度
                        pred_norm = it.gradient_pred_norm if it.gradient_pred_norm is not None else 0.0
                        comment = f"{round_label}  Inner Iteration {it.iteration}, |pred_grad|={pred_norm:.6f}"
                    else:
                        # 其他情况（pyberny）
                        comment = f"Outer Iteration {it.iteration}, Energy={it.energy:.10f}, |grad|={it.gradient_norm:.6f}"
                else:
                    # 纯 PyBerny：简单格式
                    comment = f"Iteration {it.iteration}, Energy={it.energy:.10f}, |grad|={it.gradient_norm:.6f}"
                
                f.write(f"{comment}\n")

                # 原子坐标和梯度
                coords = it.coords.reshape(n_atoms, 3)
                
                if is_hybrid and it.stage == 'inner' and it.gradient_pred is not None:
                    # 内层：显示预测梯度
                    grad_pred = it.gradient_pred.reshape(n_atoms, 3)
                    for i, sym in enumerate(atom_symbols):
                        f.write(f"{sym:2s} {coords[i,0]:12.6f} {coords[i,1]:12.6f} {coords[i,2]:12.6f}  "
                               f"{grad_pred[i,0]:12.6f} {grad_pred[i,1]:12.6f} {grad_pred[i,2]:12.6f}\n")
                else:
                    # 外层/纯 PyBerny：显示真实梯度
                    gradient = it.gradient.reshape(n_atoms, 3)
                    for i, sym in enumerate(atom_symbols):
                        f.write(f"{sym:2s} {coords[i,0]:12.6f} {coords[i,1]:12.6f} {coords[i,2]:12.6f}  "
                               f"{gradient[i,0]:12.6f} {gradient[i,1]:12.6f} {gradient[i,2]:12.6f}\n")

        print(f"优化轨迹已保存：{filepath}")
        return filepath
    
    def save_structure(self, molecule: Molecule, filename: str,
                       prefix: str = "") -> str:
        """
        保存分子结构

        Args:
            molecule: 分子对象
            filename: 文件名
            prefix: 文件名前缀

        Returns:
            filepath: 保存的文件路径
        """
        filepath = os.path.join(self.method_dir, 'structures',
                               f"{prefix}{filename}.xyz")
        molecule.save_xyz(filepath)
        print(f"分子结构已保存：{filepath}")
        return filepath
    
    def save_initial_structure(self, molecule: Molecule, method_name: str,
                              ai_method: str = None) -> str:
        """
        保存初始结构
        
        Args:
            molecule: 分子对象
            method_name: 方法名称
            ai_method: AI 方法类型（可选）
        
        Returns:
            filepath: 保存的文件路径
        """
        # 如果有 ai_method，添加到文件名中
        if ai_method:
            ai_suffix = ai_method.replace('gradient_', '').replace('_', '')
            prefix = f"{method_name}_{ai_suffix}_"
        else:
            prefix = f"{method_name}_"
        
        return self.save_structure(molecule, "initial", prefix)
    
    def save_final_structure(self, molecule: Molecule, method_name: str,
                            ai_method: str = None) -> str:
        """
        保存最终/最优结构
        
        Args:
            molecule: 分子对象
            method_name: 方法名称
            ai_method: AI 方法类型（可选）
        
        Returns:
            filepath: 保存的文件路径
        """
        # 如果有 ai_method，添加到文件名中
        if ai_method:
            # 从 ai_method 提取后缀，如 'gradient_predicting' -> 'predicting'
            ai_suffix = ai_method.replace('gradient_', '').replace('_', '')
            prefix = f"{method_name}_{ai_suffix}_"
        else:
            prefix = f"{method_name}_"
        
        return self.save_structure(molecule, "final", prefix)
    
    def save_iteration_details(self, history: OptimizationHistory,
                               method_name: str,
                               atom_symbols: List[str]) -> str:
        """
        保存详细的迭代信息（包括梯度矩阵、位移等）

        Args:
            history: 优化历史
            method_name: 方法名称
            atom_symbols: 原子符号列表

        Returns:
            filepath: 保存的文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 构建文件名：如果是 hybrid 方法且指定了 AI 方法，则添加后缀
        if method_name == 'hybrid' and self.ai_method:
            ai_suffix = self._get_ai_method_suffix()
            filename = f"{method_name}_{ai_suffix}_details_{timestamp}.json"
        else:
            filename = f"{method_name}_details_{timestamp}.json"
        
        filepath = os.path.join(self.method_dir, filename)

        n_atoms = len(atom_symbols)
        details = {
            'method': method_name,
            'ai_method': self.ai_method,  # 添加 AI 方法类型到文件内容
            'timestamp': timestamp,
            'n_atoms': n_atoms,
            'atom_symbols': atom_symbols,
            'iterations': []
        }
        
        for it in history.iterations:
            coords = it.coords.reshape(n_atoms, 3)
            gradient = it.gradient.reshape(n_atoms, 3)
            displacement = it.displacement.reshape(n_atoms, 3) if it.displacement is not None else None
            
            iter_data = {
                'iteration': it.iteration,
                'energy': it.energy,
                'gradient_norm': it.gradient_norm,
                'coords': coords.tolist(),
                'gradient': gradient.tolist(),
                'displacement': displacement.tolist() if displacement is not None else None,
                'timestamp': it.timestamp
            }
            details['iterations'].append(iter_data)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(details, f, indent=2)
        
        print(f"详细迭代信息已保存：{filepath}")
        return filepath
    
    def save_summary(self, histories: Dict[str, OptimizationHistory],
                     metadata: Dict[str, Any] = None) -> str:
        """
        保存对比总结
        
        Args:
            histories: 优化历史字典 {method_name: history}
            metadata: 元数据
        
        Returns:
            filepath: 保存的文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.save_dir, f"comparison_summary_{timestamp}.json")
        
        summary = {
            'timestamp': timestamp,
            'methods': {}
        }
        
        for method_name, history in histories.items():
            summary['methods'][method_name] = {
                'statistics': self._compute_statistics(history),
                'best_energy': float(history.get_energies().min()) if len(history) > 0 else None,
                'best_gradient_norm': float(history.get_gradient_norms().min()) if len(history) > 0 else None,
                'iterations': len(history)
            }
        
        if metadata:
            summary['metadata'] = metadata
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        print(f"对比总结已保存：{filepath}")
        return filepath
    
    def save_log(self, message: str, method_name: str) -> str:
        """
        保存日志消息
        
        Args:
            message: 日志内容
            method_name: 方法名称
        
        Returns:
            filepath: 保存的文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.save_dir, f"{method_name}_log_{timestamp}.txt")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Method: {method_name}\n")
            f.write("=" * 60 + "\n\n")
            f.write(message)
        
        return filepath


def create_output_manager(config: Dict[str, Any],
                          ai_method: str = None,
                          method_name: str = None) -> OutputManager:
    """
    便捷函数：创建输出管理器

    Args:
        config: 配置字典
        ai_method: AI 方法类型（'simple'/'gradient'/'random_forest'等）
        method_name: 优化方法名称（'pyberny'/'hybrid'）

    Returns:
        OutputManager
    """
    output_config = config.get('output', {})
    save_dir = output_config.get('save_dir', './output')
    format = output_config.get('format', 'json')

    return OutputManager(save_dir, format, ai_method=ai_method, method_name=method_name)
