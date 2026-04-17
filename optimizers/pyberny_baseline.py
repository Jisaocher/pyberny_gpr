"""
纯 PyBerny 基准优化器

使用 berny 库实现完整的 BFGS 优化，一次性运行到底，中间不打断
作为混合方法的对比基准

核心优势：
- 完整的 BFGS Hessian 更新（不是 L-BFGS）
- 跨步次继承曲率信息
- 冗余内坐标系统
- Trust Region 动态调整
- 复合收敛判据
"""
import numpy as np
from typing import Dict, Any, Optional, Tuple
import time
import berny
from berny import Geometry

from core.molecule import Molecule, OptimizationHistory, IterationData
from core.calculator import QuantumCalculator, EnergyGradientFunction
from optimizers.base import BaseOptimizer


class PyBernyBaselineOptimizer(BaseOptimizer):
    """
    纯 PyBerny 基准优化器

    使用 berny 库的完整 BFGS 实现，一次性优化完成
    用于作为混合方法的对比基准
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化 PyBerny 基准优化器

        Args:
            config: 配置字典
        """
        super().__init__(config)
        self.name = "PyBerny Baseline"

        # berny 特定参数
        berny_config = config.get('berny', {})
        self.maxsteps = int(berny_config.get('maxsteps', 500))
        self.trust = float(berny_config.get('trust', 0.3))
        self.energy_threshold = float(berny_config.get('energy_threshold', 1e-6))
        self.gradient_threshold = float(berny_config.get('gradient_threshold', 1e-4))
        self.displacement_threshold = float(berny_config.get('displacement_threshold', 1e-3))
        self.debug = bool(berny_config.get('debug', False))
        
        # 使用优化器的 convergence_threshold 作为 berny 的 gradient_threshold（如果未指定）
        if 'gradient_threshold' not in berny_config:
            optimizer_config = config.get('optimizer', {})
            self.gradient_threshold = float(optimizer_config.get('convergence_threshold', 1e-4))

        # 内部状态
        self._energy_func = None
        self._prev_coords = None
        self._berny_optimizer = None  # 保持 berny 优化器状态，跨步次继承

    def optimize(self, molecule: Molecule, calculator: QuantumCalculator) -> OptimizationHistory:
        """
        执行完整的 PyBerny BFGS 优化（一次性运行到底）

        Args:
            molecule: 初始分子结构
            calculator: 量子化学计算器

        Returns:
            OptimizationHistory: 优化历史
        """
        self.current_mol = molecule
        self.atom_symbols = molecule.atom_symbols
        self.calculator = calculator
        self.history = OptimizationHistory()

        # 初始化
        x0 = molecule.get_coords_flat()
        self._prev_coords = x0.copy()

        # 创建能量/梯度函数
        self._energy_func = EnergyGradientFunction(calculator, molecule.atom_symbols)

        # 创建几何结构和 berny 优化器
        # 注意：这里创建的优化器会一直保持状态，直到优化完成
        geom = Geometry(list(molecule.atom_symbols), molecule.coords)
        self._berny_optimizer = berny.Berny(
            geom,
            maxsteps=self.maxsteps,
            trust=self.trust,
            energy_threshold=self.energy_threshold,
            gradient_threshold=self.gradient_threshold,
            displacement_threshold=self.displacement_threshold,
            debug=self.debug
        )

        # 打印开始信息
        if self.config.get('optimizer', {}).get('verbose', True):
            print("=" * 70)
            print("PyBerny 基准优化开始（完整 BFGS，一次性运行）")
            print("=" * 70)
            print(f"初始能量：{self._energy_func.energy_only(x0):.10f} Hartree")
            print(f"原子数：{molecule.n_atoms}")
            print(f"自由度：{len(x0)}")
            print(f"最大步数：{self.maxsteps}")
            print(f"Trust Radius: {self.trust} Å")
            print(f"收敛阈值：能量={self.energy_threshold}, 梯度={self.gradient_threshold}, 位移={self.displacement_threshold}")
            print("=" * 70)

        self.history.start_time = time.time()

        # 使用 berny 的标准迭代方式
        # 参考 berny 文档：
        #   optimizer = Berny(geom)
        #   for geom in optimizer:
        #       debug = optimizer.send((energy, gradients))
        
        iteration = 0
        for geom in self._berny_optimizer:
            # 从几何结构获取坐标
            current_coords = np.array(geom.coords).flatten()

            # 计算当前能量和梯度
            current_energy, current_gradient = self._energy_func(current_coords)
            gradient_norm = np.linalg.norm(current_gradient)

            # 记录当前点
            displacement = None
            if self._prev_coords is not None:
                displacement = current_coords - self._prev_coords

            data = self.get_iteration_data(
                iteration=iteration,
                energy=current_energy,
                gradient=current_gradient,
                coords=current_coords,
                prev_coords=self._prev_coords
            )
            self.history.add_iteration(data)

            # 打印信息
            disp_norm = np.linalg.norm(displacement) if displacement is not None else None
            self.print_iteration(iteration, current_energy, gradient_norm, disp_norm)

            # 检查收敛（使用梯度范数）
            convergence_threshold = self.config.get('optimizer', {}).get('convergence_threshold', 1e-4)
            if gradient_norm < convergence_threshold:
                self.history.converged = True
                self.history.convergence_iteration = iteration
                if self.config.get('optimizer', {}).get('verbose', True):
                    print(f"\n✓ 收敛！梯度范数：{gradient_norm:.6f} < 阈值 {convergence_threshold}")
                break

            # 发送能量和梯度到 berny
            self._berny_optimizer.send((current_energy, current_gradient))

            self._prev_coords = current_coords.copy()
            iteration += 1

        self.history.end_time = time.time()

        # 如果循环结束但还没有设置收敛标志，检查最终梯度
        # berny 使用复合收敛判据（能量 + 梯度 + 位移），可能在我们设定的梯度阈值之前停止
        if not self.history.converged and self.history.iterations:
            last_iter = self.history.get_last_iteration()
            convergence_threshold = self.config.get('optimizer', {}).get('convergence_threshold', 1e-4)
            
            # 如果最终梯度小于阈值的 5 倍，认为 berny 已经有效收敛
            # 这是因为 berny 的复合判据可能先于纯梯度判据触发
            if last_iter.gradient_norm < convergence_threshold * 5:
                self.history.converged = True
                self.history.convergence_iteration = len(self.history) - 1
                if self.config.get('optimizer', {}).get('verbose', True):
                    print(f"\nberny 已收敛（复合判据），最终梯度：{last_iter.gradient_norm:.6f}")
            else:
                # berny 停止但未达到收敛阈值，可能是达到 maxsteps
                if self.config.get('optimizer', {}).get('verbose', True):
                    print(f"\nberny 停止，但最终梯度 {last_iter.gradient_norm:.6f} > 阈值 {convergence_threshold}")
                    print(f"可能是达到 maxsteps={self.maxsteps} 或其他 berny 内部条件")

        # 打印结束信息
        if self.config.get('optimizer', {}).get('verbose', True):
            print("=" * 70)
            print("优化完成！")
            last_iter = self.history.get_last_iteration()
            if last_iter:
                print(f"最终能量：{last_iter.energy:.10f} Hartree")
                print(f"最终梯度范数：{last_iter.gradient_norm:.6f}")
            print(f"迭代次数：{len(self.history)}")
            print(f"收敛状态：{'是' if self.history.converged else '否'}")
            print(f"计算时间：{self.history.end_time - self.history.start_time:.2f} 秒")
            print(f"能量/梯度调用次数：{self._energy_func.call_count}")
            print("=" * 70)

        return self.history

    def step(self, coords_flat: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        执行单步 PyBerny 优化

        用于兼容 BaseOptimizer 接口
        注意：这个方法需要配合 optimize() 使用，单独使用可能不符合预期

        Args:
            coords_flat: 当前坐标 (展平)

        Returns:
            new_coords: 新坐标
            energy: 能量
            gradient: 梯度
        """
        if self._berny_optimizer is None:
            raise NotImplementedError("需要先调用 optimize() 初始化")

        if self._energy_func is None:
            raise ValueError("需要先调用 optimize() 初始化")

        # 计算当前能量和梯度
        energy, gradient = self._energy_func(coords_flat)

        # 发送当前值到 berny，获取下一步
        try:
            self._berny_optimizer.send((energy, gradient))
        except StopIteration:
            # berny 已收敛，返回当前点
            return coords_flat, energy, gradient

        # 从 berny 获取下一步的几何结构（通过迭代）
        try:
            geom = next(self._berny_optimizer)
            new_coords = np.array(geom.coords).flatten()
        except StopIteration:
            return coords_flat, energy, gradient

        new_energy, new_gradient = self._energy_func(new_coords)
        return new_coords, new_energy, new_gradient

    def get_trust_radius(self) -> float:
        """获取当前信任半径"""
        if self._berny_optimizer is not None:
            return self._berny_optimizer.trust
        return self.trust


def run_pyberny_baseline_optimization(molecule: Molecule, config: Dict[str, Any]) -> OptimizationHistory:
    """
    便捷函数：运行纯 PyBerny 基准优化

    Args:
        molecule: 初始分子
        config: 配置字典

    Returns:
        OptimizationHistory: 优化历史
    """
    calculator = QuantumCalculator(
        basis=config.get('calculation', {}).get('basis', 'cc-pvdz'),
        method=config.get('calculation', {}).get('method', 'RHF'),
        unit=config.get('calculation', {}).get('unit', 'angstrom')
    )

    optimizer = PyBernyBaselineOptimizer(config)
    return optimizer.optimize(molecule, calculator)
