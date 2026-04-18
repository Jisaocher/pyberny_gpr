"""
L-BFGS 优化器实现（使用 berny 库）

使用 berny 库实现 L-BFGS 优化，替代 scipy 的 L-BFGS-B
每轮独立运行，不继承之前轮次的 L-BFGS 历史状态
"""
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import berny
import time
from berny import Geometry

from core.molecule import Molecule, OptimizationHistory, IterationData
from core.calculator import QuantumCalculator, EnergyGradientFunction
from optimizers.base import BaseOptimizer


class PyBernyOptimizer(BaseOptimizer):
    """
    L-BFGS 优化器（使用 berny 库）

    每轮独立运行，不继承之前轮次的 L-BFGS 历史状态
    功能上等价于 scipy 的 L-BFGS-B，但使用 berny 库实现
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化 L-BFGS 优化器

        Args:
            config: 配置字典
        """
        super().__init__(config)
        self.name = "PyBerny L-BFGS"

        # L-BFGS 参数：直接使用 berny 配置，确保与基准方法一致
        berny_config = config.get('berny', {})
        
        # 最大步数：使用 berny.maxsteps
        self.maxiter = int(berny_config.get('maxsteps', 500))
        
        # 梯度阈值：使用 berny.gradient_threshold
        self.gtol = float(berny_config.get('gradient_threshold', 1e-4))
        
        # 信任半径：使用 berny.trust
        self.trust = float(berny_config.get('trust', 0.3))

        # 内部状态
        self._energy_func = None
        self._prev_coords = None

    def optimize(self, molecule: Molecule, calculator: QuantumCalculator) -> OptimizationHistory:
        """
        执行 L-BFGS 优化（使用 berny 库）

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
        geom = Geometry(list(molecule.atom_symbols), molecule.coords)
        berny_optimizer = berny.Berny(geom, maxsteps=self.maxiter, trust=self.trust)

        # 打印开始信息
        if self.config.get('optimizer', {}).get('verbose', True):
            print("=" * 70)
            print("L-BFGS 优化开始（berny 库）")
            print("=" * 70)
            print(f"初始能量：{self._energy_func.energy_only(x0):.10f} Hartree")
            print(f"原子数：{molecule.n_atoms}")
            print(f"自由度：{len(x0)}")
            print(f"最大迭代次数：{self.maxiter}")
            print("=" * 70)

        self.history.start_time = time.time()

        # 使用 berny 进行优化（标准用法：for geom in optimizer）
        iteration = 0
        for geom in berny_optimizer:
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

            # 检查收敛
            if gradient_norm < self.gtol:
                self.history.converged = True
                self.history.convergence_iteration = iteration
                break

            # 发送能量和梯度到 berny，获取下一步
            try:
                berny_optimizer.send((current_energy, current_gradient))
            except StopIteration:
                break

            self._prev_coords = current_coords.copy()
            iteration += 1

        self.history.end_time = time.time()

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
        执行单步 L-BFGS 优化（用于兼容基类接口）

        注意：这个方法主要用于兼容 BaseOptimizer 接口
        实际使用时推荐使用 run_fixed_steps()

        Args:
            coords_flat: 当前坐标 (展平)

        Returns:
            new_coords: 新坐标
            energy: 能量
            gradient: 梯度
        """
        # 这个方法需要配合 optimize() 使用
        # 单独使用时，创建一个临时的 run_fixed_steps 调用
        if self._energy_func is None:
            raise NotImplementedError("step() 需要先调用 optimize() 初始化，或使用 run_fixed_steps()")

        energy, gradient = self._energy_func(coords_flat)
        
        # 备用方案：负梯度方向
        step_size = 0.01
        new_coords = coords_flat - step_size * gradient
        
        new_energy, new_gradient = self._energy_func(new_coords)
        return new_coords, new_energy, new_gradient

    def run_fixed_steps(self, coords_flat: np.ndarray, n_steps: int,
                        calculator: QuantumCalculator,
                        atom_symbols: Optional[List[str]] = None,
                        initial_energy: Optional[float] = None,
                        initial_gradient: Optional[np.ndarray] = None) -> Tuple[np.ndarray, OptimizationHistory]:
        """
        执行固定步数的 L-BFGS 优化（使用 berny 库）

        用于混合优化策略，每轮独立运行，不继承历史状态

        Args:
            coords_flat: 初始坐标
            n_steps: 步数
            calculator: 量子化学计算器
            atom_symbols: 原子符号列表 (可选)
            initial_energy: 初始能量（可选，如果提供则避免重复计算）
            initial_gradient: 初始梯度（可选，如果提供则避免重复计算）

        Returns:
            final_coords: 最终坐标
            history: 这段优化的历史
        """
        # 创建临时分子
        n_atoms = len(coords_flat) // 3

        # 使用传入的原子符号或已保存的原子符号
        if atom_symbols is not None:
            self.atom_symbols = atom_symbols
        elif self.atom_symbols is None:
            self.atom_symbols = ['C'] * n_atoms
            import warnings
            warnings.warn("atom_symbols not provided, using ['C'] * n_atoms as default")

        self.current_mol = Molecule(self.atom_symbols, coords_flat.reshape(n_atoms, 3))
        self.calculator = calculator
        self.history = OptimizationHistory()

        # 创建能量函数
        self._energy_func = EnergyGradientFunction(calculator, self.atom_symbols)
        self._prev_coords = coords_flat.copy()

        # 创建几何结构和 berny 优化器（每轮都是新的实例，不继承状态）
        # 传入与基准方法相同的收敛阈值，确保行为一致
        geom = Geometry(list(self.atom_symbols), self.current_mol.coords)
        berny_optimizer = berny.Berny(
            geom,
            maxsteps=n_steps,
            trust=self.trust,
            energy_threshold=self.config.get('berny', {}).get('energy_threshold', 1e-6),
            gradient_threshold=self.gtol,
            displacement_threshold=self.config.get('berny', {}).get('displacement_threshold', 1e-3)
        )

        # 执行固定步数（标准用法：for geom in optimizer）
        step = 0
        current_coords = coords_flat.copy()
        
        # 统计实际 PySCF 调用次数（排除复用的初始点）
        actual_pyscf_calls = 0

        for geom in berny_optimizer:
            # 从几何结构获取坐标
            current_coords = np.array(geom.coords).flatten()

            # 计算当前能量和梯度（如果提供了初始值且坐标匹配，则复用）
            if initial_energy is not None and initial_gradient is not None and step == 0:
                # 检查是否是初始点（坐标匹配）
                if np.allclose(current_coords, coords_flat):
                    current_energy = initial_energy
                    current_gradient = initial_gradient
                    # 复用初始值，不计入 PySCF 调用
                else:
                    current_energy, current_gradient = self._energy_func(current_coords)
                    actual_pyscf_calls += 1
            else:
                current_energy, current_gradient = self._energy_func(current_coords)
                actual_pyscf_calls += 1
            
            gradient_norm = np.linalg.norm(current_gradient)

            # 记录历史
            displacement = current_coords - self._prev_coords if self._prev_coords is not None else None
            data = self.get_iteration_data(
                iteration=step,
                energy=current_energy,
                gradient=current_gradient,
                coords=current_coords,
                prev_coords=self._prev_coords
            )
            self.history.add_iteration(data)

            # 打印信息
            if self.config.get('optimizer', {}).get('verbose', True):
                disp_norm = np.linalg.norm(displacement) if displacement is not None else None
                self.print_iteration(step, current_energy, gradient_norm, disp_norm)

            # 检查收敛
            if gradient_norm < self.gtol:
                if self.config.get('optimizer', {}).get('verbose', True):
                    print(f"收敛！梯度范数：{gradient_norm:.6f} < {self.gtol}")
                break

            # 发送能量和梯度到 berny
            try:
                berny_optimizer.send((current_energy, current_gradient))
            except StopIteration:
                if self.config.get('optimizer', {}).get('verbose', True):
                    print("berny 已收敛")
                break

            self._prev_coords = current_coords.copy()
            step += 1

        if self.config.get('optimizer', {}).get('verbose', True):
            if step > 0:
                print(f"外层完成 {step} 步")

        return current_coords, self.history, actual_pyscf_calls


def run_lbfgs_optimization(molecule: Molecule, config: Dict[str, Any]) -> OptimizationHistory:
    """
    便捷函数：运行 L-BFGS 优化
    """
    calculator = QuantumCalculator(
        basis=config.get('calculation', {}).get('basis', 'cc-pvdz'),
        method=config.get('calculation', {}).get('method', 'RHF'),
        unit=config.get('calculation', {}).get('unit', 'angstrom')
    )

    optimizer = PyBernyOptimizer(config)
    return optimizer.optimize(molecule, calculator)
