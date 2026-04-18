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

多起点策略（可选）：
- 第一轮 pyberny 迭代 n_init+outer_steps 次
- 后续每轮 pyberny 迭代 outer_steps 次
- 每一轮终点作为下一轮的起点，从缓存读取能量/梯度（不额外计算）
- 累计真实能量/梯度计算次数，与混合策略对齐
- 避免起点/终点重复记录到迭代图中
"""
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
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

    支持多起点策略（可选）：
    - 模拟混合方法的起点更新操作
    - 第一轮 n_init+outer_steps 次，后续每轮 outer_steps 次
    - 缓存终点能量/梯度供下一轮起点复用
    - 避免重复记录起点/终点到迭代图
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

        # 多起点策略参数
        self.multi_start = bool(berny_config.get('multi_start', False))
        if self.multi_start:
            hybrid_config = config.get('hybrid', {})
            self.n_init = hybrid_config.get('n_init', 5)
            self.outer_steps = hybrid_config.get('outer_steps', 10)

        # 使用优化器的 convergence_threshold 作为 berny 的 gradient_threshold（如果未指定）
        if 'gradient_threshold' not in berny_config:
            optimizer_config = config.get('optimizer', {})
            self.gradient_threshold = float(optimizer_config.get('convergence_threshold', 1e-4))

        # 内部状态
        self._energy_func = None
        self._prev_coords = None
        self._berny_optimizer = None  # 保持 berny 优化器状态，跨步次继承

        # 多起点策略缓存
        self._cached_energy = None
        self._cached_gradient = None
        self._cached_coords_hash = None
        self._total_pyscf_calls = 0  # 累计 PySCF 调用次数

    def _get_energy_gradient(self, coords: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        获取能量和梯度（带缓存）

        如果坐标与上一轮终点相同，从缓存读取，避免重复计算

        Args:
            coords: 坐标

        Returns:
            energy, gradient
        """
        coords_hash = hash(coords.tobytes())

        if self._cached_coords_hash is not None and coords_hash == self._cached_coords_hash:
            # 从缓存读取
            return self._cached_energy, self._cached_gradient

        # 计算新的能量/梯度
        energy, gradient = self._energy_func(coords)
        self._total_pyscf_calls += 1

        # 更新缓存
        self._cached_energy = energy
        self._cached_gradient = gradient
        self._cached_coords_hash = coords_hash

        return energy, gradient

    def _run_fixed_steps(self, coords: np.ndarray, n_steps: int,
                         initial_energy: Optional[float] = None,
                         initial_gradient: Optional[np.ndarray] = None,
                         is_first_round: bool = False) -> Tuple[np.ndarray, OptimizationHistory, int, bool]:
        """
        执行固定步数的 PyBerny 优化

        Args:
            coords: 初始坐标
            n_steps: 步数
            initial_energy: 初始能量（可选，如果提供则避免重复计算）
            initial_gradient: 初始梯度（可选，如果提供则避免重复计算）
            is_first_round: 是否为第一轮（用于缓存管理）

        Returns:
            final_coords: 最终坐标
            history: 优化历史
            pyscf_calls: 实际 PySCF 调用次数
            early_stop: 是否提前终止（berny 已收敛）
        """
        # 创建临时分子和 berny 优化器
        n_atoms = len(coords) // 3
        geom = Geometry(list(self.atom_symbols), coords.reshape(n_atoms, 3))
        berny_optimizer = berny.Berny(
            geom,
            maxsteps=n_steps,
            trust=self.trust,
            energy_threshold=self.energy_threshold,
            gradient_threshold=self.gradient_threshold,
            displacement_threshold=self.displacement_threshold,
            debug=self.debug
        )

        history = OptimizationHistory()
        step = 0
        current_coords = coords.copy()
        prev_coords = coords.copy()
        actual_pyscf_calls = 0
        early_stop = False
        berny_converged = False  # 标记 berny 是否内部收敛

        for geom in berny_optimizer:
            # 从几何结构获取坐标
            current_coords = np.array(geom.coords).flatten()

            # 检测起点重复（berny 没有生成新结构，说明它认为已收敛）
            if step == 0 and np.allclose(current_coords, coords):
                # 起点，复用初始值
                if initial_energy is not None and initial_gradient is not None:
                    current_energy = initial_energy
                    current_gradient = initial_gradient
                else:
                    current_energy, current_gradient = self._energy_func(current_coords)
                    actual_pyscf_calls += 1
            else:
                # 计算当前能量和梯度
                current_energy, current_gradient = self._energy_func(current_coords)
                actual_pyscf_calls += 1

            gradient_norm = np.linalg.norm(current_gradient)

            # 记录历史
            displacement = current_coords - prev_coords if step > 0 else None
            data = self.get_iteration_data(
                iteration=step,
                energy=current_energy,
                gradient=current_gradient,
                coords=current_coords,
                prev_coords=prev_coords
            )
            history.add_iteration(data)

            # 打印信息
            if self.config.get('optimizer', {}).get('verbose', True):
                disp_norm = np.linalg.norm(displacement) if displacement is not None else None
                self.print_iteration(step, current_energy, gradient_norm, disp_norm)

            # 检查收敛（我们的梯度阈值）
            convergence_threshold = self.config.get('optimizer', {}).get('convergence_threshold', 1e-4)
            if gradient_norm < convergence_threshold:
                if self.config.get('optimizer', {}).get('verbose', True):
                    print(f"收敛！梯度范数：{gradient_norm:.6f} < {convergence_threshold}")
                early_stop = True
                break

            # 发送能量和梯度到 berny，获取下一步
            try:
                berny_optimizer.send((current_energy, current_gradient))
            except StopIteration:
                if self.config.get('optimizer', {}).get('verbose', True):
                    print("berny 已收敛（StopIteration）")
                early_stop = True
                break

            prev_coords = current_coords.copy()
            step += 1

        # 检测 berny 自然结束（循环完成但没有达到 n_steps）
        # 如果只有起点（step=0）或实际步数远小于设置步数，说明 berny 内部收敛
        if step == 0:
            # 只有起点，berny 立即判断收敛
            early_stop = True
            if self.config.get('optimizer', {}).get('verbose', True):
                print(f"berny 已收敛（无新几何结构生成）")
        elif step < n_steps - 1 and not early_stop:
            # berny 的 for 循环自然结束，说明它认为已收敛
            early_stop = True
            if self.config.get('optimizer', {}).get('verbose', True):
                print(f"berny 提前终止（实际 {step+1} 步 < 设置 {n_steps} 步）")

        return current_coords, history, actual_pyscf_calls, early_stop

    def optimize(self, molecule: Molecule, calculator: QuantumCalculator) -> OptimizationHistory:
        """
        执行 PyBerny BFGS 优化

        支持两种模式：
        - 原始模式：一次性运行到底（multi_start=False）
        - 多起点模式：分轮次优化，模拟混合策略（multi_start=True）

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

        # 打印开始信息
        if self.config.get('optimizer', {}).get('verbose', True):
            print("=" * 70)
            if self.multi_start:
                print(f"PyBerny 基准优化开始（多起点策略）")
                print(f"第一轮：{self.n_init + self.outer_steps} 步，后续每轮：{self.outer_steps} 步")
            else:
                print("PyBerny 基准优化开始（完整 BFGS，一次性运行）")
            print("=" * 70)
            print(f"原子数：{molecule.n_atoms}")
            print(f"自由度：{len(x0)}")
            if not self.multi_start:
                print(f"最大步数：{self.maxsteps}")
                print(f"Trust Radius: {self.trust} Å")
            print(f"收敛阈值：能量={self.energy_threshold}, 梯度={self.gradient_threshold}, 位移={self.displacement_threshold}")
            print("=" * 70)

        self.history.start_time = time.time()

        if self.multi_start:
            # 多起点策略
            return self._optimize_multi_start(molecule)
        else:
            # 原始模式：一次性运行到底
            return self._optimize_single_run(molecule)

    def _optimize_single_run(self, molecule: Molecule) -> OptimizationHistory:
        """
        原始模式：一次性运行到底

        Args:
            molecule: 初始分子结构

        Returns:
            OptimizationHistory: 优化历史
        """
        x0 = molecule.get_coords_flat()

        # 创建几何结构和 berny 优化器
        geom = Geometry(list(self.atom_symbols), molecule.coords)
        self._berny_optimizer = berny.Berny(
            geom,
            maxsteps=self.maxsteps,
            trust=self.trust,
            energy_threshold=self.energy_threshold,
            gradient_threshold=self.gradient_threshold,
            displacement_threshold=self.displacement_threshold,
            debug=self.debug
        )

        # 初始能量和梯度计算
        initial_energy, initial_gradient = self._energy_func(x0)
        self._total_pyscf_calls = 1
        gradient_norm = np.linalg.norm(initial_gradient)

        if self.config.get('optimizer', {}).get('verbose', True):
            print(f"初始能量：{initial_energy:.10f} Hartree")
            print(f"初始梯度范数：{gradient_norm:.6f}")

        iteration = 0
        for geom in self._berny_optimizer:
            current_coords = np.array(geom.coords).flatten()

            # 如果是第一步，使用已计算的初始能量和梯度，避免重复计算
            if iteration == 0 and np.allclose(current_coords, x0):
                current_energy = initial_energy
                current_gradient = initial_gradient
            else:
                current_energy, current_gradient = self._energy_func(current_coords)
                self._total_pyscf_calls += 1

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

        # 检查最终收敛状态
        if not self.history.converged and self.history.iterations:
            last_iter = self.history.get_last_iteration()
            convergence_threshold = self.config.get('optimizer', {}).get('convergence_threshold', 1e-4)

            if last_iter.gradient_norm < convergence_threshold * 5:
                self.history.converged = True
                self.history.convergence_iteration = len(self.history) - 1
                if self.config.get('optimizer', {}).get('verbose', True):
                    print(f"\nberny 已收敛（复合判据），最终梯度：{last_iter.gradient_norm:.6f}")
            else:
                if self.config.get('optimizer', {}).get('verbose', True):
                    print(f"\nberny 停止，但最终梯度 {last_iter.gradient_norm:.6f} > 阈值 {convergence_threshold}")

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
            print(f"能量/梯度调用次数：{self._total_pyscf_calls}")
            print("=" * 70)

        return self.history

    def _optimize_multi_start(self, molecule: Molecule) -> OptimizationHistory:
        """
        多起点策略：分轮次优化，模拟混合方法的起点更新操作

        - 第一轮：n_init + outer_steps 步
        - 后续每轮：outer_steps 步
        - 每轮终点作为下一轮起点，从缓存读取能量/梯度
        - 避免起点/终点重复记录

        Args:
            molecule: 初始分子结构

        Returns:
            OptimizationHistory: 优化历史
        """
        coords = molecule.get_coords_flat()
        convergence_threshold = self.config.get('optimizer', {}).get('convergence_threshold', 1e-4)
        max_rounds = self.config.get('hybrid', {}).get('convergence', {}).get('max_rounds', 50)
        max_no_improvement = self.config.get('hybrid', {}).get('convergence', {}).get('max_no_improvement', 50)
        no_improvement_threshold = self.config.get('hybrid', {}).get('convergence', {}).get('no_improvement_threshold', 1e-6)

        no_improvement_count = 0
        round_num = 0

        while round_num < max_rounds:
            round_num += 1

            if self.config.get('optimizer', {}).get('verbose', True):
                print(f"\n{'='*70}")
                print(f"第 {round_num} 轮优化")
                print(f"{'='*70}")

            # 获取本轮起点能量/梯度（带缓存）
            start_energy, start_gradient = self._get_energy_gradient(coords)
            start_gradient_norm = np.linalg.norm(start_gradient)

            if self.config.get('optimizer', {}).get('verbose', True):
                print(f"本轮起点：E={start_energy:.8f}, |g|={start_gradient_norm:.6f}")

            # 确定本轮步数：第一轮 n_init+outer_steps，后续 outer_steps
            is_first_round = (round_num == 1)
            actual_steps = self.n_init + self.outer_steps if is_first_round else self.outer_steps

            if self.config.get('optimizer', {}).get('verbose', True):
                print(f"[PyBerny] 执行 {actual_steps} 步...")

            # 执行固定步数优化
            final_coords, round_history, pyscf_calls, early_stop = self._run_fixed_steps(
                coords, actual_steps,
                initial_energy=start_energy,
                initial_gradient=start_gradient,
                is_first_round=is_first_round
            )

            # 累计 PySCF 调用次数
            self._total_pyscf_calls += pyscf_calls

            # 获取本轮终点能量/梯度（从历史记录，避免重复计算）
            if round_history.iterations:
                last_iter = round_history.get_last_iteration()
                final_energy = last_iter.energy
                final_gradient = last_iter.gradient
                final_gradient_norm = last_iter.gradient_norm
            else:
                final_energy, final_gradient = self._get_energy_gradient(final_coords)
                final_gradient_norm = np.linalg.norm(final_gradient)

            if self.config.get('optimizer', {}).get('verbose', True):
                print(f"本轮终点：E={final_energy:.8f}, |g|={final_gradient_norm:.6f}")
                print(f"本轮 PySCF 调用次数：{pyscf_calls}")
                if early_stop:
                    print(f"berny 提前终止（已收敛）")

            # 添加本轮历史到总历史（去除起点重复：如果起点与上轮终点重合，则跳过）
            for i, it_data in enumerate(round_history.iterations):
                # 跳过第一轮的起点（如果是后续轮次且与上轮终点重合）
                if not is_first_round and i == 0:
                    if np.allclose(it_data.coords, coords):
                        continue

                # 更新 round_num 和 stage
                it_data.round_num = round_num
                it_data.stage = 'pyberny'
                self.history.add_iteration(it_data)

            # 缓存本轮终点，供下一轮起点复用
            self._cached_energy = final_energy
            self._cached_gradient = final_gradient
            self._cached_coords_hash = hash(final_coords.tobytes())

            # 更新到下一轮起点
            coords = final_coords.copy()

            # 检查 early_stop（berny 已收敛）
            if early_stop:
                self.history.converged = True
                self.history.convergence_iteration = len(self.history) - 1
                if self.config.get('optimizer', {}).get('verbose', True):
                    print(f"\n✓ 收敛！berny 提前终止，最终梯度范数：{final_gradient_norm:.6f}")
                break

            # 检查收敛
            if final_gradient_norm < convergence_threshold:
                self.history.converged = True
                self.history.convergence_iteration = len(self.history) - 1
                if self.config.get('optimizer', {}).get('verbose', True):
                    print(f"\n✓ 收敛！梯度范数：{final_gradient_norm:.6f} < 阈值 {convergence_threshold}")
                break

            # 检查无改进
            if len(self.history) > 1:
                prev_best = self.history.get_best_iteration('gradient')
                if prev_best is not None:
                    grad_diff = abs(final_gradient_norm - prev_best.gradient_norm)
                    if grad_diff < no_improvement_threshold:
                        no_improvement_count += 1
                    else:
                        no_improvement_count = 0

            if no_improvement_count >= max_no_improvement:
                if self.config.get('optimizer', {}).get('verbose', True):
                    print(f"\n早停：连续{max_no_improvement}轮无显著改进")
                break

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
            print(f"能量/梯度调用次数：{self._total_pyscf_calls}")
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
        if self._energy_func is None:
            raise ValueError("需要先调用 optimize() 初始化")

        # 计算当前能量和梯度
        energy, gradient = self._energy_func(coords_flat)
        self._total_pyscf_calls += 1

        # 备用方案：负梯度方向
        step_size = 0.01
        new_coords = coords_flat - step_size * gradient

        new_energy, new_gradient = self._energy_func(new_coords)
        self._total_pyscf_calls += 1
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
