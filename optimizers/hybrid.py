"""
L-BFGS-AI 混合优化策略（融合版）

核心思想：
- 外层：使用 PyBerny (L-BFGS) 真实计算，可靠优化 + 收集训练数据
- 内层：使用 AI 模型预测值 + 自研梯度下降法，快速探索
- 择优：从内外层结果中选择最优作为下一轮起点

验证策略：
- validate_every: 验证频率（每 N 步验证一次）
  - =0: 仅对内层最后一次迭代计算真实值
  - >0: 每 N 步验证一次，预测误差超阈值则提前退出

注意：外层 PyBerny 每轮独立运行，不继承之前轮次的 L-BFGS 历史状态

融合设计：
- 去除独立的初始采样阶段
- 第 1 轮外层迭代使用 n_init + outer_steps 步，融合初始采样和外层优化
- 轨迹输出只区分 Outer/Inner，不再展示"初始采样"标签

支持的 AI 方法：
- gpr: 梯度预测 GPR（默认）
- nn: 神经网络（能量 - 梯度联合预测）
- 可扩展其他 AI 方法（如 KRR、SVR 等）
"""
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import time

from core.molecule import Molecule, OptimizationHistory, IterationData
from core.calculator import QuantumCalculator
from optimizers.base import BaseOptimizer
from models.energy_gradient_gpr import EnergyGradientGPR
from models.gpr_base import BaseGPRModel
from models import NN_AVAILABLE, EnergyGradientNN


class HybridOptimizer(BaseOptimizer):
    """
    L-BFGS-AI 混合优化器（融合版）

    策略流程：
    1. 第 1 轮外层：使用 n_init + outer_steps 步 PyBerny 真实计算（融合初始采样）
    2. 训练 AI 模型：使用滑动窗口管理训练数据
    3. 内层探索：AI 预测 + 自研梯度下降法，快速探索
    4. 验证择优：从内外层结果中选择最优作为下一轮起点
    5. 后续轮次：外层 outer_steps 步 + 内层探索 + 择优

    支持的 AI 方法：
    - gpr: 梯度预测 GPR（默认）
    - nn: 神经网络（能量 - 梯度联合预测）
    - 可扩展其他 AI 方法（如 KRR、SVR 等）
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化混合优化器

        Args:
            config: 配置字典
        """
        super().__init__(config)
        self.name = "L-BFGS-AI Hybrid (New)"

        # 混合策略参数
        hybrid_config = config.get('hybrid', {})
        self.outer_steps = hybrid_config.get('outer_steps', 10)    # 外层步数
        self.inner_steps = hybrid_config.get('inner_steps', 5)     # 内层步数

        # AI 方法选择
        self.ai_method = hybrid_config.get('ai_method', 'gpr')

        # GPR 初始采样点数（用于第 1 轮外层迭代，供 AI 方法通用）
        self.n_init = hybrid_config.get('n_init', 5)

        # 内层优化配置（自研梯度下降法）
        inner_opt_config = hybrid_config.get('inner_opt', {})
        self.inner_gtol = inner_opt_config.get('gtol', 1e-4)
        self.base_step_size = inner_opt_config.get('base_step_size', 0.01)
        self.max_step_size = inner_opt_config.get('max_step_size', 0.1)
        self.min_step_size = inner_opt_config.get('min_step_size', 1e-4)
        self.adaptive_step = inner_opt_config.get('adaptive_step', True)
        self.adaptive_factor = inner_opt_config.get('adaptive_factor', 20.0)  # 自适应步长系数
        self.inner_disp = inner_opt_config.get('disp', False)

        # 验证策略
        self.validate_every = hybrid_config.get('validate_every', 0)  # 0=仅验证最后一点
        self.prediction_error_threshold = hybrid_config.get(
            'prediction_error_threshold', 1e-3
        )

        # 择优策略
        self.selection_metric = hybrid_config.get('selection_metric', 'gradient')
        weights_config = config.get('selection_weights', {})
        self.energy_weight = weights_config.get('energy_weight', 0.3)
        self.gradient_weight = weights_config.get('gradient_weight', 0.7)

        # AI 模型组件（根据 ai_method 动态选择）
        self.ai_model = None  # AI 模型（BaseGPRModel 或其他 AI 模型）
        self.calculator = None
        self.lbfgs_optimizer = None  # PyBerny 优化器（支持状态继承）

        # 状态
        self.current_round = 0
        self._bounds = None

        # 训练数据管理
        self.training_data = {
            'coords': [],
            'energy': [],
            'gradient': []
        }

        # PySCF 调用计数（独立于 training_data，不受滑动窗口限制）
        self.outer_pyscf_calls = 0  # 外层真实计算次数
        self.inner_pyscf_calls = 0  # 内层验证次数

        # 缓存上一轮的终点能量/梯度，避免下一轮起点重复计算
        self._cached_energy = None
        self._cached_gradient = None
        self._cached_coords_hash = None

        # GPR 训练数据：只保留最近指定次数的外层迭代数据
        hybrid_config = config.get('hybrid', {})
        self.max_outer_iterations = hybrid_config.get('max_outer_iterations', 15)
        self.max_training_points = config.get('gpr', {}).get('max_training_points', 30)

    def _initialize_ai_model(self, molecule: Molecule) -> BaseGPRModel:
        """
        根据 ai_method 初始化 AI 模型

        Args:
            molecule: 分子对象

        Returns:
            AI 模型实例
        """
        dim = molecule.n_atoms * 3

        if self.ai_method == 'gpr':
            # 梯度预测 GPR
            if self.config.get('optimizer', {}).get('verbose', True):
                print(f"使用 AI 方法：GPR (梯度预测)")
            return EnergyGradientGPR(self.config, dim)

        elif self.ai_method == 'nn':
            # 神经网络
            if not NN_AVAILABLE:
                raise ImportError(
                    "错误：使用神经网络方法需要安装 torch。\n"
                    "请运行：pip install torch\n"
                    "或者使用 --ai_method gpr 来使用 GPR 方法"
                )
            if self.config.get('optimizer', {}).get('verbose', True):
                print(f"使用 AI 方法：Neural Network (能量 - 梯度联合预测)")
            return EnergyGradientNN(self.config, dim)

        else:
            # 默认使用 GPR
            if self.config.get('optimizer', {}).get('verbose', True):
                print(f"使用 AI 方法：GPR (梯度预测) [默认]")
            return EnergyGradientGPR(self.config, dim)

    def optimize(self, molecule: Molecule, calculator: QuantumCalculator) -> OptimizationHistory:
        """
        执行混合优化

        Args:
            molecule: 初始分子
            calculator: 量子化学计算器

        Returns:
            OptimizationHistory: 优化历史
        """
        self.current_mol = molecule
        self.calculator = calculator
        self.atom_symbols = molecule.atom_symbols
        self.history = OptimizationHistory()

        # 初始化 AI 模型（根据 ai_method 动态选择）
        self.ai_model = self._initialize_ai_model(molecule)

        # 初始化 PyBerny 优化器（用于外层 L-BFGS）
        from optimizers.pyberny_optimizer import PyBernyOptimizer
        self.lbfgs_optimizer = PyBernyOptimizer(self.config)
        # 注意：不需要在这里设置 current_mol 等，run_fixed_steps 会处理

        # 设置边界
        self._setup_bounds(molecule)
        
        # 如果 AI 模型支持 set_bounds 方法，则设置边界
        if hasattr(self.ai_model, 'set_bounds'):
            self.ai_model.set_bounds(self._bounds)

        # 打印开始信息
        if self.config.get('optimizer', {}).get('verbose', True):
            # 根据 AI 方法动态生成标题
            ai_method_names = {
                'gpr': 'GPR',
                'nn': 'Neural Network'
            }
            ai_method_name = ai_method_names.get(self.ai_method, self.ai_method.upper())
            print("=" * 70)
            print(f"PyBerny-{ai_method_name} 混合优化开始")
            print("=" * 70)
            print(f"外层步数：{self.outer_steps}（第 1 轮：{self.n_init + self.outer_steps} 步）")
            print(f"内层步数：{self.inner_steps}")
            print(f"验证频率：{self.validate_every}（0=仅验证最后一点）")
            print(f"预测误差阈值：{self.prediction_error_threshold}")
            # 使用 _get_energy_gradient 计算初始能量并缓存，供第一轮迭代起点复用
            initial_energy, _ = self._get_energy_gradient(molecule.get_coords_flat())
            print(f"初始能量：{initial_energy:.10f} Hartree")
            print("=" * 70)

        self.history.start_time = time.time()

        # 从初始分子开始，不再进行初始采样
        coords = molecule.get_coords_flat()

        # 主循环 - 获取收敛配置
        hybrid_config = self.config.get('hybrid', {})
        conv_config = hybrid_config.get('convergence', {})
        max_rounds = int(conv_config.get('max_rounds', 50))
        convergence_threshold = float(conv_config.get(
            'threshold', self.config.get('optimizer', {}).get('convergence_threshold', 5e-4)
        ))
        max_no_improvement = int(conv_config.get('max_no_improvement', 50))
        no_improvement_threshold = float(conv_config.get('no_improvement_threshold', 1e-6))

        round_num = 0
        no_improvement_count = 0

        while round_num < max_rounds:
            round_num += 1
            self.current_round = round_num

            if self.config.get('optimizer', {}).get('verbose', True):
                print(f"\n{'='*70}")
                print(f"第 {self.current_round} 轮优化")
                print(f"{'='*70}")

            # 保存本轮起点
            # 使用缓存方法，如果起点是上一轮终点（已计算过），则复用缓存
            outer_start_energy, outer_start_gradient = self._get_energy_gradient(coords)
            outer_start = {
                'coords': coords.copy(),
                'energy': outer_start_energy,
                'gradient': outer_start_gradient
            }

            if self.config.get('optimizer', {}).get('verbose', True):
                print(f"本轮起点：E={outer_start_energy:.8f}, |g|={np.linalg.norm(outer_start_gradient):.6f}")

            # 清除缓存（下一轮起点会在 _get_energy_gradient 中自动检查缓存）
            self._cached_energy = None
            self._cached_gradient = None
            self._cached_coords_hash = None

            # 阶段 1： 外层初始采样（仅第 1 轮，融合到外层 L-BFGS 中）
            # 阶段 2A: 外层 L-BFGS（传入起点能量/梯度，避免重复计算）
            # 第 1 轮使用 n_init + outer_steps 步，后续轮次使用 outer_steps 步
            outer_result = self._run_outer_bfgs(
                coords, outer_start_energy, outer_start_gradient,
                is_first_round=(round_num == 1)
            )

            # 检查外层是否提前终止（已满足 scipy 收敛条件）
            if outer_result.get('early_stop', False):
                # 外层 L-BFGS 已收敛，直接使用外层结果作为最终结果
                coords = outer_result['coords'].copy()
                
                if self.config.get('optimizer', {}).get('verbose', True):
                    print(f"\n外层 L-BFGS 已收敛，跳过内层探索")
                    print(f"最终结果：E={outer_result['energy']:.8f}, |g|={np.linalg.norm(outer_result['gradient']):.6f}")
                
                # 设置收敛标志
                self.history.converged = True
                self.history.convergence_iteration = len(self.history) - 1
                break

            # 阶段 2B: 训练 GPR（已兼容NN模型）
            self._train_gpr()

            # 阶段 2C: 内层探索（从外层终点开始，而不是起点！）
            inner_result = self._run_inner_exploration(outer_result)
            
            # 阶段 2D: 验证择优
            best_candidate = self._select_best_candidate(
                outer_start, outer_result, inner_result
            )
            
            # 更新到下一轮起点
            coords = best_candidate['coords'].copy()
            
            # 记录本轮数据到历史
            # 择优点作为下一轮outer起点记录到历史，此处不需要重复记录
            # self._record_round_history(outer_start, outer_result, inner_result, best_candidate)
            
            if self.config.get('optimizer', {}).get('verbose', True):
                print(f"\n本轮最佳：{best_candidate['source']}")
                print(f"下一轮起点：E={best_candidate['energy']:.8f}, |g|={np.linalg.norm(best_candidate['gradient']):.6f}")
            
            # 阶段 2E: 收敛检查
            gradient_norm = np.linalg.norm(best_candidate['gradient'])
            
            if gradient_norm < convergence_threshold:
                self.history.converged = True
                self.history.convergence_iteration = len(self.history) - 1
                if self.config.get('optimizer', {}).get('verbose', True):
                    print(f"\n✓ 收敛！梯度范数：{gradient_norm:.6f} < 阈值 {convergence_threshold}")
                break
            
            # 检查无改进
            if len(self.history) > 1:
                prev_best = self.history.get_best_iteration('gradient')
                if prev_best is not None:
                    grad_diff = abs(gradient_norm - prev_best.gradient_norm)
                    if grad_diff < no_improvement_threshold:
                        no_improvement_count += 1
                    else:
                        no_improvement_count = 0
            
            if no_improvement_count >= max_no_improvement:
                if self.config.get('optimizer', {}).get('verbose', True):
                    print(f"\n早停：连续{max_no_improvement}轮无显著改进")
                break
        
        self.history.end_time = time.time()
        
        # 设置最终结果
        if self.history.iterations:
            best_iteration = min(self.history.iterations, key=lambda x: x.gradient_norm)
            self.history.converged = True
            self.history.best_iteration = best_iteration
        
        # 打印总结
        if self.config.get('optimizer', {}).get('verbose', True):
            self._print_summary()
        
        return self.history

    def _setup_bounds(self, molecule: Molecule) -> None:
        """设置优化边界"""
        coords = molecule.coords
        radius = self.config.get('gpr', {}).get('local_radius', 0.5)
        
        self._bounds = []
        for i in range(molecule.n_atoms):
            for j in range(3):
                low = coords[i, j] - radius
                high = coords[i, j] + radius
                self._bounds.append((low, high))

    def _run_outer_bfgs(self, coords: np.ndarray,
                        initial_energy: Optional[float] = None,
                        initial_gradient: Optional[np.ndarray] = None,
                        is_first_round: bool = False) -> Dict[str, Any]:
        """
        阶段 2A: 外层 BFGS（使用 PyBerny，支持跨轮次继承 L-BFGS 历史）

        使用 PyBerny 进行真实计算，可靠优化，同时收集训练数据
        PyBerny 内部维护 L-BFGS 的 s, y 向量对，可在轮次间继承

        Args:
            coords: 初始坐标
            initial_energy: 初始能量（可选，如果提供则避免重复计算）
            initial_gradient: 初始梯度（可选，如果提供则避免重复计算）
            is_first_round: 是否为第 1 轮（如果是，则使用 n_init + outer_steps 步）

        Returns:
            dict: {coords, energy, gradient, history}
        """
        # 第 1 轮使用 n_init + outer_steps 步，后续轮次使用 outer_steps 步
        actual_outer_steps = self.n_init + self.outer_steps if is_first_round else self.outer_steps
        
        if self.config.get('optimizer', {}).get('verbose', True):
            print(f"\n[外层 PyBerny] 执行 {actual_outer_steps} 步...")

        history = []

        # 使用 PyBerny 的 run_fixed_steps 方法执行固定步数的 L-BFGS 优化
        final_coords, lbfgs_history, outer_pyscf_calls = self.lbfgs_optimizer.run_fixed_steps(
            coords,
            actual_outer_steps,
            self.calculator,
            self.atom_symbols,
            initial_energy,
            initial_gradient
        )

        # 累加 PySCF 调用次数（外层真实计算，已排除复用的初始点）
        self.outer_pyscf_calls += outer_pyscf_calls

        # 将 PyBerny 的历史记录添加到总历史和训练数据（stage='outer'）
        for iteration in lbfgs_history.iterations:
            # 添加到总历史（更新 round_num 和 stage）
            outer_data = IterationData(
                iteration=iteration.iteration,
                energy=iteration.energy,
                gradient=iteration.gradient,
                coords=iteration.coords.copy(),
                displacement=iteration.displacement.copy() if iteration.displacement is not None else None,
                round_num=self.current_round,
                stage='outer'
            )
            self.history.add_iteration(outer_data)
            
            history.append({
                'coords': iteration.coords.copy(),
                'energy': iteration.energy,
                'gradient': iteration.gradient
            })

            # 添加到训练数据
            self.training_data['coords'].append(iteration.coords.copy())
            self.training_data['energy'].append(iteration.energy)
            self.training_data['gradient'].append(iteration.gradient.copy())

            # 注意：不额外打印，pyberny_optimizer 已经打印了 Iter X: 信息

        # 从最后一次迭代获取能量和梯度（避免重复计算）
        if history:
            final_energy = history[-1]['energy']
            final_gradient = history[-1]['gradient']
        else:
            # 如果没有历史（一步都没走），使用 calculate_energy_gradient 一次计算
            final_energy, final_gradient = self.calculator.calculate_energy_gradient(
                self.atom_symbols, final_coords.reshape(-1, 3)
            )
            self.outer_pyscf_calls += 1

        # 检查是否提前终止（实际迭代次数 < 设置的 outer_steps）
        actual_steps = len(history)
        early_stop = actual_steps < actual_outer_steps

        if self.config.get('optimizer', {}).get('verbose', True):
            print(f"外层终点：E={final_energy:.8f}, |g|={np.linalg.norm(final_gradient):.6f}")
            if early_stop:
                print(f"⚠ 外层 PyBerny 提前终止（实际 {actual_steps} 步 < 设置 {actual_outer_steps} 步）")
                print(f"  原因：PyBerny L-BFGS 已满足收敛条件")
            else:
                print(f"外层 PyBerny 完成 {actual_steps} 步")

        return {
            'coords': final_coords,
            'energy': final_energy,
            'gradient': final_gradient,
            'history': history,
            'early_stop': early_stop,
            'actual_steps': actual_steps
        }

    def _train_gpr(self) -> None:
        """
        阶段 2B: 训练 AI 模型

        使用最近的外层迭代数据训练（按梯度筛选）
        """
        # 限制训练数据量（只保留最近的外层迭代 + 按梯度筛选）
        self._limit_training_data()

        # 转换为 numpy 数组
        X = np.array(self.training_data['coords'])
        y = np.array(self.training_data['energy'])
        gradients = np.array(self.training_data['gradient'])

        # 训练 AI 模型
        if hasattr(self.ai_model, 'clear_data'):
            self.ai_model.clear_data()
        for i in range(len(X)):
            self.ai_model.add_data(X[i], y[i], gradients[i])

        # 训练 AI 模型（优先使用 fit 方法，兼容旧版 train 方法）
        if hasattr(self.ai_model, 'fit'):
            self.ai_model.fit(X, y, gradients)
        elif hasattr(self.ai_model, 'train'):
            self.ai_model.train(X, y, gradients)

    def _limit_training_data(self) -> None:
        """
        限制训练数据量（按梯度范数筛选的滑动窗口）

        策略：
        1. 只保留最近 max_outer_iterations 次外层迭代的数据
        2. 在窗口内按梯度范数排序，梯度范数越小越优先
        """
        n_points = len(self.training_data['coords'])

        if n_points <= self.max_training_points:
            return  # 不需要限制

        # 策略 1: 只保留最近的 max_outer_iterations 次数据
        if n_points > self.max_outer_iterations:
            # 保留最近的 max_outer_iterations 个点
            start_idx = max(0, n_points - self.max_outer_iterations)
            self.training_data['coords'] = self.training_data['coords'][start_idx:]
            self.training_data['energy'] = self.training_data['energy'][start_idx:]
            self.training_data['gradient'] = self.training_data['gradient'][start_idx:]
            n_points = len(self.training_data['coords'])

        if n_points <= self.max_training_points:
            if self.config.get('optimizer', {}).get('verbose', True):
                print(f"训练数据：{n_points} 点（最近 {self.max_outer_iterations} 次外层迭代）")
            return

        # 策略 2: 在窗口内按梯度范数排序，保留梯度范数最小的 N 个点
        n_keep = self.max_training_points

        # 计算每个点的梯度范数
        gradient_norms = []
        for grad in self.training_data['gradient']:
            grad_array = np.array(grad)
            gradient_norms.append(np.linalg.norm(grad_array))
        gradient_norms = np.array(gradient_norms)

        # 按梯度范数排序（越小越优先）
        sorted_indices = np.argsort(gradient_norms)

        # 保留梯度范数最小的 n_keep 个点
        keep_indices = set(sorted_indices[:n_keep])

        # 过滤
        new_coords = []
        new_energy = []
        new_gradient = []

        for i in range(n_points):
            if i in keep_indices:
                new_coords.append(self.training_data['coords'][i])
                new_energy.append(self.training_data['energy'][i])
                new_gradient.append(self.training_data['gradient'][i])

        self.training_data['coords'] = new_coords
        self.training_data['energy'] = new_energy
        self.training_data['gradient'] = new_gradient

        if self.config.get('optimizer', {}).get('verbose', True):
            print(f"训练数据：按梯度范数筛选保留 {len(new_coords)} 点")

    def _run_inner_exploration(self, outer_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        阶段 2C: 内层 AI 探索（自研梯度下降法）

        使用 AI 模型预测梯度 + 梯度下降法快速探索
        AI 模型只预测梯度，不预测能量
        记录每一步的预测梯度到历史
        仅在内层最后一步计算真实能量和梯度

        Args:
            outer_result: 外层 L-BFGS 的终点（作为内层起点）

        Returns:
            dict: {coords, energy_true, gradient, early_stop, actual_steps}
        """
        verbose = self.config.get('optimizer', {}).get('verbose', True)

        if verbose:
            print(f"\n[内层探索] 执行 {self.inner_steps} 步（AI 预测梯度，梯度下降法）...")
            print(f"内层起点：E={outer_result['energy']:.8f}, |g|={np.linalg.norm(outer_result['gradient']):.6f}")

        # 从外层终点开始内层探索
        coords = outer_result['coords'].copy()

        # 记录上一步的坐标和梯度（用于停滞检测）
        prev_coords = None
        prev_gradient = None
        early_stop = False
        actual_steps = 0

        # 自适应步长参数
        step_size = self.base_step_size

        for step in range(self.inner_steps):
            actual_steps = step + 1

            # 1. AI 模型预测当前梯度（只预测梯度，不预测能量）
            gradient_pred = self.ai_model.predict_gradient(coords)
            g_norm = np.linalg.norm(gradient_pred)

            # 2. 检查收敛（梯度足够小）
            if g_norm < self.inner_gtol:
                if verbose:
                    print(f"  Step {step + 1}: |g|={g_norm:.6f} < gtol={self.inner_gtol}, 已收敛")
                # 记录最后一步到历史（stage='inner'）
                inner_data = IterationData(
                    iteration=actual_steps,
                    energy=outer_result['energy'],  # 使用外层终点能量作为占位符
                    gradient=gradient_pred,  # 使用预测梯度作为真实梯度的占位符
                    coords=coords.copy(),
                    displacement=coords - prev_coords if prev_coords is not None else None,
                    round_num=self.current_round,
                    stage='inner',
                    gradient_pred=gradient_pred
                )
                self.history.add_iteration(inner_data)
                break

            # 3. 自适应步长：梯度越大，步长越大（初期快速探索，后期精细优化）
            if self.adaptive_step:
                # step = base_step * (1 + g_norm * adaptive_factor)
                adaptive_factor = 1.0 * (0.1 + g_norm * self.adaptive_factor)
                step_size = self.base_step_size * adaptive_factor
                # 限制步长范围
                step_size = max(self.min_step_size, min(step_size, self.max_step_size))

            # 4. 梯度下降步进：x_new = x - alpha * g
            new_coords = coords - step_size * gradient_pred

            # 5. 计算位移
            displacement = np.linalg.norm(new_coords - coords)

            # 6. 检测停滞（坐标变化极小）
            stalled = False
            if prev_coords is not None:
                coord_change = np.linalg.norm(new_coords - prev_coords)
                if coord_change < 1e-10:
                    stalled = True

            if verbose:
                stall_marker = " [停滞]" if stalled else ""
                print(f"  Step {step + 1}: |g_pred|={g_norm:.6f}, step={step_size:.6f}, "
                      f"d={displacement:.6f}{stall_marker}")

            # 记录当前步到历史（stage='inner'）
            inner_data = IterationData(
                iteration=actual_steps,
                energy=outer_result['energy'],  # 使用外层终点能量作为占位符
                gradient=gradient_pred,  # 使用预测梯度作为真实梯度的占位符
                coords=coords.copy(),
                displacement=coords - prev_coords if prev_coords is not None else None,
                round_num=self.current_round,
                stage='inner',
                gradient_pred=gradient_pred
            )
            self.history.add_iteration(inner_data)

            # 7. 检测连续停滞，提前退出
            if stalled and step > 0:
                if verbose:
                    print(f"  → 检测到重复迭代，提前退出内层探索")
                early_stop = True
                break

            # 保存上一步状态
            prev_coords = coords.copy()
            prev_gradient = gradient_pred

            # 更新坐标
            coords = new_coords

        # 内层最后一步：计算真实能量和梯度（用于与外层择优）
        coords_reshaped = coords.reshape(-1, 3)
        inner_energy_true, inner_gradient_true = self.calculator.calculate_energy_gradient(
            self.atom_symbols, coords_reshaped
        )

        # 计数 PySCF 调用（内层验证）
        self.inner_pyscf_calls += 1

        if verbose:
            print(f"内层终点：E_true={inner_energy_true:.8f}, |g_true|={np.linalg.norm(inner_gradient_true):.6f}")
            print(f"内层实际执行：{actual_steps} 步 / 设置 {self.inner_steps} 步")
            print(f"验证状态：{'提前退出' if early_stop else '完成'}")

        return {
            'coords': coords,
            'energy_true': inner_energy_true,
            'gradient': inner_gradient_true,
            'early_stop': early_stop,
            'actual_steps': actual_steps
        }

    def _select_best_candidate(self, outer_start: Dict, outer_result: Dict,
                               inner_result: Dict) -> Dict[str, Any]:
        """
        阶段 2D: 验证择优

        从候选点中选择最优作为下一轮起点

        Args:
            outer_start: 外层起点
            outer_result: 外层结果
            inner_result: 内层结果

        Returns:
            dict: 最佳候选点
        """
        candidates = []

        # 候选 1: 外层终点（已验证）
        candidates.append({
            'source': 'outer_final',
            'coords': outer_result['coords'],
            'energy': outer_result['energy'],
            'gradient': outer_result['gradient'],
            'verified': True
        })

        # 候选 2: 内层终点（已验证）- 只有在没有提前退出时才考虑
        if not inner_result['early_stop']:
            candidates.append({
                'source': 'inner_final',
                'coords': inner_result['coords'],
                'energy': inner_result['energy_true'],
                'gradient': inner_result['gradient'],
                'verified': True
            })
        elif self.config.get('optimizer', {}).get('verbose', True):
            print(f"  内层探索提前退出，内层终点不参与择优")

        # 如果只有外层终点作为候选（内层提前退出），直接返回
        if len(candidates) == 1:
            if self.config.get('optimizer', {}).get('verbose', True):
                print(f"\n择优结果：")
                print(f"  ✓ outer_final: E={candidates[0]['energy']:.8f}, "
                      f"|g|={np.linalg.norm(candidates[0]['gradient']):.6f}")
            
            # 缓存最佳候选点的能量/梯度，供下一轮起点复用
            self._cached_energy = candidates[0]['energy']
            self._cached_gradient = candidates[0]['gradient']
            self._cached_coords_hash = hash(candidates[0]['coords'].tobytes())
            
            return candidates[0]

        # 计算评分（相对于外层起点）
        E_start = outer_start['energy']
        g_start_norm = np.linalg.norm(outer_start['gradient'])

        for cand in candidates:
            ΔE = cand['energy'] - E_start
            Δg = np.linalg.norm(cand['gradient']) - g_start_norm

            # 归一化
            ΔE_norm = ΔE / abs(E_start) if abs(E_start) > 1e-10 else ΔE
            Δg_norm = Δg / g_start_norm if g_start_norm > 1e-10 else Δg

            # 加权评分
            cand['score'] = self.energy_weight * ΔE_norm + self.gradient_weight * Δg_norm

        # 选择评分最小的
        best = min(candidates, key=lambda x: x['score'])

        if self.config.get('optimizer', {}).get('verbose', True):
            print(f"\n择优结果：")
            for cand in candidates:
                marker = "✓" if cand == best else " "
                print(f"  {marker} {cand['source']}: "
                      f"E={cand['energy']:.8f}, |g|={np.linalg.norm(cand['gradient']):.6f}, "
                      f"score={cand['score']:.6f}")

        # 缓存最佳候选点的能量/梯度，供下一轮起点复用
        self._cached_energy = best['energy']
        self._cached_gradient = best['gradient']
        self._cached_coords_hash = hash(best['coords'].tobytes())

        return best

    def _record_round_history(self, outer_start: Dict, outer_result: Dict,
                              inner_result: Dict, best_candidate: Dict) -> None:
        """记录本轮历史"""
        # 注意：外层和内层的关键点已经在各自函数的 callback 中记录到 history
        # 这里只需要记录择优结果（作为本轮的代表点）

        # 记录最佳候选点
        data = IterationData(
            iteration=len(self.history),
            energy=best_candidate['energy'],
            gradient=best_candidate['gradient'],
            coords=best_candidate['coords']
        )
        self.history.add_iteration(data)

    def _get_energy_gradient(self, coords_flat: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        获取能量和梯度（带缓存，避免重复计算）

        Args:
            coords_flat: 坐标

        Returns:
            energy, gradient
        """
        # 检查是否命中缓存（相同的坐标）
        coords_hash = hash(coords_flat.tobytes())
        if self._cached_coords_hash is not None and coords_hash == self._cached_coords_hash:
            # print(f"  → 使用缓存的能量和梯度（命中缓存）")
            return self._cached_energy, self._cached_gradient

        # 未命中缓存，计算并更新缓存
        # print(f"  → 计算能量和梯度（未命中缓存）")
        self.outer_pyscf_calls += 1
        energy, gradient = self.calculator.calculate_energy_gradient(
            self.atom_symbols, coords_flat.reshape(-1, 3)
        )

        self._cached_energy = energy
        self._cached_gradient = gradient
        self._cached_coords_hash = coords_hash

        return energy, gradient

    def _print_summary(self) -> None:
        """打印优化总结"""
        print("\n" + "=" * 70)
        print("优化完成！")
        print("=" * 70)

        best = self.history.get_best_iteration('gradient')
        if best:
            print(f"最优能量（全局梯度最小）：{best.energy:.10f} Hartree")
            print(f"最优梯度范数：{best.gradient_norm:.6f}")
            print(f"最优迭代：{best.iteration}")

        print(f"总迭代次数：{len(self.history)}")
        print(f"总轮数：{self.current_round}")
        print(f"收敛状态：{'收敛' if self.history.converged else '未收敛'}")
        print(f"计算时间：{self.history.end_time - self.history.start_time:.2f} 秒")
        print(f"AI 模型训练点数：{self.ai_model.n_training_points()}")

        # PySCF 调用次数 = 外层真实计算次数 + 内层验证次数
        # 使用独立计数器，不受 training_data 滑动窗口限制
        total_pyscf_calls = self.outer_pyscf_calls + self.inner_pyscf_calls

        print(f"PySCF 调用次数：{total_pyscf_calls} (外层：{self.outer_pyscf_calls}, 内层验证：{self.inner_pyscf_calls})")
        print(f"内层探索步数：{self.current_round * self.inner_steps}（使用 AI 预测，节省 PySCF 计算）")
        print("=" * 70)

    def step(self, coords_flat: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
        """单步优化（用于兼容基类接口）"""
        raise NotImplementedError("HybridOptimizer 需要使用 optimize() 方法")


def run_hybrid_optimization(molecule: Molecule, config: Dict[str, Any]) -> OptimizationHistory:
    """
    便捷函数：运行混合优化
    
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
    
    optimizer = HybridOptimizer(config)
    return optimizer.optimize(molecule, calculator)
