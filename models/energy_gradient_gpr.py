"""
能量 - 梯度联合预测 GPR 模型（旧版结构）

使用多个独立的 GPR 模型：
- 1 个 GPR 预测能量
- 3N 个 GPR 预测梯度分量（每个维度独立）

优点：
- 梯度直接学习，预测快且准确
- 不依赖数值微分，无累积误差
"""
import numpy as np
import warnings
from typing import Dict, Any, Optional, Tuple, List
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

from models.gpr_base import BaseGPRModel

# 过滤 sklearn GPR 的收敛警告
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.gaussian_process')


class EnergyGradientGPR(BaseGPRModel):
    """
    能量 - 梯度联合预测 GPR 模型（旧版结构）

    使用独立模型：
    - 主模型：预测能量 E(x)
    - 梯度模型：每个梯度分量独立 GPR 预测
    """

    def __init__(self, config: Dict[str, Any], dim: int):
        """
        初始化能量 - 梯度 GPR 模型

        Args:
            config: 配置字典
            dim: 坐标维度（3 * n_atoms）
        """
        super().__init__(config)
        self.name = "EnergyGradientGPR"
        self.dim = dim

        # GPR 配置（改用旧版的小噪声）
        gpr_config = config.get('gpr', {})
        self.noise_variance = gpr_config.get('noise_variance', 1e-4)  # 旧版：1e-4
        self.length_scale = gpr_config.get('length_scale', 1.0)

        # 创建能量 GPR 模型（使用各向同性核）
        kernel = (
            ConstantKernel(1.0, (1e-1, 1e1)) *
            Matern(length_scale=self.length_scale, nu=2.5,
                   length_scale_bounds=(0.1, 100.0)) +
            WhiteKernel(self.noise_variance, (1e-4, 1e-1))
        )

        # 1 个模型用于能量预测
        self.energy_model = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=3,
            random_state=24
        )

        # dim 个模型用于梯度预测（每个分量独立）
        self.gradient_models = []
        for i in range(dim):
            gpr = GaussianProcessRegressor(
                kernel=kernel,
                normalize_y=True,
                n_restarts_optimizer=2,
                random_state=24
            )
            self.gradient_models.append(gpr)

        self.bounds = None
        self.is_trained = False

        # 存储训练数据（使用基类的统一存储）
        # self.X_train: 坐标
        # self.y_train: 能量
        # self.grad_train: 梯度

    def set_bounds(self, bounds: List[Tuple[float, float]]) -> None:
        """设置变量边界"""
        self.bounds = bounds

    def add_data(self, coords: np.ndarray, energy: float,
                 gradient: Optional[np.ndarray] = None) -> None:
        """
        添加训练数据

        Args:
            coords: 坐标 (dim,) 或 (n_atoms, 3)
            energy: 能量 (标量)
            gradient: 梯度 (dim,) - 必须，用于梯度模型训练
        """
        coords_flat = coords.flatten()
        self.X_train.append(coords_flat)
        self.y_train.append(energy)
        if gradient is not None:
            self.grad_train.append(gradient.flatten())
        else:
            # 如果没有梯度数据，填充零
            self.grad_train.append(np.zeros_like(coords_flat))

    def train(self, X: np.ndarray, y: np.ndarray,
              gradients: Optional[np.ndarray] = None) -> None:
        """
        训练 GPR 模型

        Args:
            X: 坐标数据 (n_samples, dim)
            y: 能量数据 (n_samples,)
            gradients: 梯度数据 (n_samples, dim) - 必须
        """
        if len(self.X_train) < 3:
            print("Warning: 训练数据不足（至少 3 个点），跳过训练")
            return

        try:
            X_train = np.array(self.X_train)
            y_train = np.array(self.y_train)
            
            # 使用传入的 gradients 或存储的 grad_train
            if gradients is not None:
                grad_train = np.array(gradients)
            else:
                grad_train = np.array(self.grad_train)

            # 确保形状正确
            if X_train.ndim != 2 or X_train.shape[1] != self.dim:
                print(f"Warning: X_train 形状不正确：{X_train.shape}")
                return

            # 训练能量 GPR
            self.energy_model.fit(X_train, y_train)
            
            # 训练梯度 GPR（每个分量独立）
            for i in range(self.dim):
                self.gradient_models[i].fit(X_train, grad_train[:, i])
            
            self.is_trained = True

            print(f"GPR 模型训练完成，使用 {len(self.X_train)} 个训练点")

        except Exception as e:
            print(f"Error: GPR 训练失败：{e}")
            self.is_trained = False

    def predict(self, x: np.ndarray) -> Tuple[float, float]:
        """
        预测能量和方差（符合基类接口）

        Args:
            x: 坐标 (dim,) 或 (n_atoms, 3)

        Returns:
            energy: 预测能量
            variance: 预测方差
        """
        if not self.is_trained:
            print("Warning: GPR 模型未训练，返回零预测")
            return 0.0, 1.0

        x_flat = x.flatten().reshape(1, -1)

        # 预测能量和方差
        energy, var = self.energy_model.predict(x_flat, return_std=True)

        return energy[0], var[0] ** 2

    def predict_energy_gradient(self, x: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        预测能量和梯度（新版接口，用于 hybrid 优化器）

        Args:
            x: 坐标 (dim,) 或 (n_atoms, 3)

        Returns:
            energy: 预测能量
            gradient: 预测梯度 (与输入形状相同)
        """
        if not self.is_trained:
            print("Warning: GPR 模型未训练，返回零预测")
            x_flat = x.flatten()
            return 0.0, np.zeros_like(x_flat)

        x_flat = x.flatten().reshape(1, -1)

        # 预测能量
        energy = self.energy_model.predict(x_flat)[0]

        # 预测梯度（每个分量独立预测）
        gradient_flat = np.zeros(self.dim)
        for i in range(self.dim):
            result = self.gradient_models[i].predict(x_flat, return_std=False)
            if isinstance(result, tuple):
                gradient_flat[i] = result[0]
            else:
                gradient_flat[i] = result

        # 恢复原始形状
        if x.ndim == 2:
            gradient = gradient_flat.reshape(x.shape)
        else:
            gradient = gradient_flat

        return energy, gradient

    def predict_gradient(self, x: np.ndarray) -> np.ndarray:
        """
        预测梯度（实现基类抽象方法）

        Args:
            x: 坐标

        Returns:
            gradient: 预测梯度
        """
        _, gradient = self.predict_energy_gradient(x)
        return gradient

    def predict_with_uncertainty(self, x: np.ndarray) -> Dict[str, Any]:
        """
        预测能量和梯度，同时返回不确定性

        Args:
            x: 坐标

        Returns:
            dict: {energy, energy_std, gradient, gradient_std}
        """
        if not self.is_trained:
            return {
                'energy': 0.0,
                'energy_std': 1.0,
                'gradient': np.zeros(self.dim),
                'gradient_std': np.ones(self.dim)
            }

        x_flat = x.flatten().reshape(1, -1)

        # 预测能量和标准差
        energy, energy_std = self.energy_model.predict(x_flat, return_std=True)

        # 预测梯度和标准差
        gradient_flat = np.zeros(self.dim)
        gradient_std_flat = np.zeros(self.dim)
        for i in range(self.dim):
            result = self.gradient_models[i].predict(x_flat, return_std=True)
            if isinstance(result, tuple):
                gradient_flat[i] = result[0]
                gradient_std_flat[i] = result[1]
            else:
                gradient_flat[i] = result
                gradient_std_flat[i] = energy_std[0]  # 近似

        return {
            'energy': energy[0],
            'energy_std': energy_std[0],
            'gradient': gradient_flat.reshape(-1, 3) if x.ndim == 2 else gradient_flat,
            'gradient_std': gradient_std_flat.reshape(-1, 3) if x.ndim == 2 else gradient_std_flat
        }

    def clear_data(self) -> None:
        """清除所有训练数据"""
        self.X_train = []
        self.y_train = []
        self.grad_train = []
        self.is_trained = False

    def n_training_points(self) -> int:
        """获取训练点数"""
        return len(self.X_train)

    def acquisition_function(self, x: np.ndarray, y_min: float = None) -> float:
        """
        EI 采集函数（用于建议新采样点）

        Args:
            x: 坐标
            y_min: 当前最优能量

        Returns:
            EI 值
        """
        from scipy.stats import norm

        if y_min is None and len(self.y_train) > 0:
            y_min = min(self.y_train)
        elif y_min is None:
            y_min = 0.0

        if not self.is_trained:
            return 0.0

        x_flat = x.flatten().reshape(1, -1)
        mean, std = self.energy_model.predict(x_flat, return_std=True)

        if std[0] < 1e-10:
            return max(0.0, y_min - mean[0])

        gamma = (y_min - mean[0]) / std[0]
        ei = (y_min - mean[0]) * norm.cdf(gamma) + std[0] * norm.pdf(gamma)

        return ei

    def suggest_next_point(self, bounds: List[Tuple[float, float]],
                           y_min: float = None) -> np.ndarray:
        """
        建议下一个采样点（通过优化采集函数）

        Args:
            bounds: 变量边界
            y_min: 当前最优能量

        Returns:
            x_next: 建议的下一个点
        """
        if bounds is None:
            bounds = self.bounds

        if bounds is None:
            raise ValueError("需要设置边界")

        # 随机生成候选点
        n_candidates = 50
        candidates = []
        for _ in range(n_candidates):
            x = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
            candidates.append(x)

        # 评估采集函数
        best_x = None
        best_ei = -np.inf

        for x in candidates:
            ei = self.acquisition_function(x, y_min)
            if ei > best_ei:
                best_ei = ei
                best_x = x

        return best_x
