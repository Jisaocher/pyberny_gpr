"""
GPR 代理模型基类
定义 GPR 模型的统一接口
"""
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List


class BaseGPRModel(ABC):
    """
    GPR 模型基类
    
    所有 GPR 模型都需要继承此类
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化 GPR 模型
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.name = "BaseGPRModel"
        
        # GPR 参数
        gpr_config = config.get('gpr', {})
        self.n_init = gpr_config.get('n_init', 10)
        self.local_radius = gpr_config.get('local_radius', 0.5)
        self.xi = gpr_config.get('xi', 0.01)  # EI 探索参数
        self.lambda_grad = gpr_config.get('lambda_grad', 0.1)  # 梯度惩罚
        self.max_training_points = gpr_config.get('max_training_points', 30)  # 滑动窗口大小

        # 训练数据
        self.X_train: List[np.ndarray] = []  # 输入坐标
        self.y_train: List[float] = []       # 能量
        self.grad_train: List[np.ndarray] = []  # 梯度

        # 模型
        self.model = None
        self.is_trained = False
    
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray, 
              gradients: Optional[np.ndarray] = None) -> None:
        """
        训练模型
        
        Args:
            X: 输入坐标 (n_samples, n_features)
            y: 能量值 (n_samples,)
            gradients: 梯度 (n_samples, n_features)
        """
        pass
    
    @abstractmethod
    def predict(self, x: np.ndarray) -> Tuple[float, float]:
        """
        预测能量
        
        Args:
            x: 输入坐标
        
        Returns:
            mean: 预测均值
            variance: 预测方差
        """
        pass
    
    @abstractmethod
    def predict_gradient(self, x: np.ndarray) -> np.ndarray:
        """
        预测梯度
        
        Args:
            x: 输入坐标
        
        Returns:
            gradient: 预测梯度
        """
        pass
    
    def add_data(self, x: np.ndarray, energy: float, gradient: np.ndarray) -> None:
        """
        添加训练数据

        Args:
            x: 坐标
            energy: 能量
            gradient: 梯度
        """
        self.X_train.append(x.copy())
        self.y_train.append(energy)
        self.grad_train.append(gradient.copy())

    def limit_training_data(self, max_points: int = None) -> None:
        """
        限制训练数据数量，只保留能量最好的点（滑动窗口）

        Args:
            max_points: 最大训练点数，如果为 None 则使用配置的 max_training_points
        """
        if max_points is None:
            max_points = self.max_training_points

        if len(self.X_train) <= max_points:
            return  # 不需要限制

        # 按能量排序，保留最好的 N 个点
        sorted_idx = np.argsort(self.y_train)
        keep_idx = sorted_idx[:max_points]

        self.X_train = [self.X_train[i] for i in keep_idx]
        self.y_train = [self.y_train[i] for i in keep_idx]
        self.grad_train = [self.grad_train[i] for i in keep_idx]

        self.is_trained = False  # 需要重新训练

    def limit_training_data_by_percentile(self, percentile: float = 50.0) -> None:
        """
        限制训练数据，只保留能量最好的前 percentile% 的点
        
        Args:
            percentile: 保留前百分之多少的点（默认 50，即保留前 50%）
        """
        if len(self.X_train) < 3:
            return  # 数据太少，不限制

        # 计算要保留的点数
        n_keep = max(3, int(len(self.X_train) * percentile / 100.0))
        n_keep = min(n_keep, self.max_training_points)  # 不超过最大限制

        # 按能量排序，保留最好的点
        sorted_idx = np.argsort(self.y_train)
        keep_idx = sorted_idx[:n_keep]

        self.X_train = [self.X_train[i] for i in keep_idx]
        self.y_train = [self.y_train[i] for i in keep_idx]
        self.grad_train = [self.grad_train[i] for i in keep_idx]

        self.is_trained = False  # 需要重新训练

    def clear_data(self) -> None:
        """清除所有训练数据"""
        self.X_train = []
        self.y_train = []
        self.grad_train = []
        self.is_trained = False
        self.model = None
    
    def get_training_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        获取所有训练数据
        
        Returns:
            X: 坐标数组
            y: 能量数组
            gradients: 梯度数组
        """
        X = np.array(self.X_train)
        y = np.array(self.y_train)
        gradients = np.array(self.grad_train) if self.grad_train else None
        return X, y, gradients
    
    def n_training_points(self) -> int:
        """获取训练点数"""
        return len(self.X_train)
    
    @abstractmethod
    def acquisition_function(self, x: np.ndarray, 
                             y_min: float = None) -> float:
        """
        采集函数
        
        Args:
            x: 输入坐标
            y_min: 当前最小能量
        
        Returns:
            acquisition_value: 采集函数值
        """
        pass
    
    def optimize_acquisition(self, bounds: List[Tuple[float, float]],
                             n_restarts: int = 5) -> np.ndarray:
        """
        优化采集函数
        
        Args:
            bounds: 变量边界
            n_restarts: 重启次数
        
        Returns:
            x_next: 下一个采样点
        """
        from scipy.optimize import minimize
        
        dim = len(bounds)
        best_x = None
        best_value = -np.inf
        
        # 多起点优化
        for _ in range(n_restarts):
            # 随机起点
            x0 = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
            
            # 优化采集函数的负值
            result = minimize(
                lambda x: -self.acquisition_function(x),
                x0=x0,
                method='L-BFGS-B',
                bounds=bounds
            )
            
            if -result.fun > best_value:
                best_value = -result.fun
                best_x = result.x
        
        return best_x
    
    def __repr__(self) -> str:
        n_points = self.n_training_points()
        return f"{self.name}(trained={self.is_trained}, n_points={n_points})"
