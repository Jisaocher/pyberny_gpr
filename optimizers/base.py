"""
优化器基类
定义优化器的统一接口
"""
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Tuple
from core.molecule import Molecule, OptimizationHistory, IterationData


class BaseOptimizer(ABC):
    """
    优化器基类
    
    所有优化器都需要继承此类并实现相应方法
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化优化器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.name = "BaseOptimizer"
        self.history = OptimizationHistory()
        self.current_mol = None
        self.atom_symbols = None
        self.calculator = None
        
    @abstractmethod
    def optimize(self, molecule: Molecule, calculator) -> OptimizationHistory:
        """
        执行优化
        
        Args:
            molecule: 初始分子结构
            calculator: 量子化学计算器
        
        Returns:
            OptimizationHistory: 优化历史
        """
        pass
    
    @abstractmethod
    def step(self, coords_flat: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        执行单步优化
        
        Args:
            coords_flat: 当前坐标 (展平)
        
        Returns:
            new_coords: 新坐标
            energy: 能量
            gradient: 梯度
        """
        pass
    
    def check_convergence(self, gradient_norm: float) -> bool:
        """检查是否收敛"""
        threshold = self.config.get('optimizer', {}).get('convergence_threshold', 1e-5)
        return gradient_norm < threshold
    
    def get_iteration_data(self, iteration: int, energy: float,
                           gradient: np.ndarray, coords: np.ndarray,
                           prev_coords: Optional[np.ndarray] = None,
                           round_num: int = 0, stage: str = 'pyberny',
                           gradient_pred: Optional[np.ndarray] = None) -> IterationData:
        """
        创建迭代数据

        Args:
            iteration: 迭代序号
            energy: 能量
            gradient: 梯度
            coords: 当前坐标
            prev_coords: 上一步坐标
            round_num: 轮次编号（混合策略用）
            stage: 阶段标记 ('outer'/'inner'/'pyberny')
            gradient_pred: 预测梯度（混合策略内层用）

        Returns:
            IterationData
        """
        displacement = None
        if prev_coords is not None:
            displacement = coords - prev_coords

        return IterationData(
            iteration=iteration,
            energy=energy,
            gradient=gradient,
            coords=coords,
            displacement=displacement,
            round_num=round_num,
            stage=stage,
            gradient_pred=gradient_pred
        )
    
    def print_iteration(self, iteration: int, energy: float, 
                        gradient_norm: float, displacement: float = None):
        """打印迭代信息"""
        if self.config.get('optimizer', {}).get('verbose', True):
            disp_str = f", disp={displacement:.6f}" if displacement is not None else ""
            print(f"Iter {iteration:4d}: Energy = {energy:.10f} Hartree, "
                  f"|grad| = {gradient_norm:.6f}{disp_str}")
    
    def get_best_result(self) -> Dict:
        """获取最优结果"""
        best = self.history.get_best_iteration()
        if best is None:
            return {}
        
        return {
            'energy': best.energy,
            'gradient_norm': best.gradient_norm,
            'coords': best.coords,
            'iteration': best.iteration
        }
    
    def __repr__(self) -> str:
        return f"{self.name}(config={self.config})"
