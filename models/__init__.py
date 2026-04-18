# Models module initialization

from models.gpr_base import BaseGPRModel
from models.energy_gradient_gpr import EnergyGradientGPR

# 神经网络模块为可选依赖，仅在安装 torch 时可用
try:
    from models.energy_gradient_nn import EnergyGradientNN
    NN_AVAILABLE = True
except ImportError:
    EnergyGradientNN = None  # type: ignore
    NN_AVAILABLE = False

__all__ = [
    'BaseGPRModel',
    'EnergyGradientGPR',
    'EnergyGradientNN',
    'NN_AVAILABLE'
]
