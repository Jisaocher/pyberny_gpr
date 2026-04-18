"""
能量 - 梯度联合预测神经网络模型

使用单个神经网络同时预测能量和梯度：
- 输入：3N 维坐标
- 输出：能量（标量）+ 梯度（3N 维）
- 共享隐藏层提取特征，多任务学习

训练数据：
- X: (n_samples, 3N) - 坐标
- y: (n_samples,) - 能量
- gradients: (n_samples, 3N) - 梯度

预测时：
- 输入：x (3N,) - 坐标
- 输出：gradient (3N,) - 梯度（与 GPR 接口一致）
"""
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from models.gpr_base import BaseGPRModel


class EnergyGradientNN(nn.Module, BaseGPRModel):
    """
    能量 - 梯度联合预测神经网络模型

    网络结构：
    - 输入层：3N 维坐标
    - 共享隐藏层：多层全连接 + 激活 + BatchNorm + Dropout
    - 能量头：输出标量能量
    - 梯度头：输出 3N 维梯度

    损失函数：
    L = energy_weight * MSE(E_pred, E_true) + gradient_weight * MSE(g_pred, g_true)
    """

    def __init__(self, config: Dict[str, Any], dim: int):
        """
        初始化神经网络模型

        Args:
            config: 配置字典
            dim: 坐标维度（3 * n_atoms）
        """
        nn.Module.__init__(self)  # 初始化 nn.Module
        BaseGPRModel.__init__(self, config)  # 初始化 BaseGPRModel
        self.name = "EnergyGradientNN"
        self.dim = dim

        # 神经网络配置
        nn_config = config.get('neural_network', {})
        self.hidden_layers = nn_config.get('hidden_layers', [128, 64, 32])
        self.activation_name = nn_config.get('activation', 'relu')
        self.use_batchnorm = nn_config.get('use_batchnorm', True)
        self.dropout_rate = nn_config.get('dropout_rate', 0.1)
        
        # 训练参数
        self.learning_rate = nn_config.get('learning_rate', 0.001)
        self.batch_size = nn_config.get('batch_size', 16)
        self.max_epochs = nn_config.get('max_epochs', 500)
        self.early_stopping_patience = nn_config.get('early_stopping_patience', 50)
        self.validation_split = nn_config.get('validation_split', 0.2)
        
        # 损失函数权重
        self.energy_weight = nn_config.get('energy_weight', 1.0)
        self.gradient_weight = nn_config.get('gradient_weight', 0.1)
        
        # 优化器
        self.optimizer_type = nn_config.get('optimizer', 'adam')
        self.weight_decay = nn_config.get('weight_decay', 0.01)
        
        # 归一化参数
        self.normalize_input = nn_config.get('normalize_input', True)
        self.normalize_output = nn_config.get('normalize_output', True)
        
        # 归一化统计量（训练时计算）
        self.input_mean: Optional[np.ndarray] = None
        self.input_std: Optional[np.ndarray] = None
        self.energy_mean: Optional[float] = None
        self.energy_std: Optional[float] = None
        self.gradient_mean: Optional[np.ndarray] = None
        self.gradient_std: Optional[np.ndarray] = None
        
        # 构建网络
        self._build_network()

    def _build_network(self) -> None:
        """构建神经网络"""
        activation_map = {
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'gelu': nn.GELU,
            'sigmoid': nn.Sigmoid,
            'elu': nn.ELU
        }
        activation_fn = activation_map.get(self.activation_name, nn.ReLU)
        
        # 输入层 + 共享隐藏层
        layers = []
        input_dim = self.dim
        for i, hidden_size in enumerate(self.hidden_layers):
            layers.append(nn.Linear(input_dim, hidden_size))
            if self.use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(activation_fn())
            if self.dropout_rate > 0:
                layers.append(nn.Dropout(self.dropout_rate))
            input_dim = hidden_size
        
        self.shared_layers = nn.Sequential(*layers)
        
        # 能量头（标量输出）
        self.energy_head = nn.Linear(self.hidden_layers[-1] if self.hidden_layers else self.dim, 1)
        
        # 梯度头（3N 维输出）
        self.gradient_head = nn.Linear(self.hidden_layers[-1] if self.hidden_layers else self.dim, self.dim)
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
        # 初始化优化器
        self._init_optimizer()

    def _init_optimizer(self) -> None:
        """初始化优化器"""
        params = self.parameters()
        if self.optimizer_type == 'adam':
            self.optimizer = optim.Adam(params, lr=self.learning_rate, 
                                        weight_decay=self.weight_decay)
        elif self.optimizer_type == 'adamw':
            self.optimizer = optim.AdamW(params, lr=self.learning_rate,
                                         weight_decay=self.weight_decay)
        elif self.optimizer_type == 'sgd':
            self.optimizer = optim.SGD(params, lr=self.learning_rate,
                                       weight_decay=self.weight_decay,
                                       momentum=0.9)
        else:
            self.optimizer = optim.Adam(params, lr=self.learning_rate)

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=20
        )

    def _compute_normalization_params(self, X: np.ndarray, y: np.ndarray, 
                                       gradients: np.ndarray) -> None:
        """计算归一化参数"""
        if self.normalize_input:
            self.input_mean = X.mean(axis=0)
            self.input_std = X.std(axis=0) + 1e-8
        
        if self.normalize_output:
            self.energy_mean = float(y.mean())
            self.energy_std = float(y.std()) + 1e-8
            self.gradient_mean = gradients.mean(axis=0)
            self.gradient_std = gradients.std(axis=0) + 1e-8

    def _normalize(self, X: np.ndarray, y: np.ndarray, 
                   gradients: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """归一化数据"""
        X_norm = X.copy()
        y_norm = y.copy()
        grad_norm = gradients.copy()
        
        if self.normalize_input and self.input_mean is not None:
            X_norm = (X - self.input_mean) / self.input_std
        
        if self.normalize_output:
            if self.energy_mean is not None:
                y_norm = (y - self.energy_mean) / self.energy_std
            if self.gradient_mean is not None:
                grad_norm = (gradients - self.gradient_mean) / self.gradient_std
        
        return X_norm, y_norm, grad_norm

    def _denormalize(self, energy: np.ndarray, gradient: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """反归一化预测结果"""
        energy_out = energy.copy()
        gradient_out = gradient.copy()
        
        if self.normalize_output:
            if self.energy_std is not None:
                energy_out = energy * self.energy_std + self.energy_mean
            if self.gradient_std is not None:
                gradient_out = gradient * self.gradient_std + self.gradient_mean
        
        return energy_out, gradient_out

    def fit(self, X: np.ndarray, y: np.ndarray,
              gradients: Optional[np.ndarray] = None) -> None:
        """
        训练神经网络模型

        Args:
            X: 坐标数据 (n_samples, dim)
            y: 能量数据 (n_samples,)
            gradients: 梯度数据 (n_samples, dim)
        """
        if len(self.X_train) < 3:
            print("Warning: 训练数据不足（至少 3 个点），跳过训练")
            return

        X_train = np.array(self.X_train)
        y_train = np.array(self.y_train)
        grad_train = np.array(self.grad_train)

        # 计算归一化参数
        self._compute_normalization_params(X_train, y_train, grad_train)
        
        # 归一化
        X_norm, y_norm, grad_norm = self._normalize(X_train, y_train, grad_train)

        # 转换为张量
        X_tensor = torch.FloatTensor(X_norm).to(self.device)
        y_tensor = torch.FloatTensor(y_norm).unsqueeze(1).to(self.device)
        grad_tensor = torch.FloatTensor(grad_norm).to(self.device)

        # 创建数据集
        dataset = TensorDataset(X_tensor, y_tensor, grad_tensor)
        
        # 划分训练集和验证集
        n_total = len(dataset)
        n_val = max(1, int(n_total * self.validation_split))
        n_train = n_total - n_val
        
        # 如果数据太少，不划分验证集
        if n_total <= 5:
            train_loader = DataLoader(dataset, batch_size=min(self.batch_size, n_total), shuffle=True)
            val_loader = None
        else:
            train_dataset, val_dataset = random_split(
                dataset, [n_train, n_val], 
                generator=torch.Generator().manual_seed(42)
            )
            train_loader = DataLoader(train_dataset, batch_size=min(self.batch_size, n_train), shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # 训练循环
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        for epoch in range(self.max_epochs):
            # 训练阶段
            self.train()  # 设置训练模式
            train_loss = 0.0
            n_batches = 0
            
            for X_batch, y_batch, grad_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                grad_batch = grad_batch.to(self.device)
                
                self.optimizer.zero_grad()
                
                # 前向传播
                shared = self.shared_layers(X_batch)
                energy_pred = self.energy_head(shared)
                gradient_pred = self.gradient_head(shared)
                
                # 计算损失
                energy_loss = nn.MSELoss()(energy_pred, y_batch)
                gradient_loss = nn.MSELoss()(gradient_pred, grad_batch)
                total_loss = self.energy_weight * energy_loss + self.gradient_weight * gradient_loss
                
                # 反向传播
                total_loss.backward()
                self.optimizer.step()
                
                train_loss += total_loss.item()
                n_batches += 1
            
            train_loss /= n_batches
            self.scheduler.step(train_loss)

            # 验证阶段
            val_loss = train_loss  # 如果没有验证集，使用训练损失
            if val_loader is not None:
                self.eval()  # 设置评估模式
                val_loss = 0.0
                n_val_batches = 0
                with torch.no_grad():
                    for X_batch, y_batch, grad_batch in val_loader:
                        X_batch = X_batch.to(self.device)
                        y_batch = y_batch.to(self.device)
                        grad_batch = grad_batch.to(self.device)
                        
                        shared = self.shared_layers(X_batch)
                        energy_pred = self.energy_head(shared)
                        gradient_pred = self.gradient_head(shared)
                        
                        energy_loss = nn.MSELoss()(energy_pred, y_batch)
                        gradient_loss = nn.MSELoss()(gradient_pred, grad_batch)
                        total_loss = self.energy_weight * energy_loss + self.gradient_weight * gradient_loss
                        
                        val_loss += total_loss.item()
                        n_val_batches += 1
                val_loss /= n_val_batches
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}
            else:
                patience_counter += 1
            
            if patience_counter >= self.early_stopping_patience:
                break

        # 恢复最佳模型
        if best_model_state is not None:
            self.load_state_dict(best_model_state)
            # 将最佳模型移回设备
            self.to(self.device)

        self.is_trained = True
        print(f"NN 模型训练完成，使用 {len(self.X_train)} 个训练点，最佳验证损失：{best_val_loss:.6f}")

    def predict(self, x: np.ndarray) -> Tuple[float, float]:
        """
        预测能量和方差

        Args:
            x: 坐标 (dim,) 或 (n_atoms, 3)

        Returns:
            energy: 预测能量
            variance: 预测方差（使用 dropout 不确定性估计）
        """
        if not self.is_trained:
            return 0.0, 1.0

        x_flat = x.flatten()
        
        # 归一化
        if self.normalize_input and self.input_mean is not None:
            x_norm = (x_flat - self.input_mean) / self.input_std
        else:
            x_norm = x_flat

        x_tensor = torch.FloatTensor(x_norm).unsqueeze(0).to(self.device)

        self.eval()
        with torch.no_grad():
            shared = self.shared_layers(x_tensor)
            energy_norm = self.energy_head(shared).cpu().numpy()[0, 0]

        # 反归一化
        if self.normalize_output and self.energy_std is not None:
            energy = energy_norm * self.energy_std + self.energy_mean
        else:
            energy = energy_norm

        # 方差估计（使用 dropout 蒙特卡洛采样）
        variance = self._estimate_variance(x_tensor)

        return float(energy), float(variance)

    def _estimate_variance(self, x_tensor: torch.Tensor, n_samples: int = 10) -> float:
        """使用 dropout 蒙特卡洛采样估计预测方差"""
        # 注意：不能使用 self.train() 因为 BatchNorm 需要 batch_size > 1
        # 我们使用 eval 模式 + 手动启用 dropout 来实现 MC Dropout
        predictions = []

        # 临时启用 dropout
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()  # 启用 dropout

        with torch.no_grad():
            self.eval()  # 其他层使用评估模式
            for _ in range(n_samples):
                shared = self.shared_layers(x_tensor)
                energy_norm = self.energy_head(shared).cpu().numpy()[0, 0]
                predictions.append(energy_norm)

        # 恢复 dropout 到评估模式
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.eval()

        variance = float(np.var(predictions))
        return variance

    def predict_energy_gradient(self, x: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        预测能量和梯度（用于 hybrid 优化器内层探索）

        Args:
            x: 坐标 (dim,) 或 (n_atoms, 3)

        Returns:
            energy: 预测能量
            gradient: 预测梯度 (与输入形状相同)
        """
        if not self.is_trained:
            x_flat = x.flatten()
            return 0.0, np.zeros_like(x_flat)

        x_flat = x.flatten()
        
        # 归一化
        if self.normalize_input and self.input_mean is not None:
            x_norm = (x_flat - self.input_mean) / self.input_std
        else:
            x_norm = x_flat

        x_tensor = torch.FloatTensor(x_norm).unsqueeze(0).to(self.device)

        self.eval()
        with torch.no_grad():
            shared = self.shared_layers(x_tensor)
            energy_norm = self.energy_head(shared).cpu().numpy()[0, 0]
            gradient_norm = self.gradient_head(shared).cpu().numpy()[0]

        # 反归一化
        energy_arr, gradient_arr = self._denormalize(
            np.array([energy_norm]),
            gradient_norm.reshape(1, -1)
        )
        
        energy = float(energy_arr[0])
        gradient = gradient_arr.flatten()
        
        # 恢复原始形状
        if x.ndim == 2:
            gradient = gradient.reshape(x.shape)
        
        return energy, gradient

    def predict_gradient(self, x: np.ndarray) -> np.ndarray:
        """
        预测梯度（实现基类抽象方法）

        Args:
            x: 坐标 (dim,) 或 (n_atoms, 3)

        Returns:
            gradient: 预测梯度 (与输入形状相同)
        """
        _, gradient = self.predict_energy_gradient(x)
        return gradient

    def clear_data(self) -> None:
        """清除所有训练数据"""
        super().clear_data()
        # 重置归一化参数
        self.input_mean = None
        self.input_std = None
        self.energy_mean = None
        self.energy_std = None
        self.gradient_mean = None
        self.gradient_std = None
        self.is_trained = False

    def acquisition_function(self, x: np.ndarray, y_min: float = None) -> float:
        """
        EI 采集函数（NN 版本）

        使用 dropout 不确定性作为探索激励

        Args:
            x: 坐标
            y_min: 当前最优能量

        Returns:
            EI 值
        """
        from scipy.stats import norm

        if not self.is_trained:
            return 0.0
        
        if y_min is None and len(self.y_train) > 0:
            y_min = min(self.y_train)
        elif y_min is None:
            y_min = 0.0
        
        energy, variance = self.predict(x)
        std = np.sqrt(max(variance, 1e-10))
        
        if std < 1e-6:
            return max(0.0, y_min - energy)
        
        gamma = (y_min - energy) / std
        ei = (y_min - energy) * norm.cdf(gamma) + std * norm.pdf(gamma)
        
        return float(ei)

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
            bounds = getattr(self, 'bounds', None)

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

    def __repr__(self) -> str:
        n_points = self.n_training_points()
        return f"{self.name}(trained={self.is_trained}, n_points={n_points}, dim={self.dim})"
