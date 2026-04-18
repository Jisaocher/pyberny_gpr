# PyBerny-GPR 混合优化策略设计文档

## 1. 核心逻辑概述

### 1.1 抽象核心逻辑

**核心思想**: 将 PyBerny (BFGS) 与 AI 代理模型相结合，实现可靠且高效的分子几何构型优化。

```
┌─────────────────────────────────────────────────────────────────┐
│                    PyBerny-AI 混合优化流程                        │
├─────────────────────────────────────────────────────────────────┤
│  第 1 轮：外层 PyBerny(n_init + outer_steps 步) → 训练 AI 模型      │
│         → 内层 AI 探索 (inner_steps 步) → 择优                    │
│                                                                 │
│  后续轮：外层 PyBerny(outer_steps 步) → 训练 AI 模型 → 内层探索    │
│         → 择优 → 循环                                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 关键设计思想

| 设计原则 | 说明 |
|----------|------|
| **外层可靠优化** | PyBerny BFGS 使用真实量子化学计算，确保优化可靠性 |
| **内层快速探索** | AI 模型预测梯度 + 自研梯度下降法，快速探索势能面 |
| **滑动窗口** | 只保留最近外层迭代数据，控制训练成本 |
| **择优策略** | 综合考虑能量和梯度，选择最优起点 |
| **配置同步** | 混合方法外层与基准方法使用相同配置，确保公平对比 |

### 1.3 支持的 AI 方法

| AI 方法 | 模型类型 | 预测目标 | 状态 |
|--------|---------|---------|------|
| `gpr` | 高斯过程回归 | 梯度 | ✅ 已实现 |
| `neural_network` | 神经网络 | 梯度 | 🔲 可扩展 |

---

## 2. 详细设计

### 2.1 混合优化器架构

```
pyberny_gpr/
├── optimizers/
│   ├── hybrid.py              # 混合优化器 ⭐
│   ├── pyberny_optimizer.py   # PyBerny BFGS（每轮独立）
│   ├── pyberny_baseline.py    # 纯 PyBerny 基准（完整 BFGS）
│   └── base.py                # 优化器基类
├── models/
│   ├── energy_gradient_gpr.py # 能量 - 梯度 GPR 模型 ⭐
│   └── gpr_base.py            # AI 模型抽象基类
├── core/
│   ├── molecule.py            # 分子数据结构
│   └── calculator.py          # 量子化学计算器
└── config/
    └── default_config.yaml
```

### 2.2 核心类关系图

```
┌──────────────────────┐
│  HybridOptimizer     │
│  - ai_method: str    │  # AI 方法类型
│  - ai_model          │  # AI 模型（动态选择）
│  - lbfgs_optimizer   │  # 外层 BFGS
│  - calculator        │  # 量子化学计算器
└──────────┬───────────┘
           │
           ├──────────────────┐
           ▼                  ▼
┌──────────────────┐  ┌──────────────────┐
│ PyBernyOptimizer │  │ EnergyGradientGPR│
│ (外层 BFGS)       │  │ (内层预测)       │
└──────────────────┘  └──────────────────┘
           │                  │
           └────────┬─────────┘
                    ▼
          ┌──────────────────┐
          │ QuantumCalculator│
          │ (PySCF 后端)      │
          └──────────────────┘
```

### 2.3 算法流程

```python
# 融合设计：去除独立初始采样，第 1 轮外层使用 n_init + outer_steps 步
while round_num < max_rounds:
    # 步骤 1: 外层 PyBerny BFGS（真实计算）
    # 第 1 轮：n_init + outer_steps 步（融合初始采样）
    # 后续轮：outer_steps 步
    outer_result = self._run_outer_bfgs(
        coords, 
        is_first_round=(round_num == 1)
    )

    # 步骤 2: 训练 AI 模型（使用滑动窗口管理数据）
    self._train_ai_model()

    # 步骤 3: 内层 AI 探索（预测梯度 + 梯度下降）
    inner_result = self._run_inner_exploration(outer_result)

    # 步骤 4: 择优选择（综合能量和梯度）
    best_candidate = self._select_best_candidate(
        outer_result, 
        inner_result
    )

    # 步骤 5: 更新起点
    coords = best_candidate['coords']

    # 步骤 6: 收敛检查
    if gradient_norm < threshold:
        break
```

---

## 3. 数学基础

### 3.1 分子几何优化问题

**优化目标**: 找到分子几何构型 $\mathbf{R}^*$，使得势能面 $E(\mathbf{R})$ 的梯度为零：

$$\nabla E(\mathbf{R}^*) = \mathbf{0}$$

**梯度范数**: 对于 $N$ 个原子的分子：

$$\mathbf{g} = \nabla E(\mathbf{R}) \in \mathbb{R}^{3N}$$

$$\|\mathbf{g}\| = \sqrt{\sum_{i=1}^{3N} g_i^2}$$

**收敛标准**: $\|\mathbf{g}\| < 10^{-4} \Rightarrow$ 优化收敛

---

### 3.2 BFGS 算法

**BFGS 更新公式**:

$$\mathbf{B}_{k+1} = \mathbf{B}_k + \frac{\mathbf{y}_k \mathbf{y}_k^T}{\mathbf{y}_k^T \mathbf{s}_k} - \frac{\mathbf{B}_k \mathbf{s}_k \mathbf{s}_k^T \mathbf{B}_k}{\mathbf{s}_k^T \mathbf{B}_k \mathbf{s}_k}$$

其中：
- $\mathbf{B}_k$: Hessian 近似矩阵
- $\mathbf{s}_k = \mathbf{R}_{k+1} - \mathbf{R}_k$: 位移
- $\mathbf{y}_k = \nabla E_{k+1} - \nabla E_k$: 梯度变化

**PyBerny 特点**:
| 特性 | 说明 |
|------|------|
| 算法 | 完整 BFGS（非 L-BFGS） |
| 坐标系统 | 冗余内坐标（比笛卡尔坐标更高效） |
| Hessian 更新 | 完整矩阵（非有限内存） |
| Trust Region | 动态调整信任半径 |
| 收敛判据 | 复合判据（能量 + 梯度 + 位移） |

---

### 3.3 高斯过程回归 (GPR)

**GPR 定义**:

$$f(\mathbf{x}) \sim \mathcal{GP}(m(\mathbf{x}), k(\mathbf{x}, \mathbf{x}'))$$

**核函数** (Matérn 5/2):

$$k(r) = \sigma^2 \left(1 + \frac{\sqrt{5}r}{\ell} + \frac{5r^2}{3\ell^2}\right) \exp\left(-\frac{\sqrt{5}r}{\ell}\right)$$

**梯度预测**:

$$\nabla f(\mathbf{x}^*) = \frac{\partial}{\partial \mathbf{x}^*} k(\mathbf{x}^*, \mathbf{X}) [K + \sigma_n^2 I]^{-1} \mathbf{y}$$

**GPR 参数**:
| 参数 | 含义 | 默认值 |
|------|------|--------|
| `n_init` | 初始采样点数 | 5 |
| `max_training_points` | 滑动窗口大小 | 10 |
| `local_radius` | 局部搜索半径 (Å) | 0.1 |
| `noise_variance` | 噪声方差 | 1e-4 |
| `kernel_type` | 核函数类型 | matern52 |

---

### 3.4 量子化学计算

**计算方法配置**:

```yaml
calculation:
  method: "RHF"      # 限制 Hartree-Fock
  basis: "cc-pvdz"   # 相关一致性极化双ζ基组
  unit: "angstrom"   # 坐标单位
```

**计算流程**:
1. 输入分子坐标
2. PySCF 执行自洽场 (SCF) 计算
3. 输出能量和解析梯度

---

## 4. 核心组件设计

### 4.1 HybridOptimizer（混合优化器）

**核心职责**:
- 管理外层/内层优化流程
- 动态选择 AI 方法
- 实现择优策略
- 收敛判定

**关键方法**:
| 方法 | 说明 |
|------|------|
| `optimize()` | 主优化流程 |
| `_run_outer_bfgs()` | 外层 BFGS 优化 |
| `_train_ai_model()` | 训练 AI 模型 |
| `_run_inner_exploration()` | 内层 AI 探索 |
| `_select_best_candidate()` | 择优选择 |
| `_initialize_ai_model()` | 初始化 AI 模型 |

### 4.2 AI 模型接口

**统一接口** (`gpr_base.py`):

```python
class BaseGPRModel(ABC):
    @abstractmethod
    def train(self, X, y, gradients) -> None:
        """训练模型"""
        pass
    
    @abstractmethod
    def predict(self, x) -> Tuple[float, float]:
        """预测能量（均值和方差）"""
        pass
    
    @abstractmethod
    def predict_gradient(self, x) -> np.ndarray:
        """预测梯度"""
        pass
    
    def add_data(self, x, energy, gradient) -> None:
        """添加训练数据"""
        pass
    
    def clear_data(self) -> None:
        """清除训练数据"""
        pass
```

### 4.3 滑动窗口数据管理

**两级筛选策略**:

```
1. 时间窗口筛选：
   只保留最近 max_outer_iterations 次外层迭代的数据

2. 数量上限筛选：
   在窗口内按梯度范数排序，最多保留 max_training_points 个点
   （梯度范数越小越优先，因为更接近收敛点）
```

**优势**:
- 控制训练成本（O(n³) → O(m³)，m << n）
- 保留最有价值的数据点
- 适应势能面局部特性

---

## 5. 择优策略

### 5.1 候选点来源

| 候选点 | 来源 | 验证状态 |
|--------|------|---------|
| `outer_final` | 外层 BFGS 终点 | ✅ 已验证（真实计算） |
| `inner_final` | 内层 AI 探索终点 | ✅ 已验证（最后一步真实计算） |

### 5.2 评分函数

$$\text{score} = w_E \cdot \Delta E_{\text{norm}} + w_g \cdot \Delta g_{\text{norm}}$$

其中：
- $\Delta E_{\text{norm}} = \frac{E - E_{\text{start}}}{|E_{\text{start}}|}$
- $\Delta g_{\text{norm}} = \frac{\|\mathbf{g}\| - \|\mathbf{g}_{\text{start}}\|}{\|\mathbf{g}_{\text{start}}\|}$
- $w_E = 0.3$（能量权重）
- $w_g = 0.7$（梯度权重）

**推荐**: $w_g > w_E$，因为分子几何优化的目标是梯度为零。

---

## 6. 收敛判定

### 6.1 主收敛条件

$$\|\mathbf{g}\| < \text{threshold} = 10^{-4}$$

### 6.2 早停条件

| 条件 | 说明 |
|------|------|
| `max_rounds` | 最大优化轮数（默认 50） |
| `max_no_improvement` | 连续无改进轮数（默认 50） |
| `no_improvement_threshold` | 无改进判定阈值（默认 1e-6） |

### 6.3 外层提前终止

当外层 BFGS 满足 scipy 收敛条件时，提前终止并跳过内层探索：
- 梯度收敛
- 步长过小
- 达到最大迭代次数

---

## 7. 配置参数详解

### 7.1 混合策略参数

| 参数 | 含义 | 默认值 | 建议范围 |
|------|------|--------|---------|
| `hybrid.ai_method` | AI 方法类型 | `gpr` | `gpr` |
| `hybrid.outer_steps` | 外层步数 | 10 | 10-15 |
| `hybrid.inner_steps` | 内层步数 | 5 | 3-5 |
| `hybrid.validate_every` | 验证频率 | 0 | 0（仅验证最后一点） |

### 7.2 内层优化参数

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `inner_opt.gtol` | 梯度收敛阈值 | 1e-4 |
| `inner_opt.base_step_size` | 基础步长 | 0.05 |
| `inner_opt.adaptive_step` | 自适应步长 | true |
| `inner_opt.adaptive_factor` | 自适应系数 | 100.0 |

**自适应步长公式**:
$$\text{step} = \text{base\_step} \times (0.1 + \|\mathbf{g}\| \times \text{adaptive\_factor})$$

### 7.3 择优权重

| 参数 | 含义 | 默认值 | 建议范围 |
|------|------|--------|---------|
| `energy_weight` | 能量权重 | 0.3 | 0.2-0.5 |
| `gradient_weight` | 梯度权重 | 0.7 | 0.5-0.8 |

---

## 8. 代码调用流程

```
main()
  └── run_optimization()
        └── HybridOptimizer.optimize()
              ├── _initialize_ai_model()
              │
              ├── [循环] while round_num < max_rounds:
              │     ├── _run_outer_bfgs()
              │     │     └── PyBernyOptimizer.run_fixed_steps()
              │     │           └── 量子化学计算 (PySCF)
              │     │
              │     ├── _train_ai_model()
              │     │     └── EnergyGradientGPR.train()
              │     │
              │     ├── _run_inner_exploration()
              │     │     └── AI 模型.predict_gradient()
              │     │
              │     └── _select_best_candidate()
              │
              └── 保存结果
```

---

## 9. 扩展 AI 方法指南

### 9.1 实现新的 AI 模型

1. 继承 `BaseGPRModel` 基类
2. 实现必需方法：
   - `train(X, y, gradients)`
   - `predict(x)`
   - `predict_gradient(x)`
   - `add_data(x, energy, gradient)`
   - `clear_data()`

### 9.2 注册新的 AI 方法

1. 在 `config/default_config.yaml` 中添加新的 `ai_method` 选项
2. 在 `main.py` 的 `--ai_method` 参数中添加 `choices`
3. 在 `HybridOptimizer._initialize_ai_model()` 中添加初始化逻辑

### 9.3 示例：神经网络

```python
# models/neural_network.py
from models.gpr_base import BaseGPRModel

class NeuralNetworkModel(BaseGPRModel):
    def __init__(self, config, dim):
        super().__init__(config)
        self.name = "NeuralNetwork"
        # 初始化神经网络...
    
    def train(self, X, y, gradients):
        # 训练神经网络...
        pass
    
    def predict_gradient(self, x):
        # 预测梯度...
        pass
```

---

## 10. 创新点总结

| 创新点 | 说明 |
|--------|------|
| **融合设计** | 去除独立初始采样，第 1 轮外层使用 `n_init + outer_steps` 步 |
| **配置同步** | 混合方法外层与基准方法使用完全相同的 `berny` 配置 |
| **外层 + 内层分离** | 外层可靠优化，内层快速探索 |
| **滑动窗口** | 只保留最近外层迭代数据，控制训练成本 |
| **梯度预测** | AI 模型直接预测梯度，符合优化目标 |
| **自适应步长** | 梯度越大步长越大，适合初期快速探索 |
| **多 AI 方法支持** | 可扩展神经网络等其他 AI 方法 |

---

## 11. 适用场景

| 场景 | 推荐方法 | 说明 |
|------|---------|------|
| 小分子快速优化 | `pyberny` | 直接 BFGS，无需 AI 代理 |
| 大分子优化 | `hybrid` | AI 加速，节省计算成本 |
| 高精度需求 | `hybrid` + 小 `local_radius` | 精细探索 |
| 快速筛选 | `hybrid` + 大 `inner_steps` | 更多 AI 探索 |

---

*最后更新：2026 年*
