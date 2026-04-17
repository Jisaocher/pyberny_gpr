# PyBerny-GPR 混合优化策略设计文档

## 1. 核心逻辑概述

### 1.1 抽象核心逻辑

**核心思想**: 将 PyBerny (BFGS) 与 GPR 相结合，实现可靠且高效的分子几何构型优化。

```
┌─────────────────────────────────────────────────────────────────┐
│                    PyBerny-GPR 混合优化流程                       │
├─────────────────────────────────────────────────────────────────┤
│  初始采样 → 外层 PyBerny(m 步) → 训练 GPR → 内层 GPR 探索 (n 步)    │
│      ↑                                           │              │
│      └────────── 择优选择 ←────────────────────────┘              │
│                         循环                                       │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 关键设计思想

1. **外层可靠优化**: PyBerny BFGS 使用真实量子化学计算
2. **内层快速探索**: GPR 预测梯度 + 自研梯度下降法
3. **滑动窗口**: 只保留最近外层迭代数据
4. **择优策略**: 综合考虑能量和梯度

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
│   └── gpr_base.py            # GPR 抽象基类
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
│  - pyberny_optimizer │  (外层 BFGS)
│  - gpr_model         │  (内层预测)
│  - calculator        │
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
while round_num < max_rounds:
    # 步骤 1: 外层 PyBerny BFGS (m 步)
    outer_result = self._run_outer_bfgs(coords)
    
    # 步骤 2: 训练 GPR 模型
    self._train_gpr()
    
    # 步骤 3: 内层 GPR 探索 (n 步)
    inner_result = self._run_inner_exploration(outer_result)
    
    # 步骤 4: 择优选择
    best_candidate = self._select_best_candidate(outer_result, inner_result)
    
    # 步骤 5: 更新起点
    coords = best_candidate['coords']
    
    # 步骤 6: 收敛检查
    if gradient_norm < threshold:
        break
```

---

## 3. 梯度范数

### 3.1 数学定义

对于 N 个原子的分子：

$$\mathbf{g} = \nabla E(\mathbf{R}) \in \mathbb{R}^{3N}$$

$$\|\mathbf{g}\| = \sqrt{\sum_{i=1}^{3N} g_i^2}$$

### 3.2 收敛标准

```yaml
optimizer:
  convergence_threshold: 1.0e-4  # 梯度范数收敛阈值
```

**收敛判定**: $\|\mathbf{g}\| < 10^{-4} \Rightarrow$ 优化收敛

---

## 4. 理论清单

### 4.1 高斯过程回归 (GPR)

$$f(\mathbf{x}) \sim \mathcal{GP}(m(\mathbf{x}), k(\mathbf{x}, \mathbf{x}'))$$

**核函数** (Matérn 5/2):

$$k(r) = \sigma^2 \left(1 + \frac{\sqrt{5}r}{\ell} + \frac{5r^2}{3\ell^2}\right) \exp\left(-\frac{\sqrt{5}r}{\ell}\right)$$

### 4.2 BFGS 算法

**BFGS 更新公式**:

$$\mathbf{B}_{k+1} = \mathbf{B}_k + \frac{\mathbf{y}_k \mathbf{y}_k^T}{\mathbf{y}_k^T \mathbf{s}_k} - \frac{\mathbf{B}_k \mathbf{s}_k \mathbf{s}_k^T \mathbf{B}_k}{\mathbf{s}_k^T \mathbf{B}_k \mathbf{s}_k}$$

**PyBerny 特点**:
- 完整 BFGS 算法
- 冗余内坐标系统
- Trust Region 动态调整
- 复合收敛判据

### 4.3 量子化学计算

```yaml
calculation:
  method: "RHF"      # 限制 Hartree-Fock
  basis: "cc-pvdz"   # 相关一致性极化双ζ基组
```

---

## 5. 变量说明

### 5.1 配置参数

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `hybrid.outer_steps` | 外层 PyBerny 步数 | 10 |
| `hybrid.inner_steps` | 内层 GPR 探索步数 | 3 |
| `selection_weights.energy_weight` | 能量权重 | 0.3 |
| `selection_weights.gradient_weight` | 梯度权重 | 0.7 |
| `gpr.max_training_points` | 最大训练点数 | 30 |

### 5.2 核心变量

| 变量 | 维度 | 含义 |
|------|------|------|
| `coords` | $(3N,)$ | 分子坐标 |
| `gradient` | $(3N,)$ | 能量梯度 |
| `gradient_norm` | 标量 | $\|\nabla E\|_2$ |

---

## 6. 代码调用流程

```
HybridOptimizer.optimize()
├── _initial_sampling()          # 初始 PyBerny 采样
├── _run_outer_bfgs()            # 外层 BFGS
├── _train_gpr()                 # 训练 GPR
├── _run_inner_exploration()     # 内层探索
├── _select_best_candidate()     # 择优选择
└── _record_round_history()      # 记录历史
```

---

## 7. 总结

### 创新点
1. 外层 + 内层分离设计
2. 滑动窗口数据管理
3. 梯度预测直接符合优化目标
4. 自适应步长梯度下降

### 适用场景
- 分子几何优化
- 势能面搜索
- 需平衡计算成本与质量的场景

---

*最后更新：2026 年*
