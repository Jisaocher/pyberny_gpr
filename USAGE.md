# PyBerny-GPR 混合优化项目使用文档

## 项目概述

本项目实现了基于 **PyBerny (BFGS)** 和 **GPR (高斯过程回归)** 的分子几何构型优化混合策略。

**核心思想**：
- **外层**: PyBerny 真实量子化学计算
- **内层**: GPR 梯度预测探索
- **择优**: 选择最优起点

**支持方法**：
- `pyberny`: 纯 PyBerny 基准（完整 BFGS）
- `hybrid`: PyBerny+GPR 混合（推荐）

---

## PyBerny 说明

**PyBerny** 基于 `berny` 库的完整 BFGS 实现：

### PyBerny vs L-BFGS

| 特性 | PyBerny | L-BFGS |
|------|---------|--------|
| 算法 | 完整 BFGS | 有限内存 BFGS |
| 坐标系统 | **冗余内坐标** | 笛卡尔坐标 |
| 收敛速度 | 更快 | 较慢 |

**注意**: 本项目使用 PyBerny (完整 BFGS)，不是 L-BFGS。

---

## 安装依赖

```bash
cd /mnt/e/wsl_dir/pyberny_gpr
pip install -r requirements.txt
```

### 依赖说明

| 库 | 用途 |
|---|------|
| `pyscf` | 量子化学计算 |
| `rdkit` | 分子结构生成 |
| `berny` | BFGS 优化器 |
| `scikit-learn` | GPR 模型 |
| `matplotlib` | 可视化 |
| `pyyaml` | 配置解析 |
| `py3Dmol` | 3D 可视化 |

---

## 快速开始

### 方法 1: 纯 PyBerny 基准

```bash
# 乙醇分子，无扰动
python main.py --method pyberny --molecule ethanol

# 添加扰动
python main.py --method pyberny --molecule ethanol --perturb 0.1
```

### 方法 2: PyBerny+GPR 混合（推荐）⭐

```bash
# 默认配置
python main.py --method hybrid --molecule ethanol --perturb 0.1

# 自定义参数
python main.py --method hybrid --molecule ethanol --perturb 0.1 --max-iter 200
```

### 方法 3: 对比实验

```bash
python run_comparison.py --smiles CCO --perturb 0.1
```

---

## 命令行参数

### main.py 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--method` | `pyberny` 或 `hybrid` | `pyberny` |
| `--molecule` | 分子名称 | `ethanol` |
| `--smiles` | SMILES 字符串 | `None` |
| `--perturb` | 初始扰动 (Å) | `0.0` |
| `--seed` | 随机种子 | `24` |
| `--max-iter` | 最大迭代次数 | `300` |
| `--threshold` | 收敛阈值 | `5e-4` |

### 支持的分子

| 名称 | SMILES | 原子数 |
|------|--------|--------|
| `water` | O | 3 |
| `methane` | C | 5 |
| `ethanol` | CCO | 9 |

---

## 配置文件

编辑 `config/default_config.yaml`:

### 分子设置
```yaml
molecule:
  smiles: "CCO"
  seed: 24
  perturb: 0.1
```

### 计算方法
```yaml
calculation:
  basis: "cc-pvdz"
  method: "RHF"
  unit: "angstrom"
```

### 优化器设置
```yaml
optimizer:
  max_iterations: 300
  convergence_threshold: 1.0e-4
  verbose: true
```

### 混合策略（核心）

```yaml
hybrid:
  outer_steps: 10            # 外层 PyBerny 步数（第 1 轮：15 步 = n_init + outer_steps）
  inner_steps: 5             # 内层 GPR 探索步数

  # 融合设计说明：
  # - 第 1 轮：外层使用 n_init + outer_steps 步，融合初始采样
  # - 后续轮：外层使用 outer_steps 步
  # - 不再需要独立的"初始采样"阶段

  inner_opt:
    gtol: 1.0e-4
    base_step_size: 0.05
    adaptive_step: true

  selection_weights:
    energy_weight: 0.3       # 能量权重
    gradient_weight: 0.7     # 梯度权重

  convergence:
    threshold: 1.0e-4
    max_rounds: 50
    max_no_improvement: 50
```

### GPR 设置

```yaml
gpr:
  n_init: 5                  # 初始采样点数（融合到第 1 轮外层）
  max_training_points: 10    # 滑动窗口大小
  local_radius: 0.1          # 局部搜索半径 (Å)
  noise_variance: 1.0e-4     # 噪声方差
```

### PyBerny 配置（基准方法 + 混合方法外层）

```yaml
berny:
  maxsteps: 500              # 最大步数
  trust: 0.3                 # 信任半径 (Å)
  energy_threshold: 1e-6     # 能量收敛阈值
  gradient_threshold: 1e-4   # 梯度收敛阈值
  displacement_threshold: 1e-3  # 位移收敛阈值
  debug: false
```

**注意**：混合方法的外层优化直接使用 `berny` 配置，确保与基准方法参数一致。

---

## 输出说明

### 输出目录结构

```
output/
├── {method}_YYYYMMDD_HHMMSS.json
├── {method}_trajectory_*.xyz
├── {method}_details_*.json
├── plots/
│   ├── {method}_energy.png
│   ├── {method}_gradient.png
│   └── {method}_combined.png
└── structures/
    ├── {method}_initial.xyz
    ├── {method}_initial.html
    ├── {method}_final.xyz
    └── {method}_final.html
```

---

## 3D 分子结构可视化

```bash
python draw_structure3D.py
```

**功能**:
- 批量生成 HTML 格式 3D 分子结构
- 球棍模型，支持旋转/缩放/平移
- 自动查找 `output/structures` 目录

---

## Python API

```python
from core.molecule import Molecule
from core.calculator import QuantumCalculator
from optimizers.hybrid import HybridOptimizer
import yaml

# 1. 创建分子
mol = Molecule.from_smiles("CCO", seed=24, perturb_strength=0.1)

# 2. 加载配置
with open("config/default_config.yaml") as f:
    config = yaml.safe_load(f)

# 3. 创建计算器
calculator = QuantumCalculator(
    basis="cc-pvdz",
    method="RHF",
    unit="angstrom"
)

# 4. 创建优化器
optimizer = HybridOptimizer(config)

# 5. 执行优化
history = optimizer.optimize(mol, calculator)

# 6. 查看结果
print(f"收敛：{history.converged}")
print(f"最优能量：{history.best_iteration.energy:.8f}")
print(f"最优梯度：{history.best_iteration.gradient_norm:.6f}")
```

---

## 配置参数详解

### 核心参数

| 参数 | 含义 | 默认值 | 建议值 |
|------|------|--------|--------|
| `hybrid.outer_steps` | 外层步数 | 10 | 10-15 |
| `hybrid.inner_steps` | 内层步数 | 5 | 3-5 |
| `gpr.n_init` | 初始采样点数 | 5 | 5-10 |
| `gpr.max_training_points` | 最大训练点数 | 10 | 10-20 |

**注意**：第 1 轮外层迭代使用 `n_init + outer_steps` 步（默认 15 步），后续轮次使用 `outer_steps` 步（默认 10 步）。

### 择优权重

| 参数 | 含义 | 默认值 | 建议值 |
|------|------|--------|--------|
| `energy_weight` | 能量权重 | 0.3 | 0.2-0.5 |
| `gradient_weight` | 梯度权重 | 0.7 | 0.5-0.8 |

**注意**: 分子优化目标是梯度为零，推荐 `gradient_weight > energy_weight`

### 收敛判定

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `convergence_threshold` | 梯度收敛阈值 | 1e-4 |
| `max_rounds` | 最大优化轮数 | 50 |
| `max_no_improvement` | 无改进早停轮数 | 50 |

### PyBerny 参数（基准方法 + 混合方法外层）

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `berny.maxsteps` | 最大步数 | 500 |
| `berny.trust` | 信任半径 | 0.3 Å |
| `berny.gradient_threshold` | 梯度收敛阈值 | 1e-4 |
| `berny.energy_threshold` | 能量收敛阈值 | 1e-6 |
| `berny.displacement_threshold` | 位移收敛阈值 | 1e-3 |

---

## 常见问题

### Q1: 优化不收敛？

**解决**:
- 增加 `max_iterations` 或 `max_rounds`
- 放宽 `convergence_threshold`
- 增加 `gpr.n_init`
- 减小 `gpr.local_radius`

### Q2: GPR 训练失败？

**解决**:
- 增加 `n_init` 或 `max_training_points`
- 调整 `noise_variance`

### Q3: 如何加速计算？

**建议**:
- 使用较小基组（如 `sto-3g`）测试
- 减少 `max_iterations`
- 减少 `inner_steps`

---

## 性能基准

乙醇分子（RHF/cc-pVDZ）典型结果：

| 方法 | 初始能量 | 最优能量 | 迭代次数 |
|------|---------|---------|---------|
| PyBerny (无扰动) | -154.0803 | -154.0927 | ~20 |
| PyBerny (扰动 0.5) | -153.0956 | -153.9109 | ~30 |

---

## 参考文献

1. Liu, D. C., & Nocedal, J. (1989). On the limited memory BFGS method.
2. Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian Processes for Machine Learning.
3. pyberny: https://github.com/jhrmnn/pyberny
4. PySCF: https://www.pyscf.org/

---

*最后更新：2026 年*
