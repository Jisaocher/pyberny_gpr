# PyBerny-GPR 混合优化项目 - 使用文档

## 项目概述

本项目实现了基于 **PyBerny (BFGS)** 和 **AI 代理模型** 的分子几何构型优化混合策略。

### 核心思想

| 组件 | 说明 |
|------|------|
| **外层优化** | PyBerny 真实量子化学计算，可靠优化 + 收集训练数据 |
| **内层探索** | AI 模型预测梯度，快速探索势能面 |
| **择优策略** | 从内外层结果中选择最优作为下一轮起点 |

### 支持方法

| 方法 | 说明 | 推荐场景 |
|------|------|---------|
| `pyberny` | 纯 PyBerny 基准（完整 BFGS） | 对比基准、小分子 |
| `hybrid` | PyBerny+AI 混合 ⭐ | 大分子、加速收敛 |

### 支持的 AI 方法（仅 hybrid 模式）

| AI 方法 | 说明 | 状态 |
|--------|------|------|
| `gpr` | 梯度预测高斯过程回归 | ✅ 已实现（默认） |
| `neural_network` | 神经网络 | 🔲 可扩展 |

---

## PyBerny 说明

**PyBerny** 基于 `berny` 库的完整 BFGS 实现：

### PyBerny vs L-BFGS

| 特性 | PyBerny | L-BFGS |
|------|---------|--------|
| 算法 | **完整 BFGS** | 有限内存 BFGS |
| 坐标系统 | **冗余内坐标** | 笛卡尔坐标 |
| 收敛速度 | 更快 | 较慢 |

**注意**: 本项目使用 PyBerny (完整 BFGS)，不是 L-BFGS。

---

## 安装依赖

### 1. 安装 pyberny

```bash
pip install -U pyberny
```

### 2. 安装项目依赖

```bash
cd /mnt/e/wsl_dir/pyberny_gpr
pip install -r requirements.txt
```

### 依赖说明

| 库 | 用途 | 最低版本 |
|---|------|---------|
| `pyscf` | 量子化学计算 | 2.0.0 |
| `rdkit` | 分子结构生成 | 2022.0.0 |
| `berny` | BFGS 优化器 | - |
| `scikit-learn` | GPR 模型 | 1.0.0 |
| `matplotlib` | 可视化 | 3.5.0 |
| `numpy` | 数值计算 | 1.20.0 |
| `pyyaml` | 配置解析 | 6.0 |
| `ase` | 分子结构处理 | 3.22.0 |
| `zhplot` | 中文字体支持 | 0.1.0 |
| `py3Dmol` | 3D 分子可视化 | 2.0.0 |

---

## 快速开始

### 方法 1: 纯 PyBerny 基准

```bash
# 乙醇分子，无扰动
python main.py --method pyberny --molecule ethanol

# 添加扰动
python main.py --method pyberny --molecule ethanol --perturb 0.1
```

### 方法 2: PyBerny+AI 混合（推荐）⭐

```bash
# 默认配置（使用 config 中的默认 AI 方法：gpr）
python main.py --method hybrid --molecule ethanol --perturb 0.1

# 显式指定 AI 方法为 GPR
python main.py --method hybrid --molecule ethanol --perturb 0.1 --ai_method gpr

# 自定义参数
python main.py --method hybrid --molecule ethanol --perturb 0.1 --max-iter 200
```

### 方法 3: 从 XYZ 文件读取初始构型

```bash
# 使用 XYZ 文件作为初始构型（两个参数必须同时指定）
python main.py --xyz_path ./config/initial_xyz/CCCC_initial.xyz --xyz_name my_CCCC --method pyberny

# 混合方法优化
python main.py --xyz_path ./config/initial_xyz/CCCC_initial.xyz --xyz_name my_CCCC --method hybrid --ai_method gpr
```

---

## 命令行参数

### main.py 参数

| 参数 | 说明 | 默认值 | 必需 |
|------|------|--------|------|
| `--method` | 优化方法 (`pyberny` / `hybrid`) | `pyberny` | 否 |
| `--molecule` | 分子名称 | `ethanol` | 否 |
| `--smiles` | SMILES 字符串（覆盖 `--molecule`） | `None` | 否 |
| `--perturb` | 初始扰动强度 (Å) | `0.0` | 否 |
| `--seed` | 随机种子 | `24` | 否 |
| `--config` | 配置文件路径 | `None` | 否 |
| `--output` | 输出目录 | `None` | 否 |
| `--max-iter` | 最大迭代次数 | `300` | 否 |
| `--threshold` | 收敛阈值 | `1.0e-4` | 否 |
| `--xyz_path` | .xyz 文件路径 | `None` | 否 |
| `--xyz_name` | 输出目录命名 | `None` | 否 |
| `--ai_method` | AI 方法（仅 hybrid 模式） | `gpr` | 否 |

**注意**:
- `--ai_method` 仅在 `--method hybrid` 时使用
- 若未指定 `--ai_method`，则使用 config 中的默认值（`hybrid.ai_method`）
- `--xyz_path` 和 `--xyz_name` 必须同时设置或同时为空，否则将抛出异常

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
  smiles: "CCO"              # SMILES 字符串
  seed: 24                   # 随机种子
  perturb: 0.1               # 初始扰动 (Å)
  xyz_path: null             # .xyz 文件路径（从文件读取时使用）
  xyz_name: null             # 输出目录命名（与 xyz_path 同时使用）
```

**注意**:
- 使用 `xyz_path` 和 `xyz_name` 时，`smiles`、`seed`、`perturb` 参数将被忽略
- `xyz_path` 和 `xyz_name` 必须同时设置或同时为空

### 计算方法

```yaml
calculation:
  basis: "cc-pvdz"           # 基组
  method: "RHF"              # 量子化学方法
  unit: "angstrom"           # 坐标单位
```

### 优化器设置

```yaml
optimizer:
  max_iterations: 300        # 最大迭代次数
  convergence_threshold: 1.0e-4  # 梯度收敛阈值
  verbose: true              # 详细输出
```

### 混合策略（核心）

```yaml
hybrid:
  # AI 方法选择（支持多种 AI 方法）
  ai_method: "gpr"           # AI 方法类型：gpr, 可扩展其他方法
                            # 仅当 --method 为 hybrid 时使用
                            # 若命令行未指定 --ai_method，则使用此默认值

  # 内外层步数
  outer_steps: 10            # 外层 PyBerny 步数（第 1 轮：15 步 = n_init + outer_steps）
  inner_steps: 5             # 内层 AI 探索步数

  # 内层优化配置（自研梯度下降法）
  inner_opt:
    gtol: 1.0e-4             # 梯度收敛阈值
    base_step_size: 0.05     # 基础步长（原子单位）
    max_step_size: 0.1       # 最大步长限制
    min_step_size: 1.0e-4    # 最小步长限制
    adaptive_step: true      # 是否启用自适应步长
    adaptive_factor: 100.0   # 自适应步长系数
    disp: false              # 是否打印收敛信息

  # 验证策略
  validate_every: 0          # 验证频率：0=仅验证最后一点
  prediction_error_threshold: 1.0e-3  # 预测误差阈值

  # 择优策略
  selection_metric: "gradient"  # 选择标准：gradient(梯度)/energy(能量)
  verify_local_minimum: false   # 验证局部极小值

  # 收敛判定
  convergence:
    threshold: 1.0e-4        # 梯度收敛阈值
    max_rounds: 50           # 最大优化轮数
    max_no_improvement: 50   # 无改进早停轮数
    no_improvement_threshold: 1.0e-6  # 无改进判定阈值
```

### AI 模型设置（GPR）

```yaml
gpr:
  n_init: 5                  # 初始采样点数（融合到第 1 轮外层）
  type: "gradient_predicting"  # AI 方法类型
  max_training_points: 10    # 滑动窗口大小
  local_radius: 0.1          # 局部搜索半径 (Å)
  noise_variance: 1.0e-4     # 噪声方差
  kernel_type: "matern52"    # 核函数类型
  xi: 0.1                    # EI 采集函数探索参数
  lambda_grad: 0.1           # 梯度惩罚权重
```

### PyBerny 配置（基准方法 + 混合方法外层）

```yaml
berny:
  maxsteps: 500              # 最大步数
  trust: 0.3                 # 信任半径 (Å)
  energy_threshold: 1e-6     # 能量收敛阈值 (Hartree)
  gradient_threshold: 1e-4   # 梯度收敛阈值 (Hartree/Å)
  displacement_threshold: 1e-3  # 位移收敛阈值 (Å)
  debug: false               # 调试模式
```

**注意**: 混合方法的外层优化直接使用 `berny` 配置，确保与基准方法参数一致。

---

## 输出说明

### 输出目录结构

```
output/
├── {smiles}_{perturb}/           # 按分子和扰动水平组织
│
│   ├── pyberny/                  # 纯 PyBerny 基准方法结果
│   │   ├── pyberny_YYYYMMDD_HHMMSS.json
│   │   ├── pyberny_trajectory_*.xyz
│   │   ├── pyberny_details_*.json
│   │   ├── plots/
│   │   │   ├── pyberny_energy.png
│   │   │   ├── pyberny_gradient.png
│   │   │   └── pyberny_combined.png
│   │   └── structures/
│   │       ├── pyberny_initial.xyz
│   │       ├── pyberny_initial.html
│   │       ├── pyberny_final.xyz
│   │       └── pyberny_final.html
│
│   ├── hybrid_gpr/               # 混合策略（GPR AI 方法）
│   │   ├── hybrid_gpr_YYYYMMDD_HHMMSS.json
│   │   ├── hybrid_gpr_trajectory_*.xyz
│   │   ├── hybrid_gpr_details_*.json
│   │   ├── plots/
│   │   │   ├── hybrid_gpr_energy.png
│   │   │   ├── hybrid_gpr_gradient.png
│   │   │   └── hybrid_gpr_combined.png
│   │   └── structures/
│   │       ├── hybrid_gpr_initial.xyz
│   │       ├── hybrid_gpr_initial.html
│   │       ├── hybrid_gpr_final.xyz
│   │       └── hybrid_gpr_final.html
│
│   └── hybrid_{ai_method}/       # 混合策略（其他 AI 方法，可扩展）
│       └── ...
```

**注意**: 混合策略的输出目录会根据 `--ai_method` 参数自动添加后缀，如 `hybrid_gpr`、`hybrid_nn` 等。

### 文件格式说明

| 文件类型 | 说明 |
|---------|------|
| `*.json` | 优化历史（能量、梯度、坐标、时间戳） |
| `*_trajectory_*.xyz` | 优化轨迹（含坐标和梯度信息） |
| `*_details_*.json` | 详细迭代信息（每个原子的梯度矩阵） |
| `structures/*.xyz` | 初始/最终分子结构 |
| `structures/*.html` | 3D 交互式分子结构（py3Dmol） |
| `plots/*.png` | 能量/梯度收敛曲线图 |

---

## 3D 分子结构可视化

```bash
python draw_structure3D.py
```

**功能**:
- 批量生成 HTML 格式 3D 分子结构
- 球棍模型，支持旋转/缩放/平移
- 自动查找 `output/structures` 目录
- 支持中文标签

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
print(f"总迭代次数：{len(history.iterations)}")
```

---

## 配置参数详解

### 核心参数

| 参数 | 含义 | 默认值 | 建议值 |
|------|------|--------|--------|
| `hybrid.ai_method` | AI 方法类型 | `gpr` | `gpr` |
| `hybrid.outer_steps` | 外层步数 | 10 | 10-15 |
| `hybrid.inner_steps` | 内层步数 | 5 | 3-5 |
| `gpr.n_init` | 初始采样点数 | 5 | 5-10 |
| `gpr.max_training_points` | 最大训练点数 | 10 | 10-20 |

**注意**: 第 1 轮外层迭代使用 `n_init + outer_steps` 步（默认 15 步），后续轮次使用 `outer_steps` 步（默认 10 步）。

### 择优权重

| 参数 | 含义 | 默认值 | 建议值 |
|------|------|--------|--------|
| `energy_weight` | 能量权重 | 0.3 | 0.2-0.5 |
| `gradient_weight` | 梯度权重 | 0.7 | 0.5-0.8 |

**注意**: 分子优化目标是梯度为零，推荐 `gradient_weight > energy_weight`

### 内层优化参数

| 参数 | 含义 | 默认值 | 说明 |
|------|------|--------|------|
| `inner_opt.gtol` | 梯度收敛阈值 | 1.0e-4 | \|g\| < gtol 则认为收敛 |
| `inner_opt.base_step_size` | 基础步长 | 0.05 | 原子单位（约 0.5 pm） |
| `inner_opt.adaptive_step` | 自适应步长 | true | 梯度越大步长越大 |
| `inner_opt.adaptive_factor` | 自适应系数 | 100.0 | 控制步长增长速率 |

### 收敛判定

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `convergence.threshold` | 梯度收敛阈值 | 1e-4 |
| `convergence.max_rounds` | 最大优化轮数 | 50 |
| `convergence.max_no_improvement` | 无改进早停轮数 | 50 |
| `convergence.no_improvement_threshold` | 无改进判定阈值 | 1e-6 |

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

**可能原因及解决方案**:

| 原因 | 解决方案 |
|------|---------|
| 迭代次数不足 | 增加 `max_iterations` 或 `max_rounds` |
| 收敛阈值过严 | 放宽 `convergence_threshold` 至 1e-3 |
| 初始扰动过大 | 减小 `perturb` 至 0.05-0.1 |
| GPR 训练数据不足 | 增加 `gpr.n_init` 至 10 |
| 局部搜索范围过大 | 减小 `gpr.local_radius` 至 0.05 |

### Q2: GPR 训练失败？

**可能原因及解决方案**:

| 原因 | 解决方案 |
|------|---------|
| 训练数据太少 | 增加 `n_init` 或 `max_training_points` |
| 噪声方差不合适 | 调整 `noise_variance` 至 1e-3 或 1e-5 |
| 核函数参数不合适 | 尝试不同 `kernel_type`（如 `rbf`） |

### Q3: 如何加速计算？

**建议**:

| 方法 | 说明 |
|------|------|
| 使用较小基组 | 如 `sto-3g` 测试，确认后再用 `cc-pvdz` |
| 减少迭代次数 | 减少 `max_iterations` 或 `inner_steps` |
| 增加 AI 探索比例 | 增加 `inner_steps`，减少 `outer_steps` |
| 关闭详细输出 | 设置 `verbose: false` |

### Q4: 如何选择 AI 方法？

| 场景 | 推荐 AI 方法 |
|------|------------|
| 小分子（<20 原子） | `gpr`（默认） |
| 中等分子（20-50 原子） | `gpr` + 增加 `max_training_points` |
| 大分子（>50 原子） | 考虑扩展 `neural_network` |
| 高精度需求 | `gpr` + 小 `local_radius` |

---

## 性能基准

乙醇分子（RHF/cc-pVDZ）典型结果：

| 方法 | 初始能量 (Hartree) | 最优能量 (Hartree) | 迭代次数 | PySCF 调用 |
|------|-------------------|-------------------|---------|-----------|
| PyBerny (无扰动) | -154.0803 | -154.0927 | ~20 | ~20 |
| PyBerny (扰动 0.5) | -153.0956 | -153.9109 | ~30 | ~30 |
| Hybrid-GPR (扰动 0.5) | -153.0956 | -153.9109 | ~25 | ~20 |

**说明**: Hybrid-GPR 通过内层 AI 探索减少真实 PySCF 调用次数，从而节省计算成本。

---

## 输出文件示例

### JSON 输出结构

```json
{
  "metadata": {
    "method": "hybrid",
    "molecule": "ethanol",
    "smiles": "CCO",
    "n_atoms": 9,
    "config": {...}
  },
  "iterations": [
    {
      "iteration": 1,
      "energy": -154.0803123456,
      "gradient_norm": 0.123456,
      "coords": [...],
      "gradient": [...],
      "displacement": [...],
      "round_num": 1,
      "stage": "outer"
    }
  ],
  "statistics": {
    "total_iterations": 25,
    "initial_energy": -154.0803123456,
    "final_energy": -154.0927654321,
    "best_energy": -154.0927654321,
    "energy_improvement": 0.0124530865,
    "initial_gradient_norm": 0.123456,
    "final_gradient_norm": 0.000098,
    "best_gradient_norm": 0.000098,
    "converged": true,
    "computation_time": 123.45
  }
}
```

---

## 参考文献

1. Liu, D. C., & Nocedal, J. (1989). On the limited memory BFGS method. *Mathematical Programming*, 45(1), 503-528.
2. Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.
3. pyberny: https://github.com/jhrmnn/pyberny
4. PySCF: https://www.pyscf.org/
5. Sun, Q. et al. (2018). Recent developments in the PySCF program package. *The Journal of Chemical Physics*, 153(2), 024109.

---

*最后更新：2026 年*
