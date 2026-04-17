# PyBerny-GPR 分子几何构型优化项目

**分子几何构型优化 - PyBerny (BFGS) 与 GPR 混合策略研究**

---

## 项目信息

- **项目维护者**: 刘喆 (Liu Zhe)
- **联系邮箱**: 3266048598@qq.com
- **研究日期**: 2026
- **项目类型**: 计算化学 + 人工智能（机器学习）

---

## 项目概述

本项目实现了基于 **PyBerny (BFGS)** 和 **GPR (高斯过程回归)** 的分子几何构型优化混合策略。

**核心思想**：
- **外层 (Outer Loop)**: 使用 PyBerny 进行真实量子化学计算，可靠优化并收集训练数据
- **内层 (Inner Loop)**: 使用 GPR 预测梯度，快速探索势能面
- **择优策略**: 从内外层结果中选择最优作为下一轮起点

---

## 项目结构

```
pyberny_gpr/
├── core/                   # 核心数据类
│   ├── molecule.py        # 分子结构、迭代历史类
│   └── calculator.py      # 量子化学计算接口
├── optimizers/             # 优化器实现
│   ├── base.py            # 优化器基类
│   ├── hybrid.py          # PyBerny+GPR 混合优化器 ⭐
│   ├── pyberny_optimizer.py  # PyBerny BFGS 优化器
│   └── pyberny_baseline.py   # 纯 PyBerny 基准优化器
├── models/                 # 机器学习模型
│   ├── gpr_base.py        # GPR 基类
│   └── energy_gradient_gpr.py  # 能量 - 梯度 GPR 模型 ⭐
├── visualization/          # 可视化模块
│   ├── structure3d.py     # 3D 分子结构可视化
│   └── plots.py           # 能量/梯度图表
├── utils/                  # 工具函数
│   ├── io_utils.py        # 输入输出工具
│   └── converters.py      # 坐标转换工具
├── config/                 # 配置文件
│   └── default_config.yaml # 默认配置
├── main.py                 # 主程序入口
├── run_comparison.py       # 对比运行脚本
├── draw_structure3D.py     # 3D 结构可视化脚本
├── requirements.txt        # 依赖列表
├── README.md               # 本文件
├── DESIGN.md               # 详细设计文档
└── USAGE.md                # 使用说明文档
```

---

## 两种优化流程

### 流程 1: 纯 PyBerny 基准方法

**用途**: 作为对比基准，验证混合方法的有效性

**特点**:
- 使用 berny 库实现完整的 BFGS 优化
- 一次性运行到底，中间不打断
- 跨步次继承 Hessian 近似状态
- **冗余内坐标系统**（比笛卡尔坐标更高效）
- Trust Region 动态调整
- 复合收敛判据（能量 + 梯度 + 位移）

**运行命令**:
```bash
python main.py --method pyberny --molecule ethanol
```

### 流程 2: PyBerny+GPR 混合策略 ⭐

**用途**: 使用 AI 方法加速收敛，同时保持优化可靠性

**特点**:
- 外层 PyBerny BFGS（真实计算，每轮独立运行）
- 内层 GPR 梯度预测（快速探索）
- 滑动窗口管理训练数据
- 择优策略选择最优起点

**运行命令**:
```bash
python main.py --method hybrid --molecule ethanol --perturb 0.1
```

---

## 核心设计

### 混合优化策略流程

```
┌─────────────────────────────────────────────────────────────┐
│                    PyBerny-GPR 混合优化流程                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  初始采样 → 外层 PyBerny(m 步) → 训练 GPR → 内层 GPR 探索 (n 步)  │
│      ↑                                           │          │
│      └────────── 择优选择 ←────────────────────────┘          │
│                         循环                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 关键参数配置

在 `config/default_config.yaml` 中调整：

```yaml
hybrid:
  outer_steps: 10           # 外层 PyBerny 步数（真实计算）
  inner_steps: 3            # 内层 GPR 探索步数（预测）
  
  # 择优策略
  selection_metric: "gradient"
  selection_weights:
    energy_weight: 0.3      # 能量权重
    gradient_weight: 0.7    # 梯度权重（推荐 > 能量权重）
  
  # 收敛判定
  convergence:
    threshold: 1.0e-4       # 梯度收敛阈值
    max_rounds: 50          # 最大优化轮数

gpr:
  n_init: 5                 # 初始采样点数
  max_training_points: 30   # 滑动窗口大小
  local_radius: 0.1         # 局部搜索半径 (Å)
```

---

## 安装与使用

### 安装依赖

```bash
cd /mnt/e/wsl_dir/pyberny_gpr
pip install -r requirements.txt
```

### 快速开始

```bash
# 1. 纯 PyBerny 基准方法
python main.py --method pyberny --molecule ethanol

# 2. PyBerny+GPR 混合优化（推荐）
python main.py --method hybrid --molecule ethanol --perturb 0.1

# 3. 对比实验
python run_comparison.py --smiles CCO --perturb 0.1
```

### 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--method` | 优化方法 (`pyberny` 或 `hybrid`) | `pyberny` |
| `--molecule` | 分子名称 | `ethanol` |
| `--smiles` | SMILES 字符串 | `None` |
| `--perturb` | 初始扰动强度 (Å) | `0.0` |
| `--seed` | 随机种子 | `42` |
| `--max-iter` | 最大迭代次数 | `300` |
| `--threshold` | 收敛阈值 | `5e-4` |

---

## 输出说明

### 输出目录结构

```
output/
├── {method}_YYYYMMDD_HHMMSS.json      # 优化历史
├── {method}_trajectory_*.xyz          # 优化轨迹
├── {method}_details_*.json            # 详细信息
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

### 3D 分子结构可视化

```bash
python draw_structure3D.py
```

---

## 技术栈

| 类别 | 工具/库 |
|------|--------|
| 量子化学 | PySCF, berny |
| 分子处理 | RDKit, ASE |
| 优化算法 | **berny (BFGS)** |
| 机器学习 | scikit-learn (GPR) |
| 可视化 | Matplotlib, py3Dmol |
| 配置管理 | PyYAML |

---

## PyBerny 说明

**PyBerny** 是基于 `berny` 库的 BFGS 优化器，具有以下特点：

### PyBerny vs L-BFGS

| 特性 | PyBerny (berny) | L-BFGS |
|------|----------------|--------|
| 算法 | **完整 BFGS** | 有限内存 BFGS |
| Hessian 更新 | 完整矩阵 | 仅保留最近 m 步 |
| 坐标系统 | **冗余内坐标** | 笛卡尔坐标 |
| 收敛速度 | 更快 | 较慢 |

**注意**：本项目使用 PyBerny (完整 BFGS)，不是 L-BFGS。

---

## 文档说明

| 文档 | 说明 |
|------|------|
| `README.md` | 项目总览（本文件） |
| `DESIGN.md` | 代码核心设计、参数说明、理论公式 |
| `USAGE.md` | 详细使用说明 |

---

## 项目维护

**维护者**: 刘喆 (Liu Zhe)  
**邮箱**: 3266048598@qq.com  
**项目目录**: `/mnt/e/wsl_dir/pyberny_gpr/`

---

*最后更新：2026 年*
