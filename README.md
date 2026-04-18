# PyBerny-GPR 分子几何构型优化项目

**分子几何构型优化 - PyBerny (BFGS) 与 AI 代理模型混合策略研究**

---

## 项目信息

| 项目 | 信息 |
|------|------|
| **项目维护者** | 刘喆 (Liu Zhe) |
| **联系邮箱** | 3266048598@qq.com |
| **研究日期** | 2026 |
| **项目类型** | 计算化学 + 人工智能（机器学习） |
| **项目路径** | `./pyberny_gpr/` |

---

## 项目概述

本项目实现了基于 **PyBerny (BFGS)** 和 **AI 代理模型** 的分子几何构型优化混合策略。

### 核心思想

```
┌─────────────────────────────────────────────────────────────┐
│                    PyBerny-AI 混合优化流程                    │
├─────────────────────────────────────────────────────────────┤
│  外层 (Outer Loop): PyBerny 真实量子化学计算 → 可靠优化        │
│  内层 (Inner Loop): AI 模型预测梯度 → 快速探索                │
│  择优策略：从内外层结果中选择最优作为下一轮起点               │
└─────────────────────────────────────────────────────────────┘
```

### 支持方法

| 方法 | 说明 | 推荐场景 |
|------|------|---------|
| `pyberny` | 纯 PyBerny 基准（完整 BFGS） | 对比基准、小分子快速优化 |
| `hybrid` | PyBerny+AI 混合策略 ⭐ | 大分子、需要加速收敛的场景 |

### 支持的 AI 方法（仅 hybrid 模式）

| AI 方法 | 说明 | 状态 |
|--------|------|------|
| `gpr` | 梯度预测高斯过程回归 | ✅ 已实现（默认） |
| `nn` | 神经网络（能量 - 梯度联合预测） | ✅ 已实现（需安装 torch） |

---

## 快速开始

### 1. 安装依赖

```bash
cd ./pyberny_gpr
pip install -r requirements.txt
```

**使用神经网络方法？** 需要先安装 PyTorch：
```bash
# CPU 版本（推荐，快速安装）
pip install torch --index-url https://download.pytorch.org/whl/cpu

# GPU 版本（需要 CUDA）
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### 2. 运行优化

```bash
# 方法 1: 纯 PyBerny 基准
python main.py --method pyberny --molecule ethanol

# 方法 2: PyBerny+AI 混合（推荐）
python main.py --method hybrid --molecule ethanol --perturb 0.1

# 方法 3: 指定 AI 方法
python main.py --method hybrid --molecule ethanol --perturb 0.1 --ai_method gpr

# 方法 4: 使用神经网络（需要先安装 torch）
python main.py --method hybrid --molecule ethanol --perturb 0.1 --ai_method nn

# 方法 5: 从 XYZ 文件读取初始构型
python main.py --xyz_path ./config/initial.xyz --xyz_name my_mol --method hybrid
```

### 3. 查看结果

```bash
# 输出目录结构
output/
├── CCO_0.1/
│   ├── pyberny/           # 纯 PyBerny 结果
│   │   ├── structures/
│   │   └── plots/
│   └── hybrid_gpr/        # 混合策略结果
│       ├── structures/
│       └── plots/

# 3D 分子结构可视化（递归查找所有 structures 目录）
python draw_structure3D.py
```

---

## 项目结构

```
pyberny_gpr/
├── core/                       # 核心模块
│   ├── molecule.py            # 分子结构、优化历史类
│   └── calculator.py          # 量子化学计算接口（PySCF 后端）
│
├── optimizers/                 # 优化器模块
│   ├── base.py                # 优化器基类
│   ├── hybrid.py              # PyBerny+AI 混合优化器 ⭐
│   ├── pyberny_optimizer.py   # PyBerny BFGS 优化器
│   └── pyberny_baseline.py    # 纯 PyBerny 基准优化器
│
├── models/                     # AI 模型模块
│   ├── gpr_base.py            # GPR 基类（统一接口）
│   ├── energy_gradient_gpr.py # 能量 - 梯度 GPR 模型 ⭐
│   └── energy_gradient_nn.py  # 能量 - 梯度神经网络模型 ⭐
│
├── visualization/              # 可视化模块
│   ├── structure3d.py         # 3D 分子结构可视化（py3Dmol）
│   └── plots.py               # 能量/梯度收敛曲线
│
├── utils/                      # 工具模块
│   ├── io_utils.py            # 输入输出管理
│   └── converters.py          # 坐标转换工具
│
├── config/                     # 配置文件
│   └── default_config.yaml    # 默认配置参数
│
├── main.py                     # 主程序入口
├── draw_structure3D.py         # 3D 结构可视化脚本
│
├── requirements.txt            # Python 依赖列表
├── README.md                   # 项目总览（本文件）
├── DESIGN.md                   # 详细设计文档
└── USAGE.md                    # 详细使用说明
```

---

## 核心特性

### 1. 混合优化策略

| 阶段 | 说明 | 优势 |
|------|------|------|
| **外层优化** | PyBerny BFGS 真实计算 | 可靠、准确 |
| **内层探索** | AI 模型预测梯度 + 梯度下降 | 快速、节省计算 |
| **择优策略** | 综合能量和梯度选择最优 | 平衡探索与利用 |

### 2. 融合设计

- **第 1 轮外层**：`n_init + outer_steps` 步（融合初始采样）
- **后续轮次**：`outer_steps` 步
- **滑动窗口**：只保留最近的外层迭代数据用于训练

### 3. 配置同步

混合方法的外层优化与基准方法使用**完全相同的 berny 配置**，确保公平对比。

### 4. 多 AI 方法支持

- **GPR**：梯度预测高斯过程回归（默认，小数据高效）
- **神经网络**：能量 - 梯度联合预测（可扩展，适合大数据）
- **可选依赖**：torch 为可选依赖，无 torch 环境仍可正常使用 GPR 方法

---

## 技术栈

| 类别 | 工具/库 |
|------|--------|
| **量子化学** | PySCF, berny |
| **分子处理** | RDKit, ASE |
| **优化算法** | berny (完整 BFGS) |
| **机器学习** | scikit-learn (GPR), PyTorch (NN) |
| **可视化** | Matplotlib, py3Dmol |
| **配置管理** | PyYAML |

### PyBerny vs L-BFGS

| 特性 | PyBerny (berny) | L-BFGS |
|------|----------------|--------|
| 算法 | **完整 BFGS** | 有限内存 BFGS |
| Hessian 更新 | 完整矩阵 | 仅保留最近 m 步 |
| 坐标系统 | **冗余内坐标** | 笛卡尔坐标 |
| 收敛速度 | 更快 | 较慢 |

---

## 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--method` | 优化方法 (`pyberny` / `hybrid`) | `pyberny` |
| `--molecule` | 分子名称 | `ethanol` |
| `--smiles` | SMILES 字符串 | `None` |
| `--perturb` | 初始扰动强度 (Å) | `0.0` |
| `--seed` | 随机种子 | `24` |
| `--max-iter` | 最大迭代次数 | `300` |
| `--threshold` | 收敛阈值 | `1.0e-4` |
| `--xyz_path` | .xyz 文件路径 | `None` |
| `--xyz_name` | 输出目录命名 | `None` |
| `--ai_method` | AI 方法（仅 hybrid） | `gpr` |

**注意**：`--xyz_path` 和 `--xyz_name` 必须同时设置或同时为空。

---

## 配置参数（核心）

编辑 `config/default_config.yaml`：

```yaml
# 混合策略核心参数
hybrid:
  ai_method: "gpr"           # AI 方法类型
  outer_steps: 10            # 外层步数（第 1 轮：15 步 = 5+10）
  inner_steps: 5             # 内层探索步数

  # 择优权重（推荐 gradient_weight > energy_weight）
  selection_weights:
    energy_weight: 0.3
    gradient_weight: 0.7

# GPR 参数
gpr:
  n_init: 5                  # 初始采样点数
  max_training_points: 10    # 滑动窗口大小
  local_radius: 0.1          # 局部搜索半径 (Å)

# PyBerny 参数（基准 + 混合外层通用）
berny:
  maxsteps: 500
  trust: 0.3                 # 信任半径 (Å)
  gradient_threshold: 1e-4   # 梯度收敛阈值
```

---

## 输出说明

### 目录结构

```
output/
├── {smiles}_{perturb}/           # 按分子和扰动水平组织
│   ├── pyberny/                  # 纯 PyBerny 结果
│   │   ├── pyberny_YYYYMMDD_HHMMSS.json
│   │   ├── pyberny_trajectory_*.xyz
│   │   ├── structures/
│   │   └── plots/
│   └── hybrid_{ai_method}/       # 混合策略结果
│       ├── hybrid_gpr_YYYYMMDD_HHMMSS.json
│       ├── hybrid_gpr_trajectory_*.xyz
│       ├── structures/
│       └── plots/
```

### 文件类型

| 文件 | 说明 |
|------|------|
| `*.json` | 优化历史（能量、梯度、坐标） |
| `*_trajectory_*.xyz` | 优化轨迹（含梯度信息） |
| `*_details_*.json` | 详细迭代信息 |
| `structures/*.xyz` | 初始/最终结构 |
| `structures/*.html` | 3D 交互式分子结构 |
| `plots/*.png` | 能量/梯度收敛曲线 |

---

## 性能基准

乙醇分子（RHF/cc-pVDZ）典型结果：

| 方法 | 初始能量 (Hartree) | 最优能量 (Hartree) | 迭代次数 |
|------|-------------------|-------------------|---------|
| PyBerny (无扰动) | -154.0803 | -154.0927 | ~20 |
| PyBerny (扰动 0.5) | -153.0956 | -153.9109 | ~30 |
| Hybrid-GPR (扰动 0.5) | -153.0956 | -153.9109 | ~25 |

---

## 文档导航

| 文档 | 说明 |
|------|------|
| [README.md](README.md) | 项目总览（本文件） |
| [DESIGN.md](DESIGN.md) | 核心设计、理论公式、算法流程 |
| [USAGE.md](USAGE.md) | 详细使用说明、配置参数、常见问题 |

---

## 项目维护

**维护者**: 刘喆 (Liu Zhe)  
**邮箱**: 3266048598@qq.com  
**项目目录**: `./pyberny_gpr/`

---

*最后更新：2026 年*
