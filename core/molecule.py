"""
分子结构数据类
封装分子的基本信息、坐标操作和转换功能
"""
import numpy as np
from typing import List, Tuple, Optional, Dict
from rdkit import Chem
from rdkit.Chem import AllChem


class Molecule:
    """
    分子结构类
    
    属性:
        atom_symbols: 原子符号列表
        coords: 笛卡尔坐标 (n_atoms, 3)
        n_atoms: 原子数量
        smiles: SMILES 字符串
    """
    
    def __init__(self, atom_symbols: List[str], coords: np.ndarray, 
                 smiles: str = "", name: str = ""):
        """
        初始化分子
        
        Args:
            atom_symbols: 原子符号列表 ['C', 'C', 'O', 'H', ...]
            coords: 笛卡尔坐标数组 (n_atoms, 3)
            smiles: SMILES 字符串
            name: 分子名称
        """
        self.atom_symbols = atom_symbols
        self.coords = np.array(coords, dtype=np.float64)
        self.smiles = smiles
        self.name = name if name else f"molecule_{smiles}"
        self.n_atoms = len(atom_symbols)
        
        # 计算原子质量（用于后续可能的质心计算）
        self._atomic_masses = self._get_atomic_masses()
    
    def _get_atomic_masses(self) -> np.ndarray:
        """获取各原子质量"""
        mass_dict = {
            'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999,
            'F': 18.998, 'P': 30.974, 'S': 32.065, 'Cl': 35.453
        }
        return np.array([mass_dict.get(sym, 12.0) for sym in self.atom_symbols])
    
    @classmethod
    def from_smiles(cls, smiles: str, seed: int = 42, 
                    perturb_strength: float = 0.0) -> 'Molecule':
        """
        从 SMILES 字符串生成分子
        
        Args:
            smiles: SMILES 字符串
            seed: 随机种子
            perturb_strength: 扰动强度 (Å)
        
        Returns:
            Molecule 对象
        """
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=seed)
        
        atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
        coords = mol.GetConformer().GetPositions().copy()
        
        # 添加扰动
        if perturb_strength > 0:
            np.random.seed(seed)
            perturbation = np.random.uniform(
                -perturb_strength, perturb_strength, size=coords.shape
            )
            coords += perturbation
        
        return cls(atom_symbols, coords, smiles=smiles)
    
    def get_coords_flat(self) -> np.ndarray:
        """获取展平的坐标数组 (3*n_atoms,)"""
        return self.coords.flatten()
    
    def set_coords_flat(self, x_flat: np.ndarray) -> None:
        """从展平的坐标数组设置坐标"""
        self.coords = x_flat.reshape(self.n_atoms, 3)
    
    def get_displacement(self, other: 'Molecule') -> np.ndarray:
        """计算与另一个分子的坐标位移"""
        return self.coords - other.coords
    
    def get_rmsd(self, other: 'Molecule') -> float:
        """计算与另一个分子的 RMSD"""
        disp = self.get_displacement(other)
        return np.sqrt(np.mean(np.sum(disp**2, axis=1)))
    
    def copy(self) -> 'Molecule':
        """创建分子的深拷贝"""
        return Molecule(
            self.atom_symbols.copy(),
            self.coords.copy(),
            self.smiles,
            self.name
        )
    
    def to_xyz_string(self) -> str:
        """转换为 XYZ 格式字符串"""
        lines = [str(self.n_atoms), f"{self.name}"]
        for i, sym in enumerate(self.atom_symbols):
            lines.append(f"{sym:2s} {self.coords[i,0]:12.6f} {self.coords[i,1]:12.6f} {self.coords[i,2]:12.6f}")
        return '\n'.join(lines)
    
    def save_xyz(self, filepath: str) -> None:
        """保存为 XYZ 文件"""
        with open(filepath, 'w') as f:
            f.write(self.to_xyz_string())
    
    def __repr__(self) -> str:
        return f"Molecule({self.name}, {self.n_atoms} atoms)"


class IterationData:
    """
    单次迭代的数据记录

    属性:
        iteration: 迭代序号
        energy: 能量 (Hartree)
        gradient: 梯度数组 (3*n_atoms,)
        gradient_norm: 梯度范数
        coords: 坐标 (3*n_atoms,)
        displacement: 与上一步的位移
        timestamp: 时间戳
        round_num: 轮次编号（混合策略用，0=初始采样）
        stage: 阶段标记 ('outer'/'inner'/'pyberny')
        gradient_pred: 预测梯度（混合策略内层用，None=真实梯度）
        gradient_pred_norm: 预测梯度范数
    """

    def __init__(self, iteration: int, energy: float, gradient: np.ndarray,
                 coords: np.ndarray, displacement: Optional[np.ndarray] = None,
                 round_num: int = 0, stage: str = 'pyberny',
                 gradient_pred: Optional[np.ndarray] = None):
        self.iteration = iteration
        self.energy = energy
        self.gradient = gradient.copy()
        self.gradient_norm = np.linalg.norm(gradient)
        self.coords = coords.copy()
        self.displacement = displacement.copy() if displacement is not None else None
        self.timestamp = 0.0  # 可在优化器中设置
        
        # 混合策略额外字段
        self.round_num = round_num  # 0=初始采样，1+=第 N 轮
        self.stage = stage  # 'outer'/'inner'/'pyberny'
        self.gradient_pred = gradient_pred.copy() if gradient_pred is not None else None
        self.gradient_pred_norm = np.linalg.norm(gradient_pred) if gradient_pred is not None else None
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'iteration': self.iteration,
            'energy': self.energy,
            'gradient_norm': self.gradient_norm,
            'gradient': self.gradient.tolist(),
            'coords': self.coords.tolist(),
            'displacement': self.displacement.tolist() if self.displacement is not None else None,
            # 混合策略额外字段
            'round_num': self.round_num,
            'stage': self.stage,
            'gradient_pred': self.gradient_pred.tolist() if self.gradient_pred is not None else None,
            'gradient_pred_norm': self.gradient_pred_norm
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'IterationData':
        """从字典创建"""
        return cls(
            data['iteration'],
            data['energy'],
            np.array(data['gradient']),
            np.array(data['coords']),
            np.array(data['displacement']) if data.get('displacement') is not None else None,
            round_num=data.get('round_num', 0),
            stage=data.get('stage', 'pyberny'),
            gradient_pred=np.array(data['gradient_pred']) if data.get('gradient_pred') is not None else None
        )


class OptimizationHistory:
    """
    优化历史记录类
    
    管理所有迭代数据，提供统计和查询功能
    """
    
    def __init__(self):
        self.iterations: List[IterationData] = []
        self.start_time = None
        self.end_time = None
        self.converged = False
        self.convergence_iteration = None
    
    def add_iteration(self, data: IterationData) -> None:
        """添加一次迭代记录"""
        self.iterations.append(data)
    
    def get_energies(self) -> np.ndarray:
        """获取所有能量值"""
        return np.array([it.energy for it in self.iterations])
    
    def get_gradient_norms(self) -> np.ndarray:
        """获取所有梯度范数"""
        return np.array([it.gradient_norm for it in self.iterations])
    
    def get_coords_history(self) -> np.ndarray:
        """获取所有坐标历史 (n_iterations, 3*n_atoms)"""
        return np.array([it.coords for it in self.iterations])
    
    def get_best_iteration(self, metric: str = 'energy') -> IterationData:
        """
        获取最优迭代

        Args:
            metric: 'energy' 或 'gradient'
        """
        if not self.iterations:
            return None

        if metric == 'energy':
            idx = np.argmin([it.energy for it in self.iterations])
        elif metric == 'gradient':
            idx = np.argmin([it.gradient_norm for it in self.iterations])
        else:
            idx = np.argmin([it.energy + 0.1 * it.gradient_norm for it in self.iterations])

        return self.iterations[idx]

    def get_last_iteration(self) -> Optional[IterationData]:
        """
        获取最后一次迭代数据

        Returns:
            最后一次迭代的数据，如果历史为空则返回 None
        """
        if not self.iterations:
            return None
        return self.iterations[-1]
    
    def get_best_coords(self) -> np.ndarray:
        """获取最优坐标"""
        best = self.get_best_iteration()
        return best.coords if best else None
    
    def check_convergence(self, threshold: float) -> bool:
        """检查是否收敛"""
        if not self.iterations:
            return False
        return self.iterations[-1].gradient_norm < threshold
    
    def get_displacements(self) -> np.ndarray:
        """获取所有位移"""
        displacements = []
        for it in self.iterations:
            if it.displacement is not None:
                displacements.append(np.linalg.norm(it.displacement))
            else:
                displacements.append(0.0)
        return np.array(displacements)
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'iterations': [it.to_dict() for it in self.iterations],
            'converged': bool(self.converged),
            'convergence_iteration': int(self.convergence_iteration) if self.convergence_iteration is not None else None,
            'start_time': float(self.start_time) if self.start_time is not None else None,
            'end_time': float(self.end_time) if self.end_time is not None else None
        }
    
    def save_json(self, filepath: str) -> None:
        """保存为 JSON 文件"""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_json(cls, filepath: str) -> 'OptimizationHistory':
        """从 JSON 文件加载"""
        import json
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        history = cls()
        for it_data in data['iterations']:
            history.add_iteration(IterationData.from_dict(it_data))
        history.converged = data.get('converged', False)
        history.convergence_iteration = data.get('convergence_iteration')
        return history
    
    def __len__(self) -> int:
        return len(self.iterations)
    
    def __repr__(self) -> str:
        status = "converged" if self.converged else "not converged"
        return f"OptimizationHistory({len(self)} iterations, {status})"
