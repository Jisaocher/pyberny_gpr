"""
量子化学计算接口
封装 PySCF 的能量和梯度计算
"""
import numpy as np
from typing import Tuple, List, Optional
from pyscf import gto, scf, grad

from core.molecule import Molecule


class QuantumCalculator:
    """
    量子化学计算器
    
    使用 PySCF 进行能量和梯度计算
    """
    
    def __init__(self, basis: str = 'cc-pvdz', method: str = 'RHF',
                 unit: str = 'angstrom', verbose: int = 0):
        """
        初始化计算器
        
        Args:
            basis: 基组名称
            method: 量子化学方法 (RHF, DFT, etc.)
            unit: 坐标单位 ('angstrom' or 'bohr')
            verbose: PySCF 输出详细程度
        """
        self.basis = basis
        self.method = method
        self.unit = unit
        self.verbose = verbose
        
        # 缓存最后的计算结果
        self._last_mol = None
        self._last_mf = None
    
    def _build_mol(self, atom_symbols: List[str], coords: np.ndarray) -> gto.Mole:
        """
        构建 PySCF Mole 对象
        
        Args:
            atom_symbols: 原子符号列表
            coords: 笛卡尔坐标 (n_atoms, 3)
        
        Returns:
            PySCF Mole 对象
        """
        mol = gto.Mole()
        atom_str = '\n'.join([
            f"{atom_symbols[i]}  {coords[i,0]:.8f}  {coords[i,1]:.8f}  {coords[i,2]:.8f}"
            for i in range(len(atom_symbols))
        ])
        mol.atom = atom_str
        mol.basis = self.basis
        mol.unit = self.unit
        mol.build(verbose=self.verbose)
        return mol
    
    def calculate_energy_gradient(self, atom_symbols: List[str], coords: np.ndarray,
                                   force_rebuild: bool = False) -> Tuple[float, np.ndarray]:
        """
        计算能量和梯度
        
        Args:
            atom_symbols: 原子符号列表
            coords: 笛卡尔坐标 (n_atoms, 3)
            force_rebuild: 是否强制重新构建 Mole 对象
        
        Returns:
            energy: 总能量 (Hartree)
            gradient: 梯度数组 (3*n_atoms,)
        """
        mol = self._build_mol(atom_symbols, coords)
        
        # 根据方法创建 SCF 对象
        if self.method.upper() == 'RHF':
            mf = scf.RHF(mol)
        elif self.method.upper() == 'ROHF':
            mf = scf.ROHF(mol)
        elif self.method.upper() == 'UHF':
            mf = scf.UHF(mol)
        else:
            raise ValueError(f"Unsupported method: {self.method}")
        
        # 执行 SCF 计算
        mf.kernel()
        
        # 计算能量
        energy = mf.e_tot
        
        # 计算解析梯度
        g = grad.RHF(mf)
        gradient_au = g.kernel()  # 单位：Hartree/Bohr
        
        # 如果输入是 Angstrom，梯度需要转换
        # PySCF 内部使用原子单位，梯度输出为 Hartree/Bohr
        # 如果坐标是 Angstrom，梯度需要乘以 Bohr/Angstrom = 1/0.529177
        if self.unit == 'angstrom':
            bohr_to_angstrom = 1.0 / 0.529177210903
            gradient_au = gradient_au * bohr_to_angstrom
        
        # 展平梯度数组
        gradient_flat = gradient_au.flatten()
        
        # 缓存结果
        self._last_mol = mol
        self._last_mf = mf
        
        return energy, gradient_flat
    
    def calculate_energy(self, atom_symbols: List[str], coords: np.ndarray) -> float:
        """仅计算能量"""
        energy, _ = self.calculate_energy_gradient(atom_symbols, coords)
        return energy
    
    def calculate_gradient(self, atom_symbols: List[str], coords: np.ndarray) -> np.ndarray:
        """仅计算梯度"""
        _, gradient = self.calculate_energy_gradient(atom_symbols, coords)
        return gradient
    
    def get_scf_object(self):
        """获取最后的 SCF 对象（用于访问轨道等）"""
        return self._last_mf
    
    def get_mol_object(self):
        """获取最后的 Mole 对象"""
        return self._last_mol


class EnergyGradientFunction:
    """
    能量和梯度函数包装器
    用于优化器的回调函数
    """
    
    def __init__(self, calculator: QuantumCalculator, atom_symbols: List[str]):
        """
        初始化
        
        Args:
            calculator: 量子化学计算器
            atom_symbols: 原子符号列表
        """
        self.calculator = calculator
        self.atom_symbols = atom_symbols
        self.n_atoms = len(atom_symbols)
        self.call_count = 0
    
    def __call__(self, x_flat: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        计算能量和梯度
        
        Args:
            x_flat: 展平的坐标 (3*n_atoms,)
        
        Returns:
            energy: 能量
            gradient: 梯度
        """
        coords = x_flat.reshape(self.n_atoms, 3)
        energy, gradient = self.calculator.calculate_energy_gradient(
            self.atom_symbols, coords
        )
        self.call_count += 1
        return energy, gradient
    
    def energy_only(self, x_flat: np.ndarray) -> float:
        """仅计算能量（用于 scipy.optimize.minimize）"""
        coords = x_flat.reshape(self.n_atoms, 3)
        energy, _ = self.calculator.calculate_energy_gradient(
            self.atom_symbols, coords
        )
        self.call_count += 1
        return energy
    
    def gradient_only(self, x_flat: np.ndarray) -> np.ndarray:
        """仅计算梯度（用于 scipy.optimize.minimize）"""
        coords = x_flat.reshape(self.n_atoms, 3)
        _, gradient = self.calculator.calculate_energy_gradient(
            self.atom_symbols, coords
        )
        self.call_count += 1
        return gradient
    
    def reset_count(self):
        """重置调用计数"""
        self.call_count = 0
