"""
坐标转换工具
笛卡尔坐标与内坐标（Z-matrix）之间的转换
"""
import numpy as np
from typing import List, Tuple, Optional
from core.molecule import Molecule


class CoordinateConverter:
    """
    坐标转换器
    
    实现笛卡尔坐标与内坐标（Z-matrix）之间的转换
    """
    
    def __init__(self, atom_symbols: List[str], reference_indices: List[Tuple[int, int, int]] = None):
        """
        初始化坐标转换器
        
        Args:
            atom_symbols: 原子符号列表
            reference_indices: 参考原子索引 [(bond_idx, angle_idx, dihedral_idx), ...]
        """
        self.atom_symbols = atom_symbols
        self.n_atoms = len(atom_symbols)
        self.reference_indices = reference_indices
        self._build_reference()
    
    def _build_reference(self) -> None:
        """构建参考原子索引"""
        if self.reference_indices is None:
            # 自动构建简单的参考索引
            # 对于乙醇等小分子，使用简单的连接关系
            self.reference_indices = []
            
            for i in range(self.n_atoms):
                if i == 0:
                    # 第一个原子：无参考
                    self.reference_indices.append(None)
                elif i == 1:
                    # 第二个原子：只参考第一个原子（键长）
                    self.reference_indices.append((0, None, None))
                elif i == 2:
                    # 第三个原子：参考前两个原子（键长、键角）
                    self.reference_indices.append((1, 0, None))
                else:
                    # 其他原子：参考前三个原子
                    self.reference_indices.append((i-1, i-2, i-3))
    
    def cartesian_to_internal(self, coords: np.ndarray) -> np.ndarray:
        """
        笛卡尔坐标转内坐标
        
        Args:
            coords: 笛卡尔坐标 (n_atoms, 3)
        
        Returns:
            internal_coords: 内坐标 [bond_lengths, bond_angles, dihedral_angles]
        """
        n_atoms = coords.shape[0]
        internal = []
        
        for i in range(n_atoms):
            ref = self.reference_indices[i]
            
            if ref is None:
                # 第一个原子：无内坐标
                continue
            elif ref[1] is None:
                # 只有键长
                bond = self._calculate_distance(coords[i], coords[ref[0]])
                internal.append(bond)
            elif ref[2] is None:
                # 键长和键角
                bond = self._calculate_distance(coords[i], coords[ref[0]])
                angle = self._calculate_angle(coords[i], coords[ref[0]], coords[ref[1]])
                internal.append(bond)
                internal.append(angle)
            else:
                # 键长、键角、二面角
                bond = self._calculate_distance(coords[i], coords[ref[0]])
                angle = self._calculate_angle(coords[i], coords[ref[0]], coords[ref[1]])
                dihedral = self._calculate_dihedral(
                    coords[i], coords[ref[0]], coords[ref[1]], coords[ref[2]]
                )
                internal.append(bond)
                internal.append(angle)
                internal.append(dihedral)
        
        return np.array(internal)
    
    def internal_to_cartesian(self, internal_coords: np.ndarray) -> np.ndarray:
        """
        内坐标转笛卡尔坐标
        
        Args:
            internal_coords: 内坐标
        
        Returns:
            coords: 笛卡尔坐标 (n_atoms, 3)
        """
        coords = np.zeros((self.n_atoms, 3))
        idx = 0
        
        for i in range(self.n_atoms):
            ref = self.reference_indices[i]
            
            if ref is None:
                # 第一个原子放在原点
                coords[i] = [0, 0, 0]
            elif ref[1] is None:
                # 第二个原子放在 z 轴上
                bond = internal_coords[idx]
                coords[i] = [0, 0, bond]
                idx += 1
            elif ref[2] is None:
                # 第三个原子放在 xz 平面上
                bond = internal_coords[idx]
                angle = internal_coords[idx + 1]
                
                coords[i, 0] = bond * np.sin(angle)
                coords[i, 2] = bond * np.cos(angle)
                idx += 2
            else:
                # 其他原子
                bond = internal_coords[idx]
                angle = internal_coords[idx + 1]
                dihedral = internal_coords[idx + 2]
                
                # 使用参考原子构建局部坐标系
                p1 = coords[ref[0]]  # 键参考
                p2 = coords[ref[1]]  # 角参考
                p3 = coords[ref[2]]  # 二面角参考
                
                # 计算新原子位置
                coords[i] = self._build_atom_position(p1, p2, p3, bond, angle, dihedral)
                idx += 3
        
        return coords
    
    def _calculate_distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """计算两点间距离"""
        return np.linalg.norm(p1 - p2)
    
    def _calculate_angle(self, p1: np.ndarray, p2: np.ndarray, 
                         p3: np.ndarray) -> float:
        """计算键角 (p1-p2-p3)"""
        v1 = p1 - p2
        v2 = p3 - p2
        
        v1_norm = v1 / np.linalg.norm(v1)
        v2_norm = v2 / np.linalg.norm(v2)
        
        cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
        return np.arccos(cos_angle)
    
    def _calculate_dihedral(self, p1: np.ndarray, p2: np.ndarray,
                            p3: np.ndarray, p4: np.ndarray) -> float:
        """计算二面角 (p1-p2-p3-p4)"""
        v1 = p2 - p1
        v2 = p3 - p2
        v3 = p4 - p3
        
        # 法向量
        n1 = np.cross(v1, v2)
        n2 = np.cross(v2, v3)
        
        # 归一化
        n1_norm = n1 / np.linalg.norm(n1)
        n2_norm = n2 / np.linalg.norm(n2)
        
        # 二面角
        m1 = np.cross(n1_norm, v2 / np.linalg.norm(v2))
        
        x = np.dot(n1_norm, n2_norm)
        y = np.dot(m1, n2_norm)
        
        return np.arctan2(y, x)
    
    def _build_atom_position(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray,
                             bond: float, angle: float, dihedral: float) -> np.ndarray:
        """
        根据内坐标构建原子位置
        
        Args:
            p1, p2, p3: 参考原子位置
            bond: 键长
            angle: 键角
            dihedral: 二面角
        
        Returns:
            new_pos: 新原子位置
        """
        # 单位向量
        e1 = (p2 - p1) / np.linalg.norm(p2 - p1)
        e2 = (p3 - p2) / np.linalg.norm(p3 - p2)
        
        # 垂直向量
        n = np.cross(e1, e2)
        n = n / np.linalg.norm(n)
        
        # 构建局部坐标系
        local_z = e2
        local_y = n
        local_x = np.cross(local_y, local_z)
        
        # 球坐标到笛卡尔坐标
        x = bond * np.sin(angle) * np.cos(dihedral)
        y = bond * np.sin(angle) * np.sin(dihedral)
        z = bond * np.cos(angle)
        
        # 转换到全局坐标
        new_pos = p2 + x * local_x + y * local_y + z * local_z
        
        return new_pos


def get_internal_coordinates(molecule: Molecule) -> np.ndarray:
    """
    便捷函数：获取分子的内坐标
    
    Args:
        molecule: 分子对象
    
    Returns:
        internal_coords: 内坐标
    """
    converter = CoordinateConverter(molecule.atom_symbols)
    return converter.cartesian_to_internal(molecule.coords)


def get_cartesian_coordinates(molecule: Molecule, 
                              internal_coords: np.ndarray) -> np.ndarray:
    """
    便捷函数：从内坐标获取笛卡尔坐标
    
    Args:
        molecule: 分子对象
        internal_coords: 内坐标
    
    Returns:
        coords: 笛卡尔坐标
    """
    converter = CoordinateConverter(molecule.atom_symbols)
    return converter.internal_to_cartesian(internal_coords)
