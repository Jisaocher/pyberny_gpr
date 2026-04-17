"""
能量和梯度可视化图表
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Dict
from core.molecule import OptimizationHistory

# 导入 zhplot 以支持中文显示
try:
    import zhplot
    zhplot.matplotlib_chineseize()
except (ImportError, AttributeError):
    # 如果 zhplot 不可用，尝试其他中文支持方式
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS',
                                        'DejaVu Sans', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


class OptimizationPlotter:
    """
    优化过程图表绘制器
    """

    def __init__(self, font_size: int = 14, figure_size: Tuple[int, int] = (12, 8),
                 dpi: int = 300, ai_method: str = None):
        """
        初始化绘图器

        Args:
            font_size: 字体大小
            figure_size: 图形尺寸
            dpi: 分辨率
            ai_method: AI 方法类型（'simple'/'gradient'/'random_forest'等）
        """
        self.font_size = font_size
        self.figure_size = figure_size
        self.dpi = dpi
        self.ai_method = ai_method
        self.ai_suffix = self._get_ai_suffix()

        # 设置全局字体
        plt.rcParams['font.size'] = font_size
        plt.rcParams['axes.labelsize'] = font_size
        plt.rcParams['axes.titlesize'] = font_size
        plt.rcParams['xtick.labelsize'] = font_size
        plt.rcParams['ytick.labelsize'] = font_size
        plt.rcParams['legend.fontsize'] = font_size
    
    def _get_ai_suffix(self) -> str:
        """获取 AI 方法的简短后缀"""
        suffix_map = {
            'simple': 'gpr',
            'gradient': 'ggpr',
            'random_forest': 'rf',
            'neural_network': 'nn'
        }
        return suffix_map.get(self.ai_method, '') if self.ai_method else ''
    
    def plot_energy_history(self, history: OptimizationHistory,
                            title: str = "能量收敛曲线",
                            save_path: str = None,
                            show: bool = False,
                            y_label: str = "Energy (Hartree)") -> plt.Figure:
        """
        绘制能量历史曲线
        
        Args:
            history: 优化历史
            title: 标题
            save_path: 保存路径
            show: 是否显示
            y_label: Y 轴标签
        
        Returns:
            fig: 图形对象
        """
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
        
        energies = history.get_energies()
        iterations = np.arange(len(energies))
        
        ax.plot(iterations, energies, 'b-o', linewidth=2, markersize=4,
               label='Energy')
        
        # 标注最优值
        best_idx = np.argmin(energies)
        ax.scatter([best_idx], [energies[best_idx]], c='red', s=100,
                  marker='*', zorder=5, label=f'Best: {energies[best_idx]:.8f}')
        
        ax.set_xlabel('Iteration', fontsize=self.font_size)
        ax.set_ylabel(y_label, fontsize=self.font_size)
        ax.set_title(title, fontsize=self.font_size, pad=15)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=self.font_size)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def plot_gradient_history(self, history: OptimizationHistory,
                              title: str = "梯度收敛曲线",
                              save_path: str = None,
                              show: bool = False,
                              log_scale: bool = True) -> plt.Figure:
        """
        绘制梯度范数历史曲线
        
        Args:
            history: 优化历史
            title: 标题
            save_path: 保存路径
            show: 是否显示
            log_scale: 是否使用对数坐标
        
        Returns:
            fig: 图形对象
        """
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
        
        grad_norms = history.get_gradient_norms()
        iterations = np.arange(len(grad_norms))
        
        ax.plot(iterations, grad_norms, 'r-o', linewidth=2, markersize=4,
               label='Gradient Norm')
        
        # 标注最优值
        best_idx = np.argmin(grad_norms)
        ax.scatter([best_idx], [grad_norms[best_idx]], c='blue', s=100,
                  marker='*', zorder=5, label=f'Best: {grad_norms[best_idx]:.6f}')
        
        if log_scale:
            ax.set_yscale('log')
        
        ax.set_xlabel('Iteration', fontsize=self.font_size)
        ax.set_ylabel('Gradient Norm', fontsize=self.font_size)
        ax.set_title(title, fontsize=self.font_size, pad=15)
        ax.grid(True, alpha=0.3, which='both' if log_scale else 'major')
        ax.legend(fontsize=self.font_size)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def plot_combined_history(self, history: OptimizationHistory,
                              title: str = "优化收敛曲线",
                              save_path: str = None,
                              show: bool = False) -> plt.Figure:
        """
        绘制能量和梯度的组合图表
        
        Args:
            history: 优化历史
            title: 标题
            save_path: 保存路径
            show: 是否显示
        
        Returns:
            fig: 图形对象
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figure_size, dpi=self.dpi)
        fig.suptitle(title, fontsize=self.font_size + 2, y=1.02)
        
        energies = history.get_energies()
        grad_norms = history.get_gradient_norms()
        iterations = np.arange(len(energies))
        
        # 能量图
        ax1.plot(iterations, energies, 'b-o', linewidth=2, markersize=4)
        ax1.set_ylabel('Energy (Hartree)', fontsize=self.font_size)
        ax1.grid(True, alpha=0.3)
        
        # 标注能量变化
        if len(energies) > 1:
            energy_change = energies[0] - energies[-1]
            ax1.text(0.02, 0.98, f'ΔE = {energy_change:.6f} Hartree',
                    transform=ax1.transAxes, fontsize=self.font_size - 2,
                    verticalalignment='top', bbox=dict(boxstyle='round',
                    facecolor='wheat', alpha=0.5))
        
        # 梯度图
        ax2.plot(iterations, grad_norms, 'r-o', linewidth=2, markersize=4)
        ax2.set_yscale('log')
        ax2.set_xlabel('Iteration', fontsize=self.font_size)
        ax2.set_ylabel('Gradient Norm', fontsize=self.font_size)
        ax2.grid(True, alpha=0.3, which='both')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def plot_comparison(self, histories: Dict[str, OptimizationHistory],
                        title: str = "优化方法对比",
                        save_path: str = None,
                        show: bool = False,
                        plot_type: str = 'both') -> plt.Figure:
        """
        绘制多个优化历史的对比图
        
        Args:
            histories: 优化历史字典 {method_name: history}
            title: 标题
            save_path: 保存路径
            show: 是否显示
            plot_type: 'energy', 'gradient', 或 'both'
        
        Returns:
            fig: 图形对象
        """
        colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))
        
        if plot_type == 'both':
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figure_size, dpi=self.dpi)
            fig.suptitle(title, fontsize=self.font_size + 2, y=1.02)
        else:
            fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
            ax.set_title(title, fontsize=self.font_size, pad=15)
        
        for idx, (name, history) in enumerate(histories.items()):
            energies = history.get_energies()
            grad_norms = history.get_gradient_norms()
            iterations = np.arange(len(energies))
            
            color = colors[idx]
            label = f"{name} ({len(energies)} steps)"
            
            if plot_type in ['energy', 'both']:
                target_ax = ax1 if plot_type == 'both' else ax
                target_ax.plot(iterations, energies, color=color,
                              linewidth=2, markersize=4, label=label,
                              marker='o', alpha=0.8)
            
            if plot_type in ['gradient', 'both']:
                target_ax = ax2 if plot_type == 'both' else ax
                target_ax.plot(iterations, grad_norms, color=color,
                              linewidth=2, markersize=4, label=label,
                              marker='s', alpha=0.8)
        
        if plot_type == 'both':
            ax1.set_ylabel('Energy (Hartree)', fontsize=self.font_size)
            ax1.grid(True, alpha=0.3)
            ax1.legend(fontsize=self.font_size - 2)
            
            ax2.set_yscale('log')
            ax2.set_xlabel('Iteration', fontsize=self.font_size)
            ax2.set_ylabel('Gradient Norm', fontsize=self.font_size)
            ax2.grid(True, alpha=0.3, which='both')
            ax2.legend(fontsize=self.font_size - 2)
        else:
            ax.set_xlabel('Iteration', fontsize=self.font_size)
            if plot_type == 'energy':
                ax.set_ylabel('Energy (Hartree)', fontsize=self.font_size)
            else:
                ax.set_yscale('log')
                ax.set_ylabel('Gradient Norm', fontsize=self.font_size)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=self.font_size)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def plot_displacement_history(self, history: OptimizationHistory,
                                  title: str = "位移收敛曲线",
                                  save_path: str = None,
                                  show: bool = False) -> plt.Figure:
        """
        绘制位移历史曲线
        
        Args:
            history: 优化历史
            title: 标题
            save_path: 保存路径
            show: 是否显示
        
        Returns:
            fig: 图形对象
        """
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
        
        displacements = history.get_displacements()
        iterations = np.arange(len(displacements))
        
        ax.plot(iterations, displacements, 'g-o', linewidth=2, markersize=4)
        ax.set_yscale('log')
        ax.set_xlabel('Iteration', fontsize=self.font_size)
        ax.set_ylabel('Displacement (Å)', fontsize=self.font_size)
        ax.set_title(title, fontsize=self.font_size, pad=15)
        ax.grid(True, alpha=0.3, which='both')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def plot_all(self, history: OptimizationHistory,
                 base_path: str,
                 title_prefix: str = "",
                 ai_method: str = None) -> List[str]:
        """
        绘制所有图表并保存

        Args:
            history: 优化历史
            base_path: 保存路径前缀（不含扩展名）
            title_prefix: 标题前缀（包含 AI 方法信息）
            ai_method: AI 方法类型

        Returns:
            saved_paths: 保存的文件路径列表
        """
        import os
        saved_paths = []
        
        # 确保目录存在
        os.makedirs(base_path, exist_ok=True)

        # 构建文件名：如果有 AI 方法后缀，则添加到文件名中
        if self.ai_suffix:
            file_suffix = f"_{self.ai_suffix}"
        else:
            file_suffix = ""

        # 能量图
        energy_path = f"{base_path}/{self.ai_suffix}_energy.png" if self.ai_suffix else f"{base_path}/energy.png"
        self.plot_energy_history(history, title=f"{title_prefix}能量收敛",
                                save_path=energy_path)
        saved_paths.append(energy_path)

        # 梯度图
        grad_path = f"{base_path}/{self.ai_suffix}_gradient.png" if self.ai_suffix else f"{base_path}/gradient.png"
        self.plot_gradient_history(history, title=f"{title_prefix}梯度收敛",
                                  save_path=grad_path)
        saved_paths.append(grad_path)

        # 组合图
        combined_path = f"{base_path}/{self.ai_suffix}_combined.png" if self.ai_suffix else f"{base_path}/combined.png"
        self.plot_combined_history(history, title=f"{title_prefix}优化收敛",
                                  save_path=combined_path)
        saved_paths.append(combined_path)

        # 位移图（添加 AI 后缀）
        disp_path = f"{base_path}/{self.ai_suffix}_displacement.png" if self.ai_suffix else f"{base_path}/displacement.png"
        self.plot_displacement_history(history, title=f"{title_prefix}位移收敛",
                                      save_path=disp_path)
        saved_paths.append(disp_path)

        return saved_paths


def create_optimization_plots(history: OptimizationHistory, save_dir: str,
                              method_name: str = "") -> List[str]:
    """
    便捷函数：创建优化图表
    
    Args:
        history: 优化历史
        save_dir: 保存目录
        method_name: 方法名称
    
    Returns:
        saved_paths: 保存的文件路径列表
    """
    plotter = OptimizationPlotter()
    title_prefix = f"{method_name} - " if method_name else ""
    base_path = f"{save_dir}/{method_name}_plot" if method_name else f"{save_dir}/plot"
    
    return plotter.plot_all(history, base_path, title_prefix)
