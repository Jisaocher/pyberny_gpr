"""
3D 分子结构可视化
使用 py3Dmol 进行交互式分子结构展示
"""
import os
from typing import Tuple


class MoleculeVisualizer3D:
    """
    3D 分子结构可视化器（使用 py3Dmol）
    """

    def __init__(self, font_size: int = 14, figure_size: Tuple[int, int] = (800, 600),
                 dpi: int = 300, show_atom_labels: bool = True,
                 ai_method: str = None):
        """
        初始化可视化器

        Args:
            font_size: 字体大小（HTML 中不使用）
            figure_size: 图形尺寸 (width, height)
            dpi: 分辨率（HTML 中不使用）
            show_atom_labels: 是否显示原子标签
            ai_method: AI 方法类型
        """
        self.font_size = font_size
        self.figure_size = figure_size
        self.dpi = dpi
        self.show_atom_labels = show_atom_labels
        self.ai_method = ai_method

    def visualize_from_xyz(self, xyz_file: str, title: str = None,
                          save_path: str = None) -> str:
        """
        从 XYZ 文件读取并生成 HTML 可视化

        Args:
            xyz_file: XYZ 文件路径
            title: 标题（默认为文件名）
            save_path: HTML 保存路径（默认为与 XYZ 同名.html）

        Returns:
            save_path: 保存的文件路径
        """
        try:
            import py3Dmol
        except ImportError:
            print("Error: py3Dmol not installed. Install with: pip install py3Dmol")
            return None

        # 读取 XYZ 文件
        if not os.path.exists(xyz_file):
            print(f"Error: XYZ file not found: {xyz_file}")
            return None

        with open(xyz_file, 'r', encoding='utf-8') as f:
            xyz_content = f.read()

        # 创建 3D 视图
        width, height = self.figure_size
        view = py3Dmol.view(width=width, height=height)

        # 添加分子模型
        view.addModel(xyz_content, 'xyz')

        # 设置样式：球棍模型
        view.setStyle({
            'sphere': {'scale': 0.3},
            'stick': {'radius': 0.15}
        })

        # 自动调整视图
        view.zoomTo()

        # 设置标题
        if title is None:
            # 使用文件名作为标题
            title = os.path.basename(xyz_file).rsplit('.', 1)[0]

        # 生成 HTML
        html_content = view._make_html()
        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>{title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; text-align: center; margin: 0; padding: 20px; }}
                h1 {{ color: #2c3e50; margin-top: 20px; }}
                .container {{ display: flex; justify-content: center; align-items: center; }}
            </style>
        </head>
        <body>
            <h1>{title}</h1>
            <div class="container">
                {html_content}
            </div>
        </body>
        </html>
        """

        # 保存 HTML 文件
        if save_path is None:
            # 默认保存为与 XYZ 同名.html
            save_path = xyz_file.rsplit('.', 1)[0] + '.html'

        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(full_html)
        print(f"分子结构图已保存：{save_path}")

        return save_path
