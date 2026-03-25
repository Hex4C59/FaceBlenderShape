"""显示层子包。

包级入口仅保留最小稳定显示能力；其余内容请从具体模块导入。
"""

from face_blender_shape.viewer.open3d_viewer import CameraProfile, Open3DMeshViewer

__all__ = ["CameraProfile", "Open3DMeshViewer"]
