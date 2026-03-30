"""使用 Open3D 实时显示三角网格的简易查看器。"""

from __future__ import annotations

import numpy as np
import open3d as o3d
from numpy.typing import NDArray

from face_blender_shape.constants import DEFAULT_OPEN3D_WINDOW_NAME

# 未指定顶点颜色时使用的默认肤色（RGB，0~1）
SKIN_TONE = np.array([0.87, 0.73, 0.62])


class Open3DMeshViewer:
    """封装 Open3D Visualizer，用于创建窗口并刷新网格几何。"""

    def __init__(self, window_name: str = DEFAULT_OPEN3D_WINDOW_NAME) -> None:
        """
        创建可视化窗口与内部 Visualizer。

        :param window_name: 窗口标题字符串。
        """
        self._visualizer = o3d.visualization.Visualizer()
        self._visualizer.create_window(window_name=window_name)
        self._mesh: o3d.geometry.TriangleMesh | None = None

    def update(
        self,
        vertices: NDArray[np.float64],
        faces: NDArray[np.int64],
        *,
        vertex_colors: NDArray[np.float64] | None = None,
    ) -> None:
        """
        首次调用时向场景添加网格，之后仅更新顶点、面与法线并重绘。

        :param vertices: 顶点坐标，形状 (N, 3)，dtype 浮点。
        :param faces: 三角形索引，形状 (M, 3)，每行三个顶点下标。
        :param vertex_colors: 可选，每顶点 RGB，形状 (N, 3)，取值 0~1；省略则使用默认肤色。
        """
        if self._mesh is None:
            self._mesh = o3d.geometry.TriangleMesh()
            self._mesh.vertices = o3d.utility.Vector3dVector(vertices)
            self._mesh.triangles = o3d.utility.Vector3iVector(faces)

            if vertex_colors is not None:
                self._mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
            else:
                self._mesh.vertex_colors = o3d.utility.Vector3dVector(
                    np.tile(SKIN_TONE, (len(vertices), 1))
                )

            # 按三角面计算顶点法线，供光照与着色使用；无法则网格可能发灰或不平滑。
            self._mesh.compute_vertex_normals()
            # 将网格加入场景；仅首次需要，之后用 update_geometry 刷新顶点即可。
            self._visualizer.add_geometry(self._mesh)
            # ViewControl 负责相机外参：观察目标、视线方向、屏幕上方向、缩放等。
            ctr = self._visualizer.get_view_control()
            # 观察目标取当前帧顶点质心，避免模型不在世界原点时视角跑偏；Open3D 绑定要求 shape (3, 1)、float64。
            lookat = np.asarray(vertices.mean(axis=0), dtype=np.float64).reshape(3, 1)
            ctr.set_lookat(lookat)
            # set_front：相机视线方向（从相机指向场景的单位向量，不必归一化也可）。此处沿 -X 看，即从 +X 侧望向 -X，得到一侧侧脸。
            ctr.set_front(np.array([-1.0, 0.0, 0.0]))
            # set_up：屏幕“上”对应的世界方向，须与 front 不共线。Z 朝上时脸可能横躺；改用世界 +Y 为屏上，相当于绕视线调整翻滚。
            ctr.set_up(np.array([0.0, 1.0, 0.0]))
            # 视距缩放：越小相机越远画面越小，越大越近；可按模型尺度在约 0.3～1.0 间微调。
            ctr.set_zoom(0.6)
        else:
            self._mesh.vertices = o3d.utility.Vector3dVector(vertices)
            self._mesh.triangles = o3d.utility.Vector3iVector(faces)
            self._mesh.compute_vertex_normals()
            self._visualizer.update_geometry(self._mesh)

        self._visualizer.poll_events()
        self._visualizer.update_renderer()
