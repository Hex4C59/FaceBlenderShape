"""使用 Open3D 实时显示三角网格的简易查看器。"""

from __future__ import annotations

import numpy as np
import open3d as o3d
from numpy.typing import NDArray

from face_blender_shape.constants import DEFAULT_OPEN3D_WINDOW_NAME

# 未指定顶点颜色时使用的默认肤色（RGB，0~1）
SKIN_TONE = np.array([0.87, 0.73, 0.62])
# 头壳线框线段颜色（RGB，0~1）
WIREFRAME_LINE_COLOR = np.array([0.38, 0.38, 0.42])


class Open3DMeshViewer:
    """封装 Open3D Visualizer，用于创建窗口并刷新网格几何。"""

    def __init__(
        self,
        window_name: str = DEFAULT_OPEN3D_WINDOW_NAME,
        *,
        wireframe_head: bool = False,
    ) -> None:
        """
        创建可视化窗口与内部 Visualizer。

        :param window_name: 窗口标题字符串。
        :param wireframe_head: True 时头壳为紧凑顶点线框、舌为紧凑实体网格。
        """
        self._visualizer = o3d.visualization.Visualizer()
        self._visualizer.create_window(window_name=window_name)
        self._wireframe_head = wireframe_head
        self._mesh: o3d.geometry.TriangleMesh | None = None
        self._line_set: o3d.geometry.LineSet | None = None
        self._tongue_mesh: o3d.geometry.TriangleMesh | None = None
        self._camera_initialized = False

    def _setup_view_control(self, lookat_points: NDArray[np.float64]) -> None:
        """首次刷新时根据采样点质心设置侧向观察相机与缩放。

        :param lookat_points: 用于计算观察目标的点集，形状 (N, 3)。
        """
        ctr = self._visualizer.get_view_control()
        lookat = np.asarray(lookat_points.mean(axis=0), dtype=np.float64).reshape(3, 1)
        ctr.set_lookat(lookat)
        # front 为视线方向（从相机指向场景），不可为零向量，否则视口无有效投影。
        ctr.set_front(np.array([-1.0, 0.0, 0.0]))
        ctr.set_up(np.array([0.0, 1.0, 0.0]))
        ctr.set_zoom(0.6)

    def _update_wireframe_head(
        self,
        shell_vertices: NDArray[np.float64],
        tongue_vertices: NDArray[np.float64],
        *,
        shell_edges: NDArray[np.int64],
        tongue_faces: NDArray[np.int64],
    ) -> None:
        """刷新「紧凑外壳线框 + 紧凑舌实体」：每帧更新两侧顶点坐标，棱与舌面拓扑不变。

        :param shell_vertices: 外壳子网格顶点，形状 (Ks, 3)。
        :param tongue_vertices: 舌子网格顶点，形状 (Kt, 3)。
        :param shell_edges: 外壳局部顶点下标构成的线段对，形状 (E, 2)。
        :param tongue_faces: 舌三角面局部下标，形状 (T, 3)。
        """
        if self._line_set is None:
            self._line_set = o3d.geometry.LineSet()
            self._line_set.points = o3d.utility.Vector3dVector(shell_vertices)
            self._line_set.lines = o3d.utility.Vector2iVector(
                shell_edges.astype(np.int32, copy=False)
            )
            self._line_set.colors = o3d.utility.Vector3dVector(
                np.tile(WIREFRAME_LINE_COLOR, (len(shell_edges), 1))
            )

            self._tongue_mesh = o3d.geometry.TriangleMesh()
            self._tongue_mesh.vertices = o3d.utility.Vector3dVector(tongue_vertices)
            self._tongue_mesh.triangles = o3d.utility.Vector3iVector(
                tongue_faces.astype(np.int32, copy=False)
            )
            self._tongue_mesh.vertex_colors = o3d.utility.Vector3dVector(
                np.tile(SKIN_TONE, (len(tongue_vertices), 1))
            )
            self._tongue_mesh.compute_vertex_normals()

            self._visualizer.add_geometry(self._line_set)
            self._visualizer.add_geometry(self._tongue_mesh)
            ro = self._visualizer.get_render_option()
            ro.line_width = 2.0
            if not self._camera_initialized:
                combined = np.concatenate([shell_vertices, tongue_vertices], axis=0)
                self._setup_view_control(combined)
                self._camera_initialized = True
        else:
            assert self._line_set is not None and self._tongue_mesh is not None
            self._line_set.points = o3d.utility.Vector3dVector(shell_vertices)
            self._tongue_mesh.vertices = o3d.utility.Vector3dVector(tongue_vertices)
            self._tongue_mesh.compute_vertex_normals()
            self._visualizer.update_geometry(self._line_set)
            self._visualizer.update_geometry(self._tongue_mesh)

        self._visualizer.poll_events()
        self._visualizer.update_renderer()

    def _update_solid_mesh(
        self,
        vertices: NDArray[np.float64],
        faces: NDArray[np.int64],
    ) -> None:
        """整头实体三角网格模式下的顶点与面刷新（固定默认肤色）。"""
        if self._mesh is None:
            self._mesh = o3d.geometry.TriangleMesh()
            self._mesh.vertices = o3d.utility.Vector3dVector(vertices)
            self._mesh.triangles = o3d.utility.Vector3iVector(faces)
            self._mesh.vertex_colors = o3d.utility.Vector3dVector(
                np.tile(SKIN_TONE, (len(vertices), 1))
            )

            self._mesh.compute_vertex_normals()
            self._visualizer.add_geometry(self._mesh)
            if not self._camera_initialized:
                self._setup_view_control(vertices)
                self._camera_initialized = True
        else:
            self._mesh.vertices = o3d.utility.Vector3dVector(vertices)
            self._mesh.triangles = o3d.utility.Vector3iVector(faces)
            self._mesh.compute_vertex_normals()
            self._visualizer.update_geometry(self._mesh)

        self._visualizer.poll_events()
        self._visualizer.update_renderer()

    def update(
        self,
        vertices: NDArray[np.float64],
        faces: NDArray[np.int64],
        *,
        shell_edges: NDArray[np.int64] | None = None,
        tongue_faces: NDArray[np.int64] | None = None,
        shell_vertices: NDArray[np.float64] | None = None,
        tongue_vertices: NDArray[np.float64] | None = None,
    ) -> None:
        """
        首次调用时向场景添加几何体，之后仅更新顶点（及线框模式下的法线）并重绘。

        线框模式下 ``vertices`` / ``faces`` 仅占位，实际绘制用 shell / tongue 子网格参数。

        :param vertices: 实体模式为全部顶点 (N,3)；线框模式可传占位（与首帧一致即可）。
        :param faces: 实体模式下的三角面 (M,3)；线框模式可传占位。
        :param shell_edges: 线框模式必填，外壳紧凑网格的线段对 (E,2)。
        :param tongue_faces: 线框模式必填，舌紧凑三角面 (T,3)。
        :param shell_vertices: 线框模式必填，外壳紧凑顶点 (Ks,3)。
        :param tongue_vertices: 线框模式必填，舌紧凑顶点 (Kt,3)。
        """
        if self._wireframe_head:
            if (
                shell_edges is None
                or tongue_faces is None
                or shell_vertices is None
                or tongue_vertices is None
            ):
                raise ValueError(
                    "wireframe_head=True 时须传入 shell_vertices、tongue_vertices、"
                    "shell_edges、tongue_faces"
                )
            self._update_wireframe_head(
                shell_vertices,
                tongue_vertices,
                shell_edges=shell_edges,
                tongue_faces=tongue_faces,
            )
            return

        if (
            shell_edges is not None
            or tongue_faces is not None
            or shell_vertices is not None
            or tongue_vertices is not None
        ):
            raise ValueError("实体模式下不应传入线框紧凑子网格参数")

        self._update_solid_mesh(vertices, faces)
