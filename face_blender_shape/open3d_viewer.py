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

# 侧面窗：视线沿 -X（与历史行为一致）
_SIDE_FRONT = np.array([-1.0, 0.0, 0.0])
_SIDE_UP = np.array([0.0, 1.0, 0.0])
# 正面窗：视线沿 +Z（相机在 -Z 一侧看向脸）。若用 -Z 会像从颅内向外看、牙弓呈舌侧，与侧窗不一致时改此向量。
_FRONT_VIEW_FRONT = np.array([0.0, 0.0, 1.0])
_FRONT_VIEW_UP = np.array([0.0, 1.0, 0.0])

# 双窗并排，避免 GLFW 默认同一 (left,top) 完全重叠
_DUAL_VIEW_WIDTH = 880
_DUAL_VIEW_HEIGHT = 720
_DUAL_VIEW_GAP = 16
_DUAL_VIEW_LEFT0 = 48
_DUAL_VIEW_TOP = 80


class Open3DMeshViewer:
    """封装 Open3D Visualizer，用于创建窗口并刷新网格几何。"""

    def __init__(
        self,
        window_name: str = DEFAULT_OPEN3D_WINDOW_NAME,
        *,
        wireframe_head: bool = False,
        dual_view: bool = False,
    ) -> None:
        """创建可视化窗口与内部 Visualizer（可选双窗：侧面 + 正面）。"""
        self._wireframe_head = wireframe_head
        self._dual = dual_view
        self._visualizers: list[o3d.visualization.Visualizer] = []
        titles = (
            [window_name]
            if not dual_view
            else [f"{window_name} (侧)", f"{window_name} (正)"]
        )
        for i, title in enumerate(titles):
            vis = o3d.visualization.Visualizer()
            if dual_view:
                left = _DUAL_VIEW_LEFT0 + i * (_DUAL_VIEW_WIDTH + _DUAL_VIEW_GAP)
                vis.create_window(
                    window_name=title,
                    width=_DUAL_VIEW_WIDTH,
                    height=_DUAL_VIEW_HEIGHT,
                    left=left,
                    top=_DUAL_VIEW_TOP,
                )  # type: ignore[call-arg]
            else:
                vis.create_window(window_name=title)
            self._visualizers.append(vis)

        self._cam_front_up: list[tuple[NDArray[np.float64], NDArray[np.float64]]] = (
            [(_SIDE_FRONT, _SIDE_UP)]
            if not dual_view
            else [(_SIDE_FRONT, _SIDE_UP), (_FRONT_VIEW_FRONT, _FRONT_VIEW_UP)]
        )

        self._mesh: list[o3d.geometry.TriangleMesh] | None = None
        self._line_set: list[o3d.geometry.LineSet] | None = None
        self._tongue_mesh: list[o3d.geometry.TriangleMesh] | None = None
        self._camera_initialized = False

    def _setup_view_control(
        self,
        visualizer: o3d.visualization.Visualizer,
        lookat_points: NDArray[np.float64],
        *,
        front: NDArray[np.float64],
        up: NDArray[np.float64],
    ) -> None:
        """首次刷新时根据采样点质心设置相机与缩放。

        :param lookat_points: 用于计算观察目标的点集，形状 (N, 3)。
        """
        ctr = visualizer.get_view_control()
        lookat = np.asarray(lookat_points.mean(axis=0), dtype=np.float64).reshape(3, 1)
        ctr.set_lookat(lookat)
        ctr.set_front(front)
        ctr.set_up(up)
        ctr.set_zoom(0.6)

    def _init_cameras(self, lookat_points: NDArray[np.float64]) -> None:
        for vis, (front, up) in zip(self._visualizers, self._cam_front_up, strict=True):
            self._setup_view_control(vis, lookat_points, front=front, up=up)
        self._camera_initialized = True

    def _poll_all(self) -> None:
        for vis in self._visualizers:
            vis.poll_events()
            vis.update_renderer()

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
            line_sets: list[o3d.geometry.LineSet] = []
            tongue_meshes: list[o3d.geometry.TriangleMesh] = []
            for vis in self._visualizers:
                ls = o3d.geometry.LineSet()
                ls.points = o3d.utility.Vector3dVector(shell_vertices)
                ls.lines = o3d.utility.Vector2iVector(
                    shell_edges.astype(np.int32, copy=False)
                )
                ls.colors = o3d.utility.Vector3dVector(
                    np.tile(WIREFRAME_LINE_COLOR, (len(shell_edges), 1))
                )
                tm = o3d.geometry.TriangleMesh()
                tm.vertices = o3d.utility.Vector3dVector(tongue_vertices)
                tm.triangles = o3d.utility.Vector3iVector(
                    tongue_faces.astype(np.int32, copy=False)
                )
                tm.vertex_colors = o3d.utility.Vector3dVector(
                    np.tile(SKIN_TONE, (len(tongue_vertices), 1))
                )
                tm.compute_vertex_normals()
                vis.add_geometry(ls)
                vis.add_geometry(tm)
                ro = vis.get_render_option()
                ro.line_width = 2.0
                line_sets.append(ls)
                tongue_meshes.append(tm)
            self._line_set = line_sets
            self._tongue_mesh = tongue_meshes
            if not self._camera_initialized:
                combined = np.concatenate([shell_vertices, tongue_vertices], axis=0)
                self._init_cameras(combined)
        else:
            assert self._line_set is not None and self._tongue_mesh is not None
            for ls, tm in zip(self._line_set, self._tongue_mesh, strict=True):
                ls.points = o3d.utility.Vector3dVector(shell_vertices)
                tm.vertices = o3d.utility.Vector3dVector(tongue_vertices)
                tm.compute_vertex_normals()
            for vis, ls, tm in zip(
                self._visualizers, self._line_set, self._tongue_mesh, strict=True
            ):
                vis.update_geometry(ls)
                vis.update_geometry(tm)

        self._poll_all()

    def _update_solid_mesh(
        self,
        vertices: NDArray[np.float64],
        faces: NDArray[np.int64],
    ) -> None:
        """整头实体三角网格模式下的顶点与面刷新（固定默认肤色）。"""
        if self._mesh is None:
            meshes: list[o3d.geometry.TriangleMesh] = []
            for vis in self._visualizers:
                m = o3d.geometry.TriangleMesh()
                m.vertices = o3d.utility.Vector3dVector(vertices)
                m.triangles = o3d.utility.Vector3iVector(faces)
                m.vertex_colors = o3d.utility.Vector3dVector(
                    np.tile(SKIN_TONE, (len(vertices), 1))
                )
                m.compute_vertex_normals()
                vis.add_geometry(m)
                meshes.append(m)
            self._mesh = meshes
            if not self._camera_initialized:
                self._init_cameras(vertices)
        else:
            assert self._mesh is not None
            for m in self._mesh:
                m.vertices = o3d.utility.Vector3dVector(vertices)
                m.triangles = o3d.utility.Vector3iVector(faces)
                m.compute_vertex_normals()
            for vis, m in zip(self._visualizers, self._mesh, strict=True):
                vis.update_geometry(m)

        self._poll_all()

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
