"""基于 Open3D 的三角网格预览窗口。"""

from __future__ import annotations

from typing import Literal

import numpy as np
import open3d as o3d

from face_blender_shape.core.viewer_defaults import (
    DEFAULT_OPEN3D_BAKED_SHADING,
    DEFAULT_OPEN3D_HEIGHT,
    DEFAULT_OPEN3D_SKIN_CREASE_STRENGTH,
    DEFAULT_OPEN3D_SKIN_DETAIL_ENABLED,
    DEFAULT_OPEN3D_SKIN_MICRO_FREQ,
    DEFAULT_OPEN3D_SKIN_MICRO_STRENGTH,
    DEFAULT_OPEN3D_WIDTH,
    DEFAULT_OPEN3D_WINDOW_NAME,
    DEFAULT_VIEW_SCALE,
)
from face_blender_shape.viewer.shading import SKIN_TONE, prepare_display_colors
from face_blender_shape.viewer.visual_presets import get_open3d_viewer_tune

CameraProfile = Literal["front", "side"]

_LARGE_MESH_THRESHOLD = 5.0
_SMALL_MESH_ZOOM_COEFF = 0.52


class Open3DMeshViewer:
    """封装 Open3D 网格窗口与相机控制。"""

    def __init__(
        self,
        window_name: str = DEFAULT_OPEN3D_WINDOW_NAME,
        *,
        view_scale: float = DEFAULT_VIEW_SCALE,
        window_width: int = DEFAULT_OPEN3D_WIDTH,
        window_height: int = DEFAULT_OPEN3D_HEIGHT,
        camera_profile: CameraProfile = "front",
    ) -> None:
        """初始化预览窗口。

        window_name: 窗口标题。
        view_scale: 取景比例。
        window_width: 窗口宽度。
        window_height: 窗口高度。
        camera_profile: 相机朝向配置。
        """
        tune = get_open3d_viewer_tune()
        self._camera_profile: CameraProfile = camera_profile
        self._view_scale = max(float(view_scale), 0.05)
        self._background_rgb = tuple(float(x) for x in tune.background_rgb)
        self._matte_gamma = float(tune.matte_gamma)
        self._baked_ambient = float(tune.baked_ambient)
        self._baked_diffuse = float(tune.baked_diffuse)
        self._baked_shading = bool(DEFAULT_OPEN3D_BAKED_SHADING)
        self._skin_detail_enabled = bool(DEFAULT_OPEN3D_SKIN_DETAIL_ENABLED)
        self._skin_crease_strength = float(DEFAULT_OPEN3D_SKIN_CREASE_STRENGTH)
        self._skin_micro_strength = float(DEFAULT_OPEN3D_SKIN_MICRO_STRENGTH)
        self._skin_micro_freq = float(DEFAULT_OPEN3D_SKIN_MICRO_FREQ)
        self._mesh: o3d.geometry.TriangleMesh | None = None
        self._camera_initialized = False
        self._visualizer = o3d.visualization.Visualizer()
        self._visualizer.create_window(
            window_name=window_name,
            width=int(window_width),
            height=int(window_height),
        )
        self._apply_friendly_render_settings()

    def _apply_friendly_render_settings(self) -> None:
        """设置更柔和的 Open3D 渲染选项。"""
        render_option = self._visualizer.get_render_option()
        render_option.mesh_color_option = o3d.visualization.MeshColorOption.Color
        render_option.mesh_shade_option = o3d.visualization.MeshShadeOption.Color
        render_option.background_color = np.asarray(self._background_rgb, dtype=np.float64)
        render_option.light_on = not self._baked_shading
        render_option.mesh_show_wireframe = False
        render_option.show_coordinate_frame = False

    def _setup_camera(self, vertices: np.ndarray) -> None:
        """根据网格尺寸与朝向设置相机。"""
        center = vertices.mean(axis=0)
        extent = vertices.max(axis=0) - vertices.min(axis=0)
        max_dim = float(np.max(extent))
        view_control = self._visualizer.get_view_control()
        z_dominant = float(extent[2]) > float(extent[1])

        if self._camera_profile == "side":
            view_control.set_front([1.0, 0.0, 0.0])
            view_control.set_up([0.0, 0.0, 1.0] if z_dominant else [0.0, 1.0, 0.0])
        elif z_dominant:
            view_control.set_front([0.0, -1.0, 0.0])
            view_control.set_up([0.0, 0.0, 1.0])
        else:
            view_control.set_front([0.0, 0.0, 1.0])
            view_control.set_up([0.0, 1.0, 0.0])

        view_control.set_lookat(center)

        if max_dim >= _LARGE_MESH_THRESHOLD:
            zoom = 0.5 / max(max_dim, 1e-6)
        else:
            zoom = (_SMALL_MESH_ZOOM_COEFF * max(max_dim, 1e-6)) / self._view_scale

        view_control.set_zoom(zoom)

    def _build_display_colors(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        vertex_colors: np.ndarray | None,
    ) -> np.ndarray:
        """构建提交到 Open3D 的最终顶点色。"""
        return prepare_display_colors(
            vertices,
            faces,
            vertex_colors=vertex_colors,
            skin_detail_enabled=self._skin_detail_enabled,
            skin_crease_strength=self._skin_crease_strength,
            skin_micro_strength=self._skin_micro_strength,
            skin_micro_freq=self._skin_micro_freq,
            matte_gamma=self._matte_gamma,
            baked_shading=self._baked_shading,
            baked_ambient=self._baked_ambient,
            baked_diffuse=self._baked_diffuse,
        )

    def _commit_mesh(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        colors: np.ndarray,
    ) -> None:
        """把几何与颜色提交到 Open3D 网格对象。"""
        if self._mesh is None:
            self._mesh = o3d.geometry.TriangleMesh()
            self._visualizer.add_geometry(self._mesh)

        self._mesh.vertices = o3d.utility.Vector3dVector(vertices)
        self._mesh.triangles = o3d.utility.Vector3iVector(faces)
        self._mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        self._mesh.compute_vertex_normals()
        self._visualizer.update_geometry(self._mesh)

    def _ensure_camera(self, vertices: np.ndarray) -> None:
        """确保相机仅在首次更新时初始化一次。"""
        if self._camera_initialized:
            return
        if len(vertices) == 0:
            return
        self._setup_camera(vertices)
        self._camera_initialized = True

    def _refresh_window(self) -> None:
        """刷新窗口事件与渲染器。"""
        self._visualizer.poll_events()
        self._visualizer.update_renderer()

    def update(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        *,
        vertex_colors: np.ndarray | None = None,
    ) -> None:
        """更新窗口中的网格内容。

        vertices: 顶点坐标数组。
        faces: 三角面索引数组。
        vertex_colors: 可选顶点色数组。
        """
        colors = self._build_display_colors(vertices, faces, vertex_colors)
        self._commit_mesh(vertices, faces, colors)
        self._ensure_camera(vertices)
        self._refresh_window()
