"""基于 Open3D 的三角网格预览窗口：顶点色、烘焙光照、相机缩放等与面部网格显示相关的封装。"""

from __future__ import annotations

from typing import Literal

import numpy as np
import open3d as o3d

CameraProfile = Literal["front", "side"]

from face_blender_shape.constants import (
    DEFAULT_OPEN3D_BAKED_SHADING,
    DEFAULT_OPEN3D_HEIGHT,
    DEFAULT_OPEN3D_SKIN_CREASE_STRENGTH,
    DEFAULT_OPEN3D_SKIN_DETAIL_ENABLED,
    DEFAULT_OPEN3D_SKIN_MICRO_FREQ,
    DEFAULT_OPEN3D_SKIN_MICRO_STRENGTH,
    DEFAULT_OPEN3D_WINDOW_NAME,
    DEFAULT_OPEN3D_WIDTH,
    DEFAULT_VIEW_SCALE,
)
from face_blender_shape.visual_presets import get_open3d_viewer_tune

# Slightly warmer, less saturated — reads less “wax figure” under simple lighting.
SKIN_TONE = np.array([0.80, 0.69, 0.62])

# Open3D: larger set_zoom() → camera farther (object smaller).
# SRanipal meshes use ~10–40 unit extents; the old 0.5/max_dim heuristic still works.
# MetaHuman / meter-scale heads have max_dim ≪ 10; same formula pushed the camera too far.
_LARGE_MESH_THRESHOLD = 5.0
_SMALL_MESH_ZOOM_COEFF = 0.52


def _vertex_neighbor_sets(vertex_count: int, faces: np.ndarray) -> list[set[int]]:
    """为每个顶点收集一环邻接顶点索引（用于法线不一致度近似褶皱）。

    vertex_count: 顶点个数。
    faces: 三角面索引，形状 (F, 3)。
    """
    neigh: list[set[int]] = [set() for _ in range(vertex_count)]
    for a, b, c in faces:
        ia, ib, ic = int(a), int(b), int(c)
        neigh[ia].update((ib, ic))
        neigh[ib].update((ia, ic))
        neigh[ic].update((ia, ib))
    return neigh


def _skin_detail_modulate_albedo(
    vertices: np.ndarray,
    faces: np.ndarray,
    rgb: np.ndarray,
    *,
    crease_strength: float,
    micro_strength: float,
    micro_freq: float,
) -> np.ndarray:
    """在 albedo 上叠加以法线变化为主的沟壑加深与弱空间噪声（模拟毛孔级起伏）。

    vertices: 顶点坐标 (N, 3)。
    faces: 三角索引 (F, 3)。
    rgb: 输入 RGB，形状 (N, 3)。
    crease_strength: 邻域法线不一致时的变暗幅度，0 关闭沟壑感。
    micro_strength: 位置正弦乘积噪声强度，0 关闭微噪。
    micro_freq: 噪声空间频率（与模型尺度成反比调节）。
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    out = np.clip(np.asarray(rgb, dtype=np.float64), 0.0, 1.0).copy()
    n_verts = v.shape[0]
    if n_verts == 0 or f.size == 0:
        return out

    tmp = o3d.geometry.TriangleMesh()
    tmp.vertices = o3d.utility.Vector3dVector(v)
    tmp.triangles = o3d.utility.Vector3iVector(f)
    tmp.compute_vertex_normals()
    normals = np.asarray(tmp.vertex_normals, dtype=np.float64)
    neigh = _vertex_neighbor_sets(n_verts, f)
    crease = np.zeros(n_verts, dtype=np.float64)
    for i in range(n_verts):
        js = neigh[i]
        if not js:
            continue
        ni = normals[i]
        acc = 0.0
        for j in js:
            acc += max(0.0, 1.0 - float(np.dot(ni, normals[j])))
        crease[i] = acc / len(js)
    q = float(np.quantile(crease, 0.92)) + 1e-6
    crease = np.clip(crease / q, 0.0, 1.0)
    out *= np.clip(1.0 - crease_strength * crease[:, np.newaxis], 0.55, 1.0)

    if micro_strength > 0.0:
        p = v * float(micro_freq)
        h = np.sin(p[:, 0]) * np.sin(p[:, 1] * 1.07) * np.sin(p[:, 2] * 0.93)
        h = (h * 0.5 + 0.5).reshape(-1, 1)
        out *= np.clip(1.0 + micro_strength * (h - 0.5) * 2.0, 0.88, 1.12)

    return np.clip(out, 0.0, 1.0)


class Open3DMeshViewer:
    def __init__(
        self,
        window_name: str = DEFAULT_OPEN3D_WINDOW_NAME,
        *,
        view_scale: float = DEFAULT_VIEW_SCALE,
        window_width: int = DEFAULT_OPEN3D_WIDTH,
        window_height: int = DEFAULT_OPEN3D_HEIGHT,
        camera_profile: CameraProfile = "front",
    ) -> None:
        """
        camera_profile: front 为正脸；side 为侧视（沿 +X 看向面部，适用于 MetaHuman Z-up 与 SRanipal Y-up 的粗分类）。
        """
        self._camera_profile: CameraProfile = camera_profile
        self._view_scale = max(float(view_scale), 0.05)
        _tune = get_open3d_viewer_tune()
        self._background_rgb = tuple(float(x) for x in _tune.background_rgb)
        self._matte_gamma = float(_tune.matte_gamma)
        self._baked_ambient = float(_tune.baked_ambient)
        self._baked_diffuse = float(_tune.baked_diffuse)
        self._visualizer = o3d.visualization.Visualizer()
        self._visualizer.create_window(
            window_name=window_name,
            width=int(window_width),
            height=int(window_height),
        )
        self._mesh: o3d.geometry.TriangleMesh | None = None
        self._camera_initialized = False
        self._baked_shading = bool(DEFAULT_OPEN3D_BAKED_SHADING)
        self._skin_detail_enabled = bool(DEFAULT_OPEN3D_SKIN_DETAIL_ENABLED)
        self._skin_crease_strength = float(DEFAULT_OPEN3D_SKIN_CREASE_STRENGTH)
        self._skin_micro_strength = float(DEFAULT_OPEN3D_SKIN_MICRO_STRENGTH)
        self._skin_micro_freq = float(DEFAULT_OPEN3D_SKIN_MICRO_FREQ)
        self._apply_friendly_render_settings()

    def _apply_friendly_render_settings(self) -> None:
        """Less harsh than Open3D defaults: smooth shading, soft backdrop, vertex colors on."""
        ro = self._visualizer.get_render_option()
        ro.mesh_color_option = o3d.visualization.MeshColorOption.Color
        ro.mesh_shade_option = o3d.visualization.MeshShadeOption.Color
        ro.background_color = np.asarray(self._background_rgb, dtype=np.float64)
        # Baked shading uses vertex colors as final display color; scene lights add plastic specular.
        ro.light_on = not self._baked_shading
        ro.mesh_show_wireframe = False
        ro.show_coordinate_frame = False

    def _setup_camera(self, vertices: np.ndarray) -> None:
        """将相机对准网格中心；正脸或侧视由 camera_profile 决定。"""
        center = vertices.mean(axis=0)
        extent = vertices.max(axis=0) - vertices.min(axis=0)
        max_dim = float(np.max(extent))

        vc = self._visualizer.get_view_control()

        z_dominant = float(extent[2]) > float(extent[1])
        if self._camera_profile == "side":
            # 侧视：从 +X 看向原点，便于观察嘴部与轮廓
            vc.set_front([1.0, 0.0, 0.0])
            vc.set_up([0.0, 0.0, 1.0] if z_dominant else [0.0, 1.0, 0.0])
        elif z_dominant:
            # MetaHuman: Z-up, face toward -Y
            vc.set_front([0, -1, 0])
            vc.set_up([0, 0, 1])
        else:
            # SRanipal: Y-up, face toward +Z
            vc.set_front([0, 0, 1])
            vc.set_up([0, 1, 0])

        vc.set_lookat(center)
        # view_scale > 1 → smaller zoom → closer camera (face fills more of the window).
        if max_dim >= _LARGE_MESH_THRESHOLD:
            # SRanipal-scale meshes: keep legacy framing (unchanged from early versions).
            zoom = 0.5 / max(max_dim, 1e-6)
        else:
            # Meter-scale heads (e.g. MetaHuman): proportional zoom; 0.5/max_dim was too large → tiny face.
            zoom = (_SMALL_MESH_ZOOM_COEFF * max(max_dim, 1e-6)) / self._view_scale
        vc.set_zoom(zoom)

    def _matte_vertex_colors(self, colors: np.ndarray) -> np.ndarray:
        """Darken bright regions — Open3D has no roughness control; this tames blown highlights."""
        x = np.clip(np.asarray(colors, dtype=np.float64), 0.0, 1.0)
        return np.clip(np.power(x, self._matte_gamma), 0.0, 1.0)

    def _light_dir_for_mesh(self, vertices: np.ndarray) -> np.ndarray:
        """Approximate key light direction in world space (matches Z-up vs Y-up heuristics)."""
        extent = vertices.max(axis=0) - vertices.min(axis=0)
        if float(extent[2]) > float(extent[1]):
            # Z-up (e.g. MetaHuman): light from upper-front
            L = np.array([0.28, 0.62, 0.74], dtype=np.float64)
        else:
            # Y-up (e.g. SRanipal): face toward +Z
            L = np.array([0.38, 0.72, 0.58], dtype=np.float64)
        L /= np.linalg.norm(L) + 1e-9
        return L

    def _bake_half_lambert_shading(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        albedo: np.ndarray,
    ) -> np.ndarray:
        """Half-Lambert + tiny warm fill in shadows — reads more like skin than flat + spec spike."""
        tmp = o3d.geometry.TriangleMesh()
        tmp.vertices = o3d.utility.Vector3dVector(vertices)
        tmp.triangles = o3d.utility.Vector3iVector(faces)
        tmp.compute_vertex_normals()
        n = np.asarray(tmp.vertex_normals, dtype=np.float64)
        L = self._light_dir_for_mesh(vertices)
        nd = (n * L).sum(axis=1, keepdims=True)
        half = np.clip(0.5 * nd + 0.5, 0.0, 1.0)
        lit = self._baked_ambient + self._baked_diffuse * half
        rgb = np.clip(np.asarray(albedo, dtype=np.float64), 0.0, 1.0) * lit
        # Slight warmth in shadow (very subtle SSS hint)
        shadow = np.clip(1.0 - lit, 0.0, 1.0)
        warm = np.concatenate(
            [0.04 * shadow, 0.015 * shadow, 0.012 * shadow],
            axis=1,
        )
        return np.clip(rgb + warm, 0.0, 1.0)

    def update(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        *,
        vertex_colors: np.ndarray | None = None,
    ) -> None:
        if vertex_colors is not None:
            display_colors = np.clip(np.asarray(vertex_colors, dtype=np.float64), 0.0, 1.0)
        else:
            display_colors = np.tile(np.asarray(SKIN_TONE, dtype=np.float64), (len(vertices), 1))

        if (
            self._skin_detail_enabled
            and len(vertices) > 0
            and len(faces) > 0
            and (self._skin_crease_strength > 0.0 or self._skin_micro_strength > 0.0)
        ):
            display_colors = _skin_detail_modulate_albedo(
                vertices,
                faces,
                display_colors,
                crease_strength=self._skin_crease_strength,
                micro_strength=self._skin_micro_strength,
                micro_freq=self._skin_micro_freq,
            )

        display_colors = self._matte_vertex_colors(display_colors)

        if self._baked_shading and len(vertices) > 0 and len(faces) > 0:
            display_colors = self._bake_half_lambert_shading(vertices, faces, display_colors)

        if self._mesh is None:
            self._mesh = o3d.geometry.TriangleMesh()
            self._mesh.vertices = o3d.utility.Vector3dVector(vertices)
            self._mesh.triangles = o3d.utility.Vector3iVector(faces)

            self._mesh.vertex_colors = o3d.utility.Vector3dVector(display_colors)

            self._mesh.compute_vertex_normals()
            self._visualizer.add_geometry(self._mesh)
        else:
            self._mesh.vertices = o3d.utility.Vector3dVector(vertices)
            self._mesh.triangles = o3d.utility.Vector3iVector(faces)
            self._mesh.vertex_colors = o3d.utility.Vector3dVector(display_colors)
            self._mesh.compute_vertex_normals()
            self._visualizer.update_geometry(self._mesh)

        if not self._camera_initialized:
            self._setup_camera(vertices)
            self._camera_initialized = True

        self._visualizer.poll_events()
        self._visualizer.update_renderer()
