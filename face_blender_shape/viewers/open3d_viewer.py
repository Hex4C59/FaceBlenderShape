from __future__ import annotations

import numpy as np
import open3d as o3d

from face_blender_shape.constants import (
    DEFAULT_OPEN3D_BACKGROUND_RGB,
    DEFAULT_OPEN3D_BAKED_AMBIENT,
    DEFAULT_OPEN3D_BAKED_DIFFUSE,
    DEFAULT_OPEN3D_BAKED_SHADING,
    DEFAULT_OPEN3D_HEIGHT,
    DEFAULT_OPEN3D_VERTEX_MATTE_GAMMA,
    DEFAULT_OPEN3D_WINDOW_NAME,
    DEFAULT_OPEN3D_WIDTH,
    DEFAULT_VIEW_SCALE,
)

# Slightly warmer, less saturated — reads less “wax figure” under simple lighting.
SKIN_TONE = np.array([0.80, 0.69, 0.62])

# Open3D: larger set_zoom() → camera farther (object smaller).
# SRanipal meshes use ~10–40 unit extents; the old 0.5/max_dim heuristic still works.
# MetaHuman / meter-scale heads have max_dim ≪ 10; same formula pushed the camera too far.
_LARGE_MESH_THRESHOLD = 5.0
_SMALL_MESH_ZOOM_COEFF = 0.52


class Open3DMeshViewer:
    def __init__(
        self,
        window_name: str = DEFAULT_OPEN3D_WINDOW_NAME,
        *,
        view_scale: float = DEFAULT_VIEW_SCALE,
        window_width: int = DEFAULT_OPEN3D_WIDTH,
        window_height: int = DEFAULT_OPEN3D_HEIGHT,
    ) -> None:
        self._view_scale = max(float(view_scale), 0.05)
        self._visualizer = o3d.visualization.Visualizer()
        self._visualizer.create_window(
            window_name=window_name,
            width=int(window_width),
            height=int(window_height),
        )
        self._mesh: o3d.geometry.TriangleMesh | None = None
        self._camera_initialized = False
        self._matte_gamma = float(DEFAULT_OPEN3D_VERTEX_MATTE_GAMMA)
        self._baked_shading = bool(DEFAULT_OPEN3D_BAKED_SHADING)
        self._baked_ambient = float(DEFAULT_OPEN3D_BAKED_AMBIENT)
        self._baked_diffuse = float(DEFAULT_OPEN3D_BAKED_DIFFUSE)
        self._apply_friendly_render_settings()

    def _apply_friendly_render_settings(self) -> None:
        """Less harsh than Open3D defaults: smooth shading, soft backdrop, vertex colors on."""
        ro = self._visualizer.get_render_option()
        ro.mesh_color_option = o3d.visualization.MeshColorOption.Color
        ro.mesh_shade_option = o3d.visualization.MeshShadeOption.Color
        ro.background_color = np.asarray(DEFAULT_OPEN3D_BACKGROUND_RGB, dtype=np.float64)
        # Baked shading uses vertex colors as final display color; scene lights add plastic specular.
        ro.light_on = not self._baked_shading
        ro.mesh_show_wireframe = False
        ro.show_coordinate_frame = False

    def _setup_camera(self, vertices: np.ndarray) -> None:
        """Point the camera at the mesh center, looking from the front."""
        center = vertices.mean(axis=0)
        extent = vertices.max(axis=0) - vertices.min(axis=0)
        max_dim = float(np.max(extent))

        vc = self._visualizer.get_view_control()

        # MetaHuman: Z-up, face toward -Y → front = [0, -1, 0], up = [0, 0, 1]
        # SRanipal:  Y-up, face toward +Z → front = [0, 0, 1],  up = [0, 1, 0]
        if extent[2] > extent[1]:
            vc.set_front([0, -1, 0])
            vc.set_up([0, 0, 1])
        else:
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
            display_colors = self._matte_vertex_colors(vertex_colors)
        else:
            display_colors = self._matte_vertex_colors(np.tile(SKIN_TONE, (len(vertices), 1)))

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
