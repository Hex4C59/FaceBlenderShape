from __future__ import annotations

import numpy as np
import open3d as o3d

from face_blender_shape.constants import (
    DEFAULT_OPEN3D_HEIGHT,
    DEFAULT_OPEN3D_WINDOW_NAME,
    DEFAULT_OPEN3D_WIDTH,
    DEFAULT_VIEW_SCALE,
)

SKIN_TONE = np.array([0.87, 0.73, 0.62])

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

    def update(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        *,
        vertex_colors: np.ndarray | None = None,
    ) -> None:
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

            self._mesh.compute_vertex_normals()
            self._visualizer.add_geometry(self._mesh)
        else:
            self._mesh.vertices = o3d.utility.Vector3dVector(vertices)
            self._mesh.triangles = o3d.utility.Vector3iVector(faces)
            if vertex_colors is not None:
                self._mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
            self._mesh.compute_vertex_normals()
            self._visualizer.update_geometry(self._mesh)

        if not self._camera_initialized:
            self._setup_camera(vertices)
            self._camera_initialized = True

        self._visualizer.poll_events()
        self._visualizer.update_renderer()
