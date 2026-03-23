from __future__ import annotations

import numpy as np
import open3d as o3d

from face_blender_shape.constants import DEFAULT_OPEN3D_WINDOW_NAME

SKIN_TONE = np.array([0.87, 0.73, 0.62])


class Open3DMeshViewer:
    def __init__(self, window_name: str = DEFAULT_OPEN3D_WINDOW_NAME) -> None:
        self._visualizer = o3d.visualization.Visualizer()
        self._visualizer.create_window(window_name=window_name)
        self._mesh: o3d.geometry.TriangleMesh | None = None

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
            self._mesh.compute_vertex_normals()
            self._visualizer.update_geometry(self._mesh)

        self._visualizer.poll_events()
        self._visualizer.update_renderer()
