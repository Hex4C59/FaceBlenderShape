from __future__ import annotations

import bpy
import bmesh
import numpy as np

from face_blender_shape.constants import (
    BLENDSHAPE_NAMES,
    DEFAULT_HEAD_OBJECT_NAME,
    DEFAULT_OPEN3D_WINDOW_NAME,
    FRAME_WIDTH,
    METAHUMAN_HEAD_OBJECT_NAME,
    METAHUMAN_TEETH_OBJECT_NAME,
)
from face_blender_shape.landmarks import (
    extract_default_landmarks,
    get_cheek_keypoints,
    get_cheek_vertices,
    get_lip_vertices,
    get_tongue_tip,
    get_tongue_vertices,
)
from face_blender_shape.paths import (
    METAHUMAN_FBX_PATH,
    resolve_fbx_path,
    resolve_texture_path,
)
from face_blender_shape.viewers.open3d_viewer import Open3DMeshViewer


class FaceBlenderRuntime:
    """Blendshape-driven face mesh viewer.

    Supports two model backends:
    - ``"sranipal"`` (default): Original SRanipal head with 37 blendshapes.
    - ``"metahuman"``: MetaHuman head with 52 ARKit blendshapes.
      SRanipal CSV input is automatically converted via the mapping layer.
    """

    def __init__(
        self,
        path: str | None = None,
        *,
        enable_viewer: bool = True,
        window_name: str = DEFAULT_OPEN3D_WINDOW_NAME,
        head_object_name: str | None = None,
        texture_path: str | None = None,
        model: str = "sranipal",
    ) -> None:
        self._model = model

        if model == "metahuman":
            from face_blender_shape.blendshape_mapping import (
                ARKIT_SHAPE_NAMES,
                convert_sranipal_to_arkit,
            )
            self._arkit_names = np.array(ARKIT_SHAPE_NAMES)
            self._convert_frame = convert_sranipal_to_arkit
            path = path or str(METAHUMAN_FBX_PATH)
            head_object_name = head_object_name or METAHUMAN_HEAD_OBJECT_NAME
        else:
            self._arkit_names = None
            self._convert_frame = None
            head_object_name = head_object_name or DEFAULT_HEAD_OBJECT_NAME

        self.blendshape_names = np.array(BLENDSHAPE_NAMES)
        self.load_fbx(path)
        self.set_active_object(head_object_name)

        if model == "metahuman":
            self._teeth_obj = bpy.data.objects.get(METAHUMAN_TEETH_OBJECT_NAME)
        else:
            self._teeth_obj = None

        if model == "metahuman" and texture_path is None:
            self._texture_image = None
        else:
            self._texture_image = self._load_texture(texture_path)
        self._triangle_uvs: np.ndarray | None = None
        self._vertex_colors: np.ndarray | None = None

        self.viewer = (
            Open3DMeshViewer(window_name=window_name)
            if enable_viewer
            else None
        )

    def load_fbx(self, path: str | None) -> None:
        self.fbx_path = resolve_fbx_path(path)
        bpy.ops.object.select_all(action="DESELECT")
        bpy.ops.import_scene.fbx(filepath=str(self.fbx_path))

    def set_active_object(self, object_name: str = DEFAULT_HEAD_OBJECT_NAME) -> None:
        self.active_obj = bpy.data.objects[object_name]
        bpy.context.view_layer.objects.active = self.active_obj

    # ------------------------------------------------------------------
    # Texture loading
    # ------------------------------------------------------------------

    def _load_texture(self, texture_path: str | None) -> np.ndarray | None:
        """Load albedo texture via bpy embedded data or an external image file."""
        img_array = self._extract_bpy_texture()
        if img_array is not None:
            return img_array

        resolved = resolve_texture_path(texture_path)
        if resolved is None:
            return None

        return self._load_external_texture(resolved)

    def _extract_bpy_texture(self) -> np.ndarray | None:
        obj = self.active_obj
        if not obj.data.materials:
            return None
        for mat in obj.data.materials:
            if mat is None or not mat.use_nodes:
                continue
            for node in mat.node_tree.nodes:
                if node.type == "TEX_IMAGE" and node.image is not None:
                    img = node.image
                    if img.size[0] == 0 or img.size[1] == 0:
                        continue
                    return self._bpy_image_to_numpy(img)
        return None

    def _load_external_texture(self, path) -> np.ndarray | None:
        path_str = str(path)
        for img in bpy.data.images:
            if img.name in ("Render Result", "Viewer Node"):
                continue
            img.filepath = path_str
            img.reload()
            if img.size[0] > 0 and img.size[1] > 0:
                return self._bpy_image_to_numpy(img)

        img = bpy.data.images.load(path_str)
        if img.size[0] > 0 and img.size[1] > 0:
            return self._bpy_image_to_numpy(img)
        return None

    @staticmethod
    def _bpy_image_to_numpy(img, max_size: int = 2048) -> np.ndarray:
        w, h = img.size
        channels = img.channels
        pixels = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, channels)
        pixels = np.flipud(pixels)
        if channels >= 4:
            pixels = pixels[:, :, :3]
        rgb = np.clip(pixels * 255, 0, 255).astype(np.uint8)

        if max(h, w) > max_size:
            from PIL import Image as PILImage

            pil_img = PILImage.fromarray(rgb)
            pil_img.thumbnail((max_size, max_size), PILImage.LANCZOS)
            rgb = np.array(pil_img)

        return np.ascontiguousarray(rgb)

    # ------------------------------------------------------------------
    # Texture → vertex color baking
    # ------------------------------------------------------------------

    @staticmethod
    def _bake_vertex_colors(
        texture: np.ndarray,
        triangle_uvs: np.ndarray,
        faces: np.ndarray,
        n_vertices: int,
    ) -> np.ndarray:
        h, w, _ = texture.shape
        flat_vert_idx = faces.ravel()
        us = np.clip(triangle_uvs[:, 0], 0.0, 1.0)
        vs = np.clip(triangle_uvs[:, 1], 0.0, 1.0)
        px = (us * (w - 1)).astype(int)
        py = ((1.0 - vs) * (h - 1)).astype(int)

        sampled = texture[py, px].astype(np.float64) / 255.0
        colors = np.zeros((n_vertices, 3), dtype=np.float64)
        counts = np.zeros(n_vertices, dtype=np.float64)
        np.add.at(colors, flat_vert_idx, sampled)
        np.add.at(counts, flat_vert_idx, 1.0)
        colors /= np.maximum(counts, 1.0)[:, None]
        return colors

    def _ensure_vertex_colors(self, faces: np.ndarray, n_vertices: int) -> None:
        if self._vertex_colors is not None:
            return
        if self._texture_image is not None and self._triangle_uvs is not None:
            self._vertex_colors = self._bake_vertex_colors(
                self._texture_image, self._triangle_uvs, faces, n_vertices,
            )

    # ------------------------------------------------------------------
    # UV extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_triangle_uvs(mesh) -> np.ndarray | None:
        if not mesh.uv_layers:
            return None
        uv_layer = mesh.uv_layers.active or mesh.uv_layers[0]
        return np.array([tuple(d.uv) for d in uv_layer.data], dtype=float)

    # ------------------------------------------------------------------
    # Core blendshape / mesh pipeline
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_frame(blendshapes: np.ndarray | list[float]) -> np.ndarray:
        frame = np.asarray(blendshapes, dtype=float).reshape(-1)
        if frame.size != FRAME_WIDTH:
            raise ValueError(f"Expected {FRAME_WIDTH} blendshape values, got {frame.size}")
        return frame

    def _apply_arkit_shapes(self, arkit_values: np.ndarray) -> None:
        """Set ARKit blendshape weights on the active object."""
        key_blocks = self.active_obj.data.shape_keys.key_blocks
        for name, value in zip(self._arkit_names, arkit_values):
            if name in key_blocks:
                key_blocks[name].value = float(value)

    def _apply_teeth_shapes(self, arkit_values: np.ndarray) -> None:
        """Mirror jaw/mouth shapes onto the teeth object."""
        if self._teeth_obj is None:
            return
        teeth_keys = self._teeth_obj.data.shape_keys
        if teeth_keys is None:
            return
        key_blocks = teeth_keys.key_blocks
        for name, value in zip(self._arkit_names, arkit_values):
            if name in key_blocks:
                key_blocks[name].value = float(value)

    def set_blendshapes(self, blendshapes: np.ndarray | list[float]):
        frame = self._validate_frame(blendshapes)
        bpy.context.view_layer.objects.active = self.active_obj
        bpy.context.object.update_from_editmode()

        if self._model == "metahuman":
            arkit_values = self._convert_frame(frame)
            self._apply_arkit_shapes(arkit_values)
            self._apply_teeth_shapes(arkit_values)
        else:
            for heading, value in zip(self.blendshape_names, frame):
                self.active_obj.data.shape_keys.key_blocks[heading].value = float(value)

        obj = bpy.context.object.copy()
        mesh = self.get_modified_mesh(self.active_obj)

        if self._triangle_uvs is None:
            self._triangle_uvs = self._extract_triangle_uvs(mesh)

        obj.modifiers.clear()
        obj.data = mesh
        return obj

    def get_modified_mesh(self, obj, cage: bool = False):
        bm = bmesh.new()
        bm.from_object(
            obj,
            bpy.context.evaluated_depsgraph_get(),
            cage=cage,
        )
        mesh = bpy.data.meshes.new("Deformed")
        bmesh.ops.triangulate(bm, faces=bm.faces)
        bm.to_mesh(mesh)
        bm.free()
        return mesh

    def _get_combined_mesh_data(self, head_obj) -> tuple[np.ndarray, np.ndarray]:
        """Get head + teeth mesh data combined for MetaHuman rendering."""
        h_verts, h_faces = self.get_mesh_data(head_obj)
        if self._teeth_obj is None:
            return h_verts, h_faces

        teeth_mesh = self.get_modified_mesh(self._teeth_obj)
        t_verts = np.array([tuple(v.co) for v in teeth_mesh.vertices], dtype=float)
        t_faces = np.array([tuple(p.vertices) for p in teeth_mesh.polygons], dtype=int)
        bpy.data.meshes.remove(teeth_mesh)

        offset = len(h_verts)
        combined_verts = np.concatenate([h_verts, t_verts], axis=0)
        combined_faces = np.concatenate([h_faces, t_faces + offset], axis=0)
        return combined_verts, combined_faces

    @staticmethod
    def get_mesh_data(obj) -> tuple[np.ndarray, np.ndarray]:
        vertices = np.array([tuple(vertex.co) for vertex in obj.data.vertices], dtype=float)
        faces = np.array([tuple(face.vertices) for face in obj.data.polygons], dtype=int)
        return vertices, faces

    def extract_frame(self, blendshapes: np.ndarray | list[float]) -> dict[str, np.ndarray]:
        obj = self.set_blendshapes(blendshapes)

        if self._model == "metahuman":
            vertices, faces = self._get_combined_mesh_data(obj)
        else:
            vertices, faces = self.get_mesh_data(obj)

        landmarks = extract_default_landmarks(vertices)
        return {"vertices": vertices, "faces": faces, **landmarks}

    def render(self, vertices: np.ndarray, faces: np.ndarray) -> None:
        if self.viewer is None:
            raise RuntimeError("Viewer is disabled for this runtime instance")
        self._ensure_vertex_colors(faces, len(vertices))
        self.viewer.update(vertices, faces, vertex_colors=self._vertex_colors)

    def update_visualizer(self, blendshapes: np.ndarray | list[float]) -> dict[str, np.ndarray]:
        frame = self.extract_frame(blendshapes)
        self.render(frame["vertices"], frame["faces"])
        return frame

    set_key_shapes = set_blendshapes
    get_keypoints = get_mesh_data
    get_lip = staticmethod(get_lip_vertices)
    get_tongue = staticmethod(get_tongue_vertices)
    get_cheek = staticmethod(get_cheek_vertices)
    get_key_tongue = staticmethod(get_tongue_tip)
    get_key_cheek = staticmethod(get_cheek_keypoints)
