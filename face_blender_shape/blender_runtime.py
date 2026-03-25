"""在 Blender 中加载 MetaHuman 等资源，将 SRanipal 帧映射为 ARKit 权重并驱动网格 blendshape，可选接入 Open3D 实时预览。"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import bpy
import bmesh  # pyright: ignore[reportMissingModuleSource]
import numpy as np
from mathutils import Vector  # pyright: ignore[reportMissingModuleSource]

from face_blender_shape.blendshape_mapping import (
    ARKIT_SHAPE_NAMES,
    convert_sranipal_to_arkit,
)
from face_blender_shape.constants import (
    BLENDSHAPE_NAMES,
    DEFAULT_OPEN3D_HEIGHT,
    DEFAULT_OPEN3D_WIDTH,
    DEFAULT_OPEN3D_WINDOW_NAME,
    DEFAULT_VIEW_SCALE,
    FRAME_WIDTH,
    METAHUMAN_EYE_LEFT_OBJECT_NAME,
    METAHUMAN_EYE_RIGHT_OBJECT_NAME,
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
from face_blender_shape.paths import METAHUMAN_FBX_PATH, resolve_fbx_path
from face_blender_shape.open3d_viewer import Open3DMeshViewer, SKIN_TONE

# 眼球顶点色相关常量，避免在循环内重复分配小数组
_DEFAULT_PRINCIPLED_RGB = np.array([0.85, 0.85, 0.88], dtype=np.float64)
_PUPIL_COLOR_FLOOR = np.array([0.10, 0.085, 0.09], dtype=np.float64)
_IRIS_RGB_SCALE = np.array([0.94, 1.06, 1.04], dtype=np.float64)
_HAZEL_TINT = np.array([0.33, 0.23, 0.16], dtype=np.float64)
_SCLERA_SOFT = np.array([0.93, 0.94, 0.97], dtype=np.float64)


class FaceBlenderRuntime:
    """由 Blendshape 驱动的面部网格预览；固定使用 MetaHuman 资产，SRanipal CSV 经映射为 ARKit 权重。"""

    def __init__(
        self,
        *,
        enable_viewer: bool = True,
        enable_side_viewer: bool = False,
        window_name: str = DEFAULT_OPEN3D_WINDOW_NAME,
        window_width: int = DEFAULT_OPEN3D_WIDTH,
        window_height: int = DEFAULT_OPEN3D_HEIGHT,
        view_scale: float = DEFAULT_VIEW_SCALE,
        head_object_name: str | None = None,
        extra_mesh_names: Sequence[str] | None = None,
        fbx_path: str | Path | None = None,
    ) -> None:
        """初始化运行时并加载场景资源。

        enable_viewer: 是否创建正面 Open3DMeshViewer。
        enable_side_viewer: 是否额外创建侧视窗口（与主视同步同一网格）。
        window_name: Open3D 主窗口标题。
        window_width: 主窗口宽度（像素）；侧视窗口同宽或略小由实现决定。
        window_height: 主窗口高度（像素）。
        view_scale: 相机/网格显示缩放。
        head_object_name: 头部网格在场景中的对象名；None 时用 MetaHuman 默认头对象名。
        extra_mesh_names: 除面部外还要合并进视图的网格对象名；None 时默认牙齿与双眼。
        fbx_path: 自定义 FBX 路径；None 时使用项目默认 MetaHuman 路径解析结果。
        """
        self._extra_mesh_names = (
            tuple(n.strip() for n in extra_mesh_names if n.strip())
            if extra_mesh_names is not None
            else None
        )

        self._arkit_names = np.array(ARKIT_SHAPE_NAMES)
        self._convert_frame = convert_sranipal_to_arkit
        head_object_name = head_object_name or METAHUMAN_HEAD_OBJECT_NAME

        self.blendshape_names = np.array(BLENDSHAPE_NAMES)
        self._fbx_path = resolve_fbx_path(fbx_path)
        self.load_fbx()
        self.set_active_object(head_object_name)

        if self._extra_mesh_names is not None:
            parts: list[bpy.types.Object] = []
            for n in self._extra_mesh_names:
                ob = bpy.data.objects.get(n)
                if ob is None:
                    print(f"[face_blender_shape] warning: extra mesh {n!r} not found in FBX")
                elif ob.type != "MESH":
                    print(f"[face_blender_shape] warning: {n!r} is not a mesh, skipped")
                else:
                    parts.append(ob)
            self._append_parts = tuple(parts)
        else:
            self._append_parts = (
                bpy.data.objects.get(METAHUMAN_TEETH_OBJECT_NAME),
                bpy.data.objects.get(METAHUMAN_EYE_LEFT_OBJECT_NAME),
                bpy.data.objects.get(METAHUMAN_EYE_RIGHT_OBJECT_NAME),
            )

        self.viewer = (
            Open3DMeshViewer(
                window_name=window_name,
                view_scale=view_scale,
                window_width=window_width,
                window_height=window_height,
                camera_profile="front",
            )
            if enable_viewer
            else None
        )
        self.side_viewer = (
            Open3DMeshViewer(
                window_name=f"{window_name} — 侧视",
                view_scale=view_scale * 1.05,
                window_width=max(window_width * 4 // 5, 480),
                window_height=max(window_height * 4 // 5, 400),
                camera_profile="side",
            )
            if enable_viewer and enable_side_viewer
            else None
        )

    def load_fbx(self) -> None:
        """解析并导入 FBX（默认或构造时传入的路径）到当前 Blender 数据。"""
        self.fbx_path = self._fbx_path
        bpy.ops.object.select_all(action="DESELECT")
        bpy.ops.import_scene.fbx(filepath=str(self.fbx_path))

    def set_active_object(self, object_name: str = METAHUMAN_HEAD_OBJECT_NAME) -> None:
        """将指定名称对象设为活动对象并绑定到 self.active_obj。

        object_name: bpy.data.objects 中的对象名。
        """
        self.active_obj = bpy.data.objects[object_name]
        bpy.context.view_layer.objects.active = self.active_obj

    # ------------------------------------------------------------------
    # Core blendshape / mesh pipeline
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_frame(blendshapes: np.ndarray | list[float]) -> np.ndarray:
        """校验一帧 Blendshape 向量长度等于 FRAME_WIDTH 并返回 float 一维数组。

        blendshapes: 任意可转为数组的帧数据。
        """
        frame = np.asarray(blendshapes, dtype=float).reshape(-1)
        if frame.size != FRAME_WIDTH:
            raise ValueError(f"Expected {FRAME_WIDTH} blendshape values, got {frame.size}")
        return frame

    @staticmethod
    def _write_shape_key_values(key_blocks, names: np.ndarray, values: np.ndarray) -> None:
        """仅更新已存在的形态键权重。

        key_blocks: shape_keys.key_blocks 映射。
        names: 形态键名称一维数组。
        values: 与 names 等长的权重数组。
        """
        for name, value in zip(names, values, strict=True):
            if name in key_blocks:
                key_blocks[name].value = float(value)

    def _apply_arkit_shapes(self, arkit_values: np.ndarray) -> None:
        """将 ARKit 权重写入活动头部对象的形态键。"""
        sk = self.active_obj.data.shape_keys
        if sk is None:
            return
        assert self._arkit_names is not None
        self._write_shape_key_values(sk.key_blocks, self._arkit_names, arkit_values)

    def _apply_arkit_to_secondary_meshes(self, arkit_values: np.ndarray) -> None:
        """将同一套 ARKit 权重同步到牙齿、眼球等附加网格上存在的形态键。"""
        for obj in self._append_parts:
            if obj is None:
                continue
            sk = obj.data.shape_keys
            if sk is None:
                continue
            assert self._arkit_names is not None
            self._write_shape_key_values(sk.key_blocks, self._arkit_names, arkit_values)

    @staticmethod
    def _principled_base_color_rgb(mat) -> np.ndarray:
        """读取 Principled BSDF 的默认 Base Color（不采样贴图）。

        mat: bpy.types.Material 或 None。
        """
        if mat is None or not getattr(mat, "use_nodes", False):
            return _DEFAULT_PRINCIPLED_RGB.copy()
        try:
            for node in mat.node_tree.nodes:
                if node.type == "BSDF_PRINCIPLED":
                    bc = node.inputs["Base Color"].default_value
                    return np.clip(np.array(bc[:3], dtype=np.float64), 0.0, 1.0)
        except (AttributeError, KeyError, TypeError):
            pass
        return _DEFAULT_PRINCIPLED_RGB.copy()

    @classmethod
    def _mesh_vertex_colors_from_materials(cls, mesh, materials) -> np.ndarray:
        """按多边形材质槽将 Base Color 平均到各顶点（用于非眼球部件）。

        mesh: 已三角化的 bpy.types.Mesh。
        materials: 源对象的 data.materials，与 polygon.material_index 对应。
        """
        n = len(mesh.vertices)
        acc = np.zeros((n, 3), dtype=np.float64)
        counts = np.zeros(n, dtype=np.float64)
        mats = materials
        for poly in mesh.polygons:
            if mats and len(mats) > 0:
                mid = min(int(poly.material_index), len(mats) - 1)
                mat = mats[mid]
            else:
                mat = None
            rgb = cls._principled_base_color_rgb(mat)
            for vi in poly.vertices:
                acc[vi] += rgb
                counts[vi] += 1.0
        counts = np.maximum(counts, 1.0)
        return acc / counts[:, None]

    @classmethod
    def _material_rgb_for_eye_viewport(cls, mat) -> np.ndarray:
        """针对 Open3D 简单光照调整眼球材质 RGB，减轻“死黑瞳孔”等观感。

        mat: 材质或 None。
        """
        rgb = cls._principled_base_color_rgb(mat)
        if mat is None:
            return rgb
        name = (mat.name or "").lower()
        if "pupil" in name:
            return np.clip(np.maximum(rgb, _PUPIL_COLOR_FLOOR), 0.0, 1.0)
        if "iris" in name:
            rich = np.clip(rgb * _IRIS_RGB_SCALE, 0.0, 1.0)
            return np.clip(0.70 * rich + 0.30 * _HAZEL_TINT, 0.0, 1.0)
        if float(rgb.mean()) >= 0.45:
            return _SCLERA_SOFT.copy()
        return rgb

    @classmethod
    def _mesh_vertex_colors_dominant_material(cls, mesh, materials) -> np.ndarray:
        """按邻接三角面积最大的材质为顶点赋色（解决眼球共享顶点被平均的问题）。

        mesh: 三角化网格。
        materials: 源对象材质列表。
        """
        from collections import defaultdict

        verts = mesh.vertices
        n = len(verts)
        mats = materials
        wby = [defaultdict(float) for _ in range(n)]

        for poly in mesh.polygons:
            vids = list(poly.vertices)
            if len(vids) < 3:
                continue
            if mats and len(mats) > 0:
                mid = min(int(poly.material_index), len(mats) - 1)
            else:
                mid = -1
            c0 = Vector(verts[vids[0]].co)
            c1 = Vector(verts[vids[1]].co)
            c2 = Vector(verts[vids[2]].co)
            area = (c1 - c0).cross(c2 - c0).length / 2.0
            for vi in poly.vertices:
                wby[vi][mid] += float(area)

        out = np.zeros((n, 3), dtype=np.float64)
        for i in range(n):
            weights = wby[i]
            if not weights:
                out[i] = cls._material_rgb_for_eye_viewport(None)
                continue
            best_mid = max(weights.items(), key=lambda kv: kv[1])[0]
            if best_mid < 0 or best_mid >= len(mats):
                mat = None
            else:
                mat = mats[best_mid]
            out[i] = cls._material_rgb_for_eye_viewport(mat)
        return out

    @staticmethod
    def _is_eye_mesh_object(obj) -> bool:
        """判断对象名是否像眼球网格（名称中包含 eye，不区分大小写）。

        obj: bpy.types.Object 或 None。
        """
        return bool(obj and obj.name and "eye" in obj.name.lower())

    def set_blendshapes(self, blendshapes: np.ndarray | list[float]) -> bpy.types.Object:
        """应用一帧 Blendshape，返回带变形网格的临时对象副本（供采样顶点/面）。

        blendshapes: 一帧权重，长度须为 FRAME_WIDTH。
        """
        frame = self._validate_frame(blendshapes)
        bpy.context.view_layer.objects.active = self.active_obj
        self.active_obj.update_from_editmode()

        arkit_values = self._convert_frame(frame)
        self._apply_arkit_shapes(arkit_values)
        self._apply_arkit_to_secondary_meshes(arkit_values)

        obj = self.active_obj.copy()
        mesh = self.get_modified_mesh(self.active_obj)

        obj.modifiers.clear()
        obj.data = mesh
        return obj

    def get_modified_mesh(self, obj, cage: bool = False) -> bpy.types.Mesh:
        """在依赖图上求值对象几何并三角化，返回新建的 Mesh 数据块。

        obj: 场景中的网格对象。
        cage: 是否使用 cage 模式传入 bmesh.from_object。
        """
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

    def _has_extra_meshes_for_view(self) -> bool:
        """当前是否在视图中合并了至少一个附加网格。"""
        return any(p is not None for p in self._append_parts)

    def _get_combined_mesh_data(self, head_obj) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """合并头部与附加部件的顶点、三角面与顶点色（RGB 位于 [0,1]）。

        head_obj: 已变形的头部对象副本。
        """
        h_verts, h_faces = self.get_mesh_data(head_obj)
        h_n = len(h_verts)
        head_cols = np.tile(np.asarray(SKIN_TONE, dtype=np.float64), (h_n, 1))

        verts_list = [h_verts]
        faces_list = [h_faces]
        colors_list = [head_cols]
        offset = h_n

        for part_obj in self._append_parts:
            if part_obj is None:
                continue
            part_mesh = self.get_modified_mesh(part_obj)
            try:
                slot_mats = part_obj.data.materials
                if self._is_eye_mesh_object(part_obj):
                    vcols = self._mesh_vertex_colors_dominant_material(part_mesh, slot_mats)
                else:
                    vcols = self._mesh_vertex_colors_from_materials(part_mesh, slot_mats)
                p_verts = self._vertices_coords_numpy(part_mesh.vertices)
                p_faces = np.array([tuple(p.vertices) for p in part_mesh.polygons], dtype=int)
            finally:
                bpy.data.meshes.remove(part_mesh)

            if len(vcols) != len(p_verts):
                vcols = np.tile(np.asarray(SKIN_TONE, dtype=np.float64), (len(p_verts), 1))

            verts_list.append(p_verts)
            faces_list.append(p_faces + offset)
            colors_list.append(vcols)
            offset += len(p_verts)

        return (
            np.concatenate(verts_list, axis=0),
            np.concatenate(faces_list, axis=0),
            np.concatenate(colors_list, axis=0),
        )

    @staticmethod
    def _vertices_coords_numpy(vertices) -> np.ndarray:
        """将 Blender MeshVertices 转为形状 (N, 3) 的 float 坐标数组。

        vertices: bpy.types.Mesh.vertices 顶点集合。
        """
        return np.array([tuple(v.co) for v in vertices], dtype=float)

    @staticmethod
    def get_mesh_data(obj) -> tuple[np.ndarray, np.ndarray]:
        """读取网格数据块的顶点与多边形顶点索引（未另行依赖图求值）。

        obj: type 为 MESH 的 bpy 对象。
        """
        vertices = FaceBlenderRuntime._vertices_coords_numpy(obj.data.vertices)
        faces = np.array([tuple(face.vertices) for face in obj.data.polygons], dtype=int)
        return vertices, faces

    def extract_frame(self, blendshapes: np.ndarray | list[float]) -> dict[str, np.ndarray]:
        """应用一帧形态键并返回顶点、面、可选顶点色及默认 landmarks；结束后销毁临时对象。

        blendshapes: 一帧 Blendshape 权重。
        """
        obj = self.set_blendshapes(blendshapes)
        mesh = obj.data
        try:
            if self._has_extra_meshes_for_view():
                vertices, faces, vcols = self._get_combined_mesh_data(obj)
                landmarks = extract_default_landmarks(vertices)
                return {
                    "vertices": vertices,
                    "faces": faces,
                    "vertex_colors": vcols,
                    **landmarks,
                }

            vertices, faces = self.get_mesh_data(obj)
            landmarks = extract_default_landmarks(vertices)
            return {"vertices": vertices, "faces": faces, **landmarks}
        finally:
            bpy.data.objects.remove(obj)
            if mesh.users == 0:
                bpy.data.meshes.remove(mesh)

    def render(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        *,
        vertex_colors: np.ndarray | None = None,
    ) -> None:
        """将几何提交给 Open3D 视窗。

        vertices: 顶点坐标数组。
        faces: 三角面索引。
        vertex_colors: 可选 RGB 顶点色；None 时由查看器使用默认肤色。
        """
        if self.viewer is None and self.side_viewer is None:
            return
        if self.viewer is not None:
            self.viewer.update(vertices, faces, vertex_colors=vertex_colors)
        if self.side_viewer is not None:
            self.side_viewer.update(vertices, faces, vertex_colors=vertex_colors)

    def update_visualizer(self, blendshapes: np.ndarray | list[float]) -> dict[str, np.ndarray]:
        """提取一帧并刷新视窗，返回与 extract_frame 相同的字典。

        blendshapes: 一帧 Blendshape 权重。
        """
        frame = self.extract_frame(blendshapes)
        self.render(
            frame["vertices"],
            frame["faces"],
            vertex_colors=frame.get("vertex_colors"),
        )
        return frame

    set_key_shapes = set_blendshapes
    get_keypoints = get_mesh_data
    get_lip = staticmethod(get_lip_vertices)
    get_tongue = staticmethod(get_tongue_vertices)
    get_cheek = staticmethod(get_cheek_vertices)
    get_key_tongue = staticmethod(get_tongue_tip)
    get_key_cheek = staticmethod(get_cheek_keypoints)
