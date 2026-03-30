from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict, cast

import bpy
import numpy as np
from numpy.typing import NDArray

# 完整 Blender 内置 bmesh；PyPI 的 bpy wheel 通常不带该模块，此时走 to_mesh + 扇形三角化回退路径。
try:
    import bmesh as _bmesh  # pyright: ignore[reportMissingModuleSource]
except ModuleNotFoundError:
    _bmesh = None

from face_blender_shape.constants import (
    BLENDSHAPE_NAMES,
    DEFAULT_HEAD_OBJECT_NAME,
    DEFAULT_OPEN3D_WINDOW_NAME,
    FRAME_WIDTH,
    METAHUMAN_HEAD_OBJECT_NAME,
    METAHUMAN_TEETH_OBJECT_NAME,
)
from face_blender_shape.landmarks import (
    build_mouth_removal_mask,
    extract_default_landmarks,
    get_cheek_keypoints,
    get_cheek_vertices,
    get_lip_vertices,
    get_tongue_tip,
    get_tongue_vertices,
)
from face_blender_shape.viewers.open3d_viewer import Open3DMeshViewer

# 资源根目录与默认 FBX 路径（相对本包上级目录的 assets）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MODELS_DIR = _PROJECT_ROOT / "assets" / "models"
_DEFAULT_FBX_PATH = _MODELS_DIR / "sranipal_head.fbx"
_METAHUMAN_FBX_PATH = _MODELS_DIR / "Metahuman_Head.fbx"


def _fan_triangulate_to_new_mesh(mesh_eval: Any) -> Any:
    """
    将已求值网格扇形三角化并写入新的 Mesh 数据块。

    参数:
        mesh_eval: ``to_mesh`` 得到的临时 ``Mesh``；本函数只读其顶点与多边形。

    返回:
        新的三角网格 ``Mesh``。不复制 UV 层（无 bmesh 时贴图烘焙可能缺少 UV）。
    """
    verts = [tuple(v.co) for v in mesh_eval.vertices]
    tris: list[tuple[int, int, int]] = []
    for poly in mesh_eval.polygons:
        vids = list(poly.vertices)
        n = len(vids)
        if n < 3:
            continue
        if n == 3:
            tris.append((vids[0], vids[1], vids[2]))
        else:
            for i in range(1, n - 1):
                tris.append((vids[0], vids[i], vids[i + 1]))
    mesh_out = bpy.data.meshes.new("Deformed")
    mesh_out.from_pydata(verts, [], tris)
    mesh_out.update()
    return mesh_out


def _modified_mesh_without_bmesh(obj: Any, cage: bool) -> Any:
    """
    用依赖图求值 + ``Object.to_mesh`` 取变形后几何，再扇形三角化（不依赖 bmesh）。

    参数:
        obj: 场景中的网格对象。
        cage: 与 bmesh 路径对齐保留参数；当前回退实现未使用（等价于未启用 cage）。

    返回:
        新的三角化 ``Mesh``。
    """
    _ = cage
    depsgraph = bpy.context.evaluated_depsgraph_get()
    obj_eval = obj.evaluated_get(depsgraph)
    try:
        mesh_eval = obj_eval.to_mesh(depsgraph=depsgraph)
    except TypeError:
        mesh_eval = obj_eval.to_mesh()
    if mesh_eval is None:
        raise RuntimeError(
            "无法将对象转为网格（to_mesh 返回 None），请确认对象为可求值的网格。"
        )
    try:
        return _fan_triangulate_to_new_mesh(mesh_eval)
    finally:
        obj_eval.to_mesh_clear()


class FrameData(TypedDict):
    """单帧变形网格与关键点，供可视化与导出。"""

    vertices: NDArray[np.float64]
    faces: NDArray[np.int64]
    lip: NDArray[np.float64]
    tongue: NDArray[np.float64]
    cheek: NDArray[np.float64]
    tongue_tip: NDArray[np.float64]
    cheek_keypoints: NDArray[np.float64]
    keypoints: NDArray[np.float64]


BlendshapeInput = NDArray[np.float64] | list[float]


class FaceBlenderRuntime:
    """由 Blendshape 驱动的面部网格运行时。

    支持两种模型后端：
    - ``sranipal``（默认）：含 37 个 blendshape 的 SRanipal 头模。
    - ``metahuman``：含 52 个 ARKit blendshape 的 MetaHuman 头模；
      SRanipal 格式的 CSV 会通过映射层自动转换。
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
        cutaway: bool = False,
    ) -> None:
        """
        初始化运行时：加载 FBX、绑定活动头对象、可选 Open3D 窗口与贴图。

        参数:
            path: FBX 文件路径；为 None 时按 model 使用默认 SRanipal 或 MetaHuman FBX。
            enable_viewer: 是否创建 Open3D 网格查看器。
            window_name: Open3D 窗口标题。
            head_object_name: 场景中头网格对象名；为 None 时使用各 model 的默认名称。
            texture_path: 外置 albedo 贴图路径；MetaHuman 且为 None 时不强制加载磁盘贴图。
            model: ``"sranipal"`` 或 ``"metahuman"``。
            cutaway: 是否在渲染时裁掉口腔区域（需配合 landmarks 掩码）。
        """
        self._model = model
        self._cutaway = cutaway
        self._cutaway_mask = None

        if model == "metahuman":
            from face_blender_shape.blendshape_mapping import (
                ARKIT_SHAPE_NAMES,
                convert_sranipal_to_arkit,
            )

            self._arkit_names = np.array(ARKIT_SHAPE_NAMES)
            self._convert_frame = convert_sranipal_to_arkit
            path = path or str(_METAHUMAN_FBX_PATH)
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
        self._triangle_uvs: NDArray[np.float64] | None = None
        self._vertex_colors: NDArray[np.float64] | None = None

        self.viewer = (
            Open3DMeshViewer(window_name=window_name) if enable_viewer else None
        )

    def load_fbx(self, path: str | None) -> None:
        """
        将 FBX 导入当前 Blender 场景。

        参数:
            path: FBX 路径；为 None 时使用包内默认 SRanipal 头模；支持 ``~`` 展开。
        """
        if path is None:
            self.fbx_path = _DEFAULT_FBX_PATH.resolve()
        else:
            self.fbx_path = Path(path).expanduser()
        bpy.ops.object.select_all(action="DESELECT")
        bpy.ops.import_scene.fbx(filepath=str(self.fbx_path))

    def set_active_object(self, object_name: str = DEFAULT_HEAD_OBJECT_NAME) -> None:
        """
        将指定名称的对象设为活动对象并绑定到当前视图层。

        参数:
            object_name: 场景中要驱动 blendshape 的网格对象名。
        """
        self.active_obj = bpy.data.objects[object_name]
        bpy.context.view_layer.objects.active = self.active_obj

    # ---------- 贴图加载 ----------

    def _load_texture(self, texture_path: str | None) -> NDArray[np.uint8] | None:
        """
        优先从活动对象材质节点取贴图，否则从磁盘路径加载 RGB 数组。

        参数:
            texture_path: 外置贴图文件；为 None 且材质无有效图像时返回 None。
        """
        img_array = self._extract_bpy_texture()
        if img_array is not None:
            return img_array
        if texture_path is None:
            return None
        resolved = Path(texture_path).expanduser()
        return self._load_external_texture(resolved)

    def _extract_bpy_texture(self) -> NDArray[np.uint8] | None:
        """
        遍历活动对象材质中的 TEX_IMAGE 节点，将首张有效图像转为 numpy RGB。

        参数: 无（使用 self.active_obj）。
        """
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

    def _load_external_texture(self, path: Path) -> NDArray[np.uint8] | None:
        """
        通过复用或加载 bpy.data.images 读取磁盘贴图并转为 RGB 数组。

        参数:
            path: 已 expanduser 的贴图文件路径。
        """
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
    def _bpy_image_to_numpy(img: Any, max_size: int = 2048) -> NDArray[np.uint8]:
        """
        将 Blender 图像像素转为 uint8 RGB；过大时按比例缩小以节省内存。

        参数:
            img: bpy 图像数据块。
            max_size: 长边超过此值时用 PIL 缩略图（LANCZOS）。
        """
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

    # ---------- 贴图烘焙到顶点色 ----------

    @staticmethod
    def _bake_vertex_colors(
        texture: NDArray[np.uint8],
        triangle_uvs: NDArray[np.float64],
        faces: NDArray[np.int64],
        n_vertices: int,
    ) -> NDArray[np.float64]:
        """
        按三角形展开 UV 从贴图采样，对每个顶点做邻接面颜色平均。

        参数:
            texture: H×W×3 的 uint8 或 float 贴图（内部按 /255 归一化）。
            triangle_uvs: 每条三角边对应一个 UV，形状 (3*F, 2)，与展开后的面索引一致。
            faces: 三角面顶点索引，形状 (F, 3)。
            n_vertices: 网格顶点数，用于分配输出 (N, 3) 顶点色。
        """
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

    def _ensure_vertex_colors(
        self, faces: NDArray[np.int64], n_vertices: int
    ) -> None:
        """
        若尚未烘焙顶点色且有贴图与三角 UV，则计算并缓存 self._vertex_colors。

        参数:
            faces: 当前帧三角面索引。
            n_vertices: 顶点数量。
        """
        if self._vertex_colors is not None:
            return
        if self._texture_image is not None and self._triangle_uvs is not None:
            self._vertex_colors = self._bake_vertex_colors(
                self._texture_image,
                self._triangle_uvs,
                faces,
                n_vertices,
            )

    # ---------- UV 提取 ----------

    @staticmethod
    def _extract_triangle_uvs(mesh: Any) -> NDArray[np.float64] | None:
        """
        读取网格活动 UV 层，按 loop 顺序展开为每行 (u, v) 的数组。

        参数:
            mesh: 已应用修改器后的 bpy 网格数据。
        """
        if not mesh.uv_layers:
            return None
        uv_layer = mesh.uv_layers.active or mesh.uv_layers[0]
        return np.array([tuple(d.uv) for d in uv_layer.data], dtype=float)

    # ---------- Blendshape 与网格管线 ----------

    @staticmethod
    def _validate_frame(blendshapes: BlendshapeInput) -> NDArray[np.float64]:
        """
        将输入展平为一维向量并校验长度等于 FRAME_WIDTH。

        参数:
            blendshapes: 一帧 SRanipal 维度的 blendshape 权重（数组或列表）。
        """
        frame = np.asarray(blendshapes, dtype=float).reshape(-1)
        if frame.size != FRAME_WIDTH:
            raise ValueError(
                f"Expected {FRAME_WIDTH} blendshape values, got {frame.size}"
            )
        return frame

    def _apply_arkit_shapes(self, arkit_values: NDArray[np.float64]) -> None:
        """
        在活动头对象上按名称写入 ARKit blendshape 权重。

        参数:
            arkit_values: 与 self._arkit_names 等长的权重向量。
        """
        names = self._arkit_names
        assert names is not None
        key_blocks = self.active_obj.data.shape_keys.key_blocks
        for name, value in zip(names, arkit_values):
            if name in key_blocks:
                key_blocks[name].value = float(value)

    def _apply_teeth_shapes(self, arkit_values: NDArray[np.float64]) -> None:
        """
        将同名 jaw/口型相关 shape 同步到牙齿对象（MetaHuman）。

        参数:
            arkit_values: 与头对象一致的 ARKit 权重向量。
        """
        if self._teeth_obj is None:
            return
        teeth_keys = self._teeth_obj.data.shape_keys
        if teeth_keys is None:
            return
        names = self._arkit_names
        assert names is not None
        key_blocks = teeth_keys.key_blocks
        for name, value in zip(names, arkit_values):
            if name in key_blocks:
                key_blocks[name].value = float(value)

    def set_blendshapes(self, blendshapes: BlendshapeInput) -> Any:
        """
        应用一帧 blendshape，从 depsgraph 取出变形网格并返回临时对象副本。

        参数:
            blendshapes: 长度为 FRAME_WIDTH 的 SRanipal 权重；metahuman 模式下会先映射到 ARKit。

        返回:
            带有当前变形 mesh 数据的对象副本（modifiers 已清空），供后续读顶点/面。
        """
        frame = self._validate_frame(blendshapes)
        bpy.context.view_layer.objects.active = self.active_obj
        bpy.context.object.update_from_editmode()

        if self._model == "metahuman":
            cf = self._convert_frame
            if cf is None:
                raise RuntimeError("metahuman 模式未初始化 ARKit 映射函数")
            arkit_values = cf(frame)
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

    def get_modified_mesh(self, obj: Any, cage: bool = False) -> Any:
        """
        从依赖图求值对象，三角化后得到新的 Mesh 数据块。

        若存在 ``bmesh``（完整 Blender），用 bmesh 求值并三角化（保留 UV 等层）。
        否则使用 ``to_mesh`` + 扇形三角化，适用于 PyPI ``bpy`` wheel（无 bmesh 模块）。

        参数:
            obj: 要带修改器栈求值的网格对象。
            cage: 是否使用 cage 模式参与求值；仅 bmesh 分支生效。
        """
        if _bmesh is not None:
            bm = _bmesh.new()
            bm.from_object(
                obj,
                bpy.context.evaluated_depsgraph_get(),
                cage=cage,
            )
            mesh = bpy.data.meshes.new("Deformed")
            _bmesh.ops.triangulate(bm, faces=bm.faces)
            bm.to_mesh(mesh)
            bm.free()
            return mesh
        return _modified_mesh_without_bmesh(obj, cage)

    def _get_combined_mesh_data(
        self, head_obj: Any
    ) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        """
        合并头部与牙齿的顶点与三角面索引（牙齿顶点索引整体偏移）。

        参数:
            head_obj: 已应用 blendshape 后的头对象副本（与 extract_frame 流程一致）。

        返回:
            (vertices, faces) 的 numpy 数组，用于 MetaHuman 一体可视化。
        """
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
    def get_mesh_data(obj: Any) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        """
        从对象当前 mesh 数据读取顶点坐标与多边形顶点索引（未三角化时按多边形边数）。

        参数:
            obj: 其 data 为 Mesh 的场景对象。
        """
        mesh = obj.data
        vertices = np.array([tuple(v.co) for v in mesh.vertices], dtype=float)
        faces = np.array([tuple(p.vertices) for p in mesh.polygons], dtype=int)
        return vertices, faces

    def extract_frame(self, blendshapes: BlendshapeInput) -> FrameData:
        """
        应用 blendshape 并输出顶点、面及默认 landmarks 字典。

        参数:
            blendshapes: 一帧 SRanipal 维度权重。
        """
        obj = self.set_blendshapes(blendshapes)

        if self._model == "metahuman":
            vertices, faces = self._get_combined_mesh_data(obj)
        else:
            vertices, faces = self.get_mesh_data(obj)

        landmarks = extract_default_landmarks(vertices)
        return cast(FrameData, {"vertices": vertices, "faces": faces, **landmarks})

    def render(self, vertices: NDArray[np.float64], faces: NDArray[np.int64]) -> None:
        """
        将网格推送到 Open3D 查看器；可选口腔裁切与顶点色。

        参数:
            vertices: 世界空间或模型空间顶点 N×3。
            faces: 三角面索引 F×3。
        """
        if self.viewer is None:
            raise RuntimeError("Viewer is disabled for this runtime instance")
        self._ensure_vertex_colors(faces, len(vertices))

        if self._cutaway:
            if self._cutaway_mask is None:
                self._cutaway_mask = build_mouth_removal_mask(faces, vertices)
            faces = faces[self._cutaway_mask]

        self.viewer.update(vertices, faces, vertex_colors=self._vertex_colors)

    def update_visualizer(self, blendshapes: BlendshapeInput) -> FrameData:
        """
        提取一帧数据并刷新 Open3D，返回与 extract_frame 相同的结构。

        参数:
            blendshapes: 一帧 blendshape 权重。
        """
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
