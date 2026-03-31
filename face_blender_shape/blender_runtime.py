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

# 与 SRanipal CSV 列顺序一致的通道名、默认头对象名、Open3D 窗口标题、每帧权重维度。
from face_blender_shape.constants import (
    BLENDSHAPE_NAMES,
    DEFAULT_HEAD_OBJECT_NAME,
    DEFAULT_OPEN3D_WINDOW_NAME,
    FRAME_WIDTH,
)

# 默认头模顶点切片：唇/舌/颊关键点、整帧 landmark 打包、口腔裁切面布尔掩码。
from face_blender_shape.landmarks import (
    build_mouth_removal_mask,
    extract_default_landmarks,
    get_cheek_keypoints,
    get_cheek_vertices,
    get_lip_vertices,
    get_tongue_tip,
    get_tongue_vertices,
)

# 变形网格的实时三角网格窗口（与 Blender 求值结果对接）。
from face_blender_shape.viewers.open3d_viewer import Open3DMeshViewer

# 资源根目录与默认 FBX 路径（相对本包上级目录的 assets）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MODELS_DIR = _PROJECT_ROOT / "assets" / "models"
_DEFAULT_FBX_PATH = _MODELS_DIR / "sranipal_head.fbx"


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
    """由 SRanipal 风格 37 维 BlendShape 驱动的面部网格运行时（默认 ``sranipal_head.fbx``）。"""

    def __init__(
        self,
        path: str | None = None,
        *,
        enable_viewer: bool = True,
        window_name: str = DEFAULT_OPEN3D_WINDOW_NAME,
        head_object_name: str | None = None,
        texture_path: str | None = None,
        cutaway: bool = False,
    ) -> None:
        """
        初始化运行时：加载 FBX、绑定活动头对象、可选 Open3D 窗口与贴图。

        参数:
            path: FBX 文件路径；为 None 时使用包内默认 SRanipal 头模。
            enable_viewer: 是否创建 Open3D 网格查看器。
            window_name: Open3D 窗口标题。
            head_object_name: 场景中头网格对象名；为 None 时使用 ``DEFAULT_HEAD_OBJECT_NAME``。
            texture_path: 外置 albedo 贴图路径；为 None 时尽量从材质读取。
            cutaway: 是否在渲染时裁掉口腔区域（需配合 landmarks 掩码）。
        """
        self._cutaway = cutaway  # True 时 render 会裁掉口腔附近面片
        self._cutaway_mask = None  # 懒计算：首帧根据舌包围盒生成 (F,) 布尔掩码

        head_object_name = head_object_name or DEFAULT_HEAD_OBJECT_NAME

        self.blendshape_names = np.array(BLENDSHAPE_NAMES)  # 与 CSV 列顺序一致，供 set_blendshapes 按名写权重
        self.load_fbx(path)  # 将 FBX 导入当前 bpy 场景
        self.set_active_object(object_name=head_object_name)  # 指定要驱动 shape key 的头网格

        self._texture_image = self._load_texture(texture_path)  # RGB uint8；无则 None
        self._triangle_uvs: NDArray[np.float64] | None = None  # 首帧从求值 mesh 抽 UV，供烘焙顶点色
        self._vertex_colors: NDArray[np.float64] | None = None  # 由贴图+UV 烘焙，懒算

        self.viewer = (
            Open3DMeshViewer(window_name=window_name) if enable_viewer else None
        )  # 关闭时 extract_frame 仍可用，render 会报错

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
        # 取消全部选中，避免残留选中影响导入后对象的 active 与视图层状态。
        bpy.ops.object.select_all(action="DESELECT")
        # 将指定路径的 FBX 并入当前场景（网格、形态键等）；运算符要求 filepath 为 str。
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
        加载头模贴图为 RGB 数组：先扫活动对象材质中首张有效 TEX_IMAGE；若无则尝试
        复用 bpy.data.images 指向该路径并 reload，否则 load 新图像。

        参数:
            texture_path: 外置贴图文件；材质中无有效图像且为 None 时返回 None。
        """
        obj = self.active_obj
        if obj.data.materials:
            for mat in obj.data.materials:
                if mat is None or not mat.use_nodes:
                    continue
                for node in mat.node_tree.nodes:
                    if node.type == "TEX_IMAGE" and node.image is not None:
                        img = node.image
                        if img.size[0] == 0 or img.size[1] == 0:
                            continue
                        return self._bpy_image_to_numpy(img)

        if texture_path is None:
            return None
        path_str = str(Path(texture_path).expanduser())
        for img in bpy.data.images:
            if img.name in ("Render Result", "Viewer Node"):
                continue
            img.filepath = path_str
            img.reload()
            if img.size[0] > 0 and img.size[1] > 0:
                return self._bpy_image_to_numpy(img)

        disk_img = bpy.data.images.load(path_str)
        if disk_img.size[0] > 0 and disk_img.size[1] > 0:
            return self._bpy_image_to_numpy(disk_img)
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

    def _ensure_vertex_colors(self, faces: NDArray[np.int64], n_vertices: int) -> None:
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

    def set_blendshapes(self, blendshapes: BlendshapeInput) -> Any:
        """
        应用一帧 blendshape，从 depsgraph 取出变形网格并返回临时对象副本。

        参数:
            blendshapes: 长度为 FRAME_WIDTH 的 SRanipal 权重向量。

        返回:
            带有当前变形 mesh 数据的对象副本（modifiers 已清空），供后续读顶点/面。
        """
        frame = self._validate_frame(blendshapes)
        bpy.context.view_layer.objects.active = self.active_obj
        bpy.context.object.update_from_editmode()

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
        self.render(vertices=frame["vertices"], faces=frame["faces"])
        return frame

    set_key_shapes = set_blendshapes
    get_keypoints = get_mesh_data
    get_lip = staticmethod(get_lip_vertices)
    get_tongue = staticmethod(get_tongue_vertices)
    get_cheek = staticmethod(get_cheek_vertices)
    get_key_tongue = staticmethod(get_tongue_tip)
    get_key_cheek = staticmethod(get_cheek_keypoints)
