from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

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

# 头舌网格拆分与线框边（与默认 FBX 舌顶点下标一致）。
from face_blender_shape.landmarks import (
    split_head_tongue_meshes,
    unique_edges_from_faces,
)

# 变形网格的实时三角网格窗口（与 Blender 求值结果对接）。
from face_blender_shape.open3d_viewer import Open3DMeshViewer

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
        新的三角网格 ``Mesh``。不复制 UV 层（无 bmesh 时扇形三角化会丢 UV）。
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
    """单帧变形网格（顶点 + 三角面）。"""

    vertices: NDArray[np.float64]
    faces: NDArray[np.int64]


class FaceBlenderRuntime:
    """由 SRanipal 风格 37 维 BlendShape 驱动的面部网格运行时（默认 ``sranipal_head.fbx``）。"""

    def __init__(
        self,
        path: str | None = None,
        *,
        enable_viewer: bool = True,
        window_name: str = DEFAULT_OPEN3D_WINDOW_NAME,
        head_object_name: str | None = None,
        wireframe_head: bool = False,
    ) -> None:
        """
        初始化运行时：加载 FBX、绑定活动头对象、可选 Open3D 窗口。

        参数:
            path: FBX 文件路径；为 None 时使用包内默认 SRanipal 头模。
            enable_viewer: 是否创建 Open3D 网格查看器。
            window_name: Open3D 窗口标题。
            head_object_name: 场景中头网格对象名；为 None 时使用 ``DEFAULT_HEAD_OBJECT_NAME``。
            wireframe_head: 是否仅将头壳画为线框、舌区域保持实体（Open3D 默认肤色）。
        """
        self._wireframe_head = (
            wireframe_head  # True 时 Open3D 用紧凑外壳 LineSet + 紧凑舌网格
        )
        self._wf_shell_edges: NDArray[np.int64] | None = (
            None  # 外壳紧凑网格上的无向边 (E,2)，局部下标
        )
        self._wf_tongue_faces: NDArray[np.int64] | None = (
            None  # 舌紧凑三角面 (T,3)，局部下标
        )
        self._wf_shell_global_idx: NDArray[np.int64] | None = (
            None  # 外壳紧凑顶点对应的全局顶点下标
        )
        self._wf_tongue_global_idx: NDArray[np.int64] | None = (
            None  # 舌紧凑顶点对应的全局顶点下标
        )

        head_object_name = head_object_name or DEFAULT_HEAD_OBJECT_NAME

        self.blendshape_names = np.array(
            BLENDSHAPE_NAMES
        )  # 与 CSV 列顺序一致，与 _key_blocks 一一对应
        self.load_fbx(path)  # 将 FBX 导入当前 bpy 场景
        self.set_active_object(
            object_name=head_object_name
        )  # 指定要驱动 shape key 的头网格

        # ---- 帧缓存：首帧三角化后复用面索引与顶点缓冲，避免逐帧 bmesh 求值 ----
        self._cached_faces: NDArray[np.int64] | None = None
        self._n_verts: int = 0
        self._co_buf: NDArray[np.float32] | None = None

        self.viewer = (
            Open3DMeshViewer(
                window_name=window_name,
                wireframe_head=wireframe_head,
            )
            if enable_viewer
            else None
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
        将指定名称的对象设为活动对象并绑定到当前视图层，同时缓存 shape key 引用。

        参数:
            object_name: 场景中要驱动 blendshape 的网格对象名。
        """
        self.active_obj = bpy.data.objects[object_name]
        bpy.context.view_layer.objects.active = self.active_obj
        self._key_blocks: list[Any] = [
            self.active_obj.data.shape_keys.key_blocks[name]
            for name in BLENDSHAPE_NAMES
        ]

    # ---------- Blendshape 与网格管线 ----------

    @staticmethod
    def _validate_frame(blendshapes: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        校验一维权重向量长度等于 FRAME_WIDTH。

        参数:
            blendshapes: 一帧 SRanipal 维度的 blendshape 权重；形状 ``(FRAME_WIDTH,)`` 的 float64 向量。
        """
        if blendshapes.ndim != 1:
            raise ValueError(
                f"Expected 1-D blendshape vector, got shape {blendshapes.shape}"
            )
        if blendshapes.size != FRAME_WIDTH:
            raise ValueError(
                f"Expected {FRAME_WIDTH} blendshape values, got {blendshapes.size}"
            )
        return blendshapes.astype(np.float64, copy=False)

    # ---------- 快速帧求值管线 ----------

    def _get_triangulated_mesh(self, obj: Any, cage: bool = False) -> Any:
        """
        从依赖图求值对象，三角化后得到新的 Mesh 数据块（仅首帧调用）。

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
    def _read_mesh_vertices_fast(mesh: Any, n_verts: int) -> NDArray[np.float64]:
        """
        用 foreach_get 批量读取 bpy 网格顶点坐标（C 层循环，比 Python 列表推导快约 10 倍）。

        参数:
            mesh: bpy 网格数据块。
            n_verts: 顶点数量。
        """
        buf = np.empty(n_verts * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", buf)
        return buf.reshape(n_verts, 3).astype(np.float64)

    @staticmethod
    def _read_mesh_faces_fast(mesh: Any) -> NDArray[np.int64]:
        """
        用 foreach_get 批量读取已三角化网格的面索引（要求所有面均为三角形）。

        参数:
            mesh: 已三角化的 bpy 网格数据块。
        """
        n_loops = len(mesh.loops)
        buf = np.empty(n_loops, dtype=np.int32)
        mesh.loops.foreach_get("vertex_index", buf)
        return buf.reshape(-1, 3).astype(np.int64)

    def _evaluate_first_frame(
        self, frame: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        """
        首帧完整路径：写 shape key → bmesh 三角化 → 缓存面与顶点数 → 返回顶点与面。

        参数:
            frame: 已校验的一帧 blendshape 权重向量。
        """
        bpy.context.view_layer.objects.active = self.active_obj
        bpy.context.object.update_from_editmode()

        for kb, val in zip(self._key_blocks, frame):
            kb.value = float(val)

        mesh = self._get_triangulated_mesh(self.active_obj)

        n_verts = len(mesh.vertices)
        vertices = self._read_mesh_vertices_fast(mesh, n_verts)
        faces = self._read_mesh_faces_fast(mesh)

        self._cached_faces = faces
        self._n_verts = n_verts
        self._co_buf = np.empty(n_verts * 3, dtype=np.float32)

        bpy.data.meshes.remove(mesh)
        return vertices, faces

    def _evaluate_fast(
        self, frame: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        """
        后续帧快速路径：只写 shape key 值，从 depsgraph 求值读回顶点，复用缓存面。

        参数:
            frame: 已校验的一帧 blendshape 权重向量。
        """
        for kb, val in zip(self._key_blocks, frame):
            kb.value = float(val)

        depsgraph = bpy.context.evaluated_depsgraph_get()
        depsgraph.update()
        obj_eval = self.active_obj.evaluated_get(depsgraph)

        try:
            mesh_eval = obj_eval.to_mesh(depsgraph=depsgraph)
        except TypeError:
            mesh_eval = obj_eval.to_mesh()

        assert self._co_buf is not None
        mesh_eval.vertices.foreach_get("co", self._co_buf)
        obj_eval.to_mesh_clear()

        vertices = self._co_buf.reshape(self._n_verts, 3).astype(np.float64)
        assert self._cached_faces is not None
        return vertices, self._cached_faces

    def extract_frame(self, blendshapes: NDArray[np.float64]) -> FrameData:
        """
        应用 blendshape 并输出顶点与三角面。
        首帧走完整三角化路径；后续帧仅更新顶点位置（复用缓存的面索引）。

        参数:
            blendshapes: 一帧 SRanipal 维度权重，形状 ``(FRAME_WIDTH,)`` 的 float64 向量。
        """
        frame = self._validate_frame(blendshapes)
        if self._cached_faces is None:
            vertices, faces = self._evaluate_first_frame(frame)
        else:
            vertices, faces = self._evaluate_fast(frame)

        return {"vertices": vertices, "faces": faces}

    def render(self, vertices: NDArray[np.float64], faces: NDArray[np.int64]) -> None:
        """
        将网格推送到 Open3D 查看器；可选线框头模式（顶点色由查看器默认肤色填充）。

        参数:
            vertices: 世界空间或模型空间顶点 N×3。
            faces: 三角面索引 F×3。
        """
        if self.viewer is None:
            raise RuntimeError("Viewer is disabled for this runtime instance")

        if self._wireframe_head:
            if self._wf_shell_edges is None:
                shell, tongue = split_head_tongue_meshes(vertices, faces)
                _, shell_faces, self._wf_shell_global_idx = shell
                _, self._wf_tongue_faces, self._wf_tongue_global_idx = tongue
                self._wf_shell_edges = unique_edges_from_faces(shell_faces)
            shell_vertices = vertices[self._wf_shell_global_idx]
            tongue_vertices = vertices[self._wf_tongue_global_idx]
            self.viewer.update(
                vertices,
                faces,
                shell_vertices=shell_vertices,
                tongue_vertices=tongue_vertices,
                shell_edges=self._wf_shell_edges,
                tongue_faces=self._wf_tongue_faces,
            )
            return

        self.viewer.update(vertices, faces)

    def update_visualizer(self, blendshapes: NDArray[np.float64]) -> FrameData:
        """
        提取一帧数据并刷新 Open3D，返回与 extract_frame 相同的结构。

        参数:
            blendshapes: 一帧 blendshape 权重，形状 ``(FRAME_WIDTH,)`` 的 float64 向量。
        """
        frame = self.extract_frame(blendshapes)
        self.render(vertices=frame["vertices"], faces=frame["faces"])
        return frame
