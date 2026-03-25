"""在 Blender 中加载角色资源并驱动面部网格。"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import bpy
import numpy as np

from face_blender_shape.blendshape_mapping import ARKIT_SHAPE_NAMES
from face_blender_shape.core.asset_names import (
    METAHUMAN_EYE_LEFT_OBJECT_NAME,
    METAHUMAN_EYE_RIGHT_OBJECT_NAME,
    METAHUMAN_HEAD_OBJECT_NAME,
    METAHUMAN_TEETH_OBJECT_NAME,
)
from face_blender_shape.core.blendshape_schema import BLENDSHAPE_NAMES
from face_blender_shape.core.paths import resolve_fbx_path
from face_blender_shape.core.viewer_defaults import (
    DEFAULT_OPEN3D_HEIGHT,
    DEFAULT_OPEN3D_WIDTH,
    DEFAULT_OPEN3D_WINDOW_NAME,
    DEFAULT_VIEW_SCALE,
)
from face_blender_shape.landmarks import (
    get_cheek_keypoints,
    get_cheek_vertices,
    get_lip_vertices,
    get_tongue_tip,
    get_tongue_vertices,
)
from face_blender_shape.runtime.frame_builder import build_frame_payload
from face_blender_shape.runtime.mesh_eval import get_mesh_data, get_modified_mesh
from face_blender_shape.runtime.shape_key_driver import (
    apply_sranipal_frame,
    resolve_sranipal_mapping,
)
from face_blender_shape.viewer.open3d_viewer import Open3DMeshViewer


class FaceBlenderRuntime:
    """驱动 Blender 中的面部 blendshape，并可选实时预览。"""

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

        enable_viewer: 是否创建主 Open3D 窗口。
        enable_side_viewer: 是否额外创建侧视窗口。
        window_name: 主窗口标题。
        window_width: 主窗口宽度。
        window_height: 主窗口高度。
        view_scale: 取景比例。
        head_object_name: 头部对象名；不传时使用默认 MetaHuman 头部。
        extra_mesh_names: 附加网格对象名序列；不传时使用默认牙齿与双眼。
        fbx_path: 自定义 FBX 路径；不传时使用项目默认模型。
        """
        self._extra_mesh_names = self._normalize_extra_mesh_names(extra_mesh_names)
        self._arkit_names = np.asarray(ARKIT_SHAPE_NAMES)
        self.blendshape_names = np.asarray(BLENDSHAPE_NAMES)
        self._fbx_path = resolve_fbx_path(fbx_path)
        self.load_fbx()
        self.set_active_object(head_object_name or METAHUMAN_HEAD_OBJECT_NAME)
        self._append_parts = self._resolve_append_parts()
        self._use_direct_sranipal_tongue, self._sranipal_to_arkit_matrix = resolve_sranipal_mapping(
            self.active_obj,
            self._append_parts,
        )
        self._print_tongue_direct_mode_status()
        self.viewer = self._build_main_viewer(
            enable_viewer=enable_viewer,
            window_name=window_name,
            view_scale=view_scale,
            window_width=window_width,
            window_height=window_height,
        )
        self.side_viewer = self._build_side_viewer(
            enable_viewer=enable_viewer,
            enable_side_viewer=enable_side_viewer,
            window_name=window_name,
            view_scale=view_scale,
            window_width=window_width,
            window_height=window_height,
        )

    @staticmethod
    def _normalize_extra_mesh_names(extra_mesh_names: Sequence[str] | None) -> tuple[str, ...] | None:
        """清洗附加网格名称序列。

        extra_mesh_names: 外部传入的附加网格名称序列。
        """
        if extra_mesh_names is None:
            return None
        return tuple(name.strip() for name in extra_mesh_names if name.strip())

    def _resolve_append_parts(self):
        """解析需要与头部一起预览的附加网格对象。"""
        if self._extra_mesh_names is None:
            return (
                bpy.data.objects.get(METAHUMAN_TEETH_OBJECT_NAME),
                bpy.data.objects.get(METAHUMAN_EYE_LEFT_OBJECT_NAME),
                bpy.data.objects.get(METAHUMAN_EYE_RIGHT_OBJECT_NAME),
            )

        parts: list[bpy.types.Object] = []
        for name in self._extra_mesh_names:
            obj = bpy.data.objects.get(name)
            if obj is None:
                print(f"[face_blender_shape] warning: extra mesh {name!r} not found in FBX")
                continue
            if obj.type != "MESH":
                print(f"[face_blender_shape] warning: {name!r} is not a mesh, skipped")
                continue
            parts.append(obj)
        return tuple(parts)

    def _print_tongue_direct_mode_status(self) -> None:
        """在启用 Tongue_* 直写时输出提示。"""
        if not self._use_direct_sranipal_tongue:
            return
        print(
            "[face_blender_shape] 检测到 Tongue_* 形态键：舌头将直接写入网格，"
            "SRanipal 舌头通道不再映射到 jawOpen。"
        )

    def _build_main_viewer(
        self,
        *,
        enable_viewer: bool,
        window_name: str,
        view_scale: float,
        window_width: int,
        window_height: int,
    ) -> Open3DMeshViewer | None:
        """按配置创建主视窗口。

        enable_viewer: 是否启用窗口。
        window_name: 窗口标题。
        view_scale: 取景比例。
        window_width: 窗口宽度。
        window_height: 窗口高度。
        """
        if not enable_viewer:
            return None
        return Open3DMeshViewer(
            window_name=window_name,
            view_scale=view_scale,
            window_width=window_width,
            window_height=window_height,
            camera_profile="front",
        )

    def _build_side_viewer(
        self,
        *,
        enable_viewer: bool,
        enable_side_viewer: bool,
        window_name: str,
        view_scale: float,
        window_width: int,
        window_height: int,
    ) -> Open3DMeshViewer | None:
        """按配置创建侧视窗口。

        enable_viewer: 是否启用主显示系统。
        enable_side_viewer: 是否启用侧视。
        window_name: 主窗口标题。
        view_scale: 取景比例。
        window_width: 主窗口宽度。
        window_height: 主窗口高度。
        """
        if not enable_viewer:
            return None
        if not enable_side_viewer:
            return None
        return Open3DMeshViewer(
            window_name=f"{window_name} — 侧视",
            view_scale=view_scale * 1.05,
            window_width=max(window_width * 4 // 5, 480),
            window_height=max(window_height * 4 // 5, 400),
            camera_profile="side",
        )

    def load_fbx(self) -> None:
        """导入当前配置指向的 FBX 文件。"""
        self.fbx_path = self._fbx_path
        bpy.ops.object.select_all(action="DESELECT")
        bpy.ops.import_scene.fbx(filepath=str(self.fbx_path))

    def set_active_object(self, object_name: str = METAHUMAN_HEAD_OBJECT_NAME) -> None:
        """设置当前驱动的头部对象。

        object_name: Blender 场景中的对象名。
        """
        self.active_obj = bpy.data.objects[object_name]
        bpy.context.view_layer.objects.active = self.active_obj

    def set_blendshapes(self, blendshapes: np.ndarray | list[float]) -> bpy.types.Object:
        """应用一帧 blendshape 并返回带变形网格的头部临时对象。

        blendshapes: 一帧 SRanipal 权重。
        """
        bpy.context.view_layer.objects.active = self.active_obj
        self.active_obj.update_from_editmode()
        apply_sranipal_frame(
            self.active_obj,
            self._append_parts,
            blendshapes,
            matrix=self._sranipal_to_arkit_matrix,
            use_direct_sranipal_tongue=self._use_direct_sranipal_tongue,
            arkit_names=self._arkit_names,
        )
        obj = self.active_obj.copy()
        mesh = get_modified_mesh(self.active_obj)
        obj.modifiers.clear()
        obj.data = mesh
        return obj

    def extract_frame(self, blendshapes: np.ndarray | list[float]) -> dict[str, np.ndarray]:
        """应用一帧并返回预览帧字典。

        blendshapes: 一帧 SRanipal 权重。
        """
        obj = self.set_blendshapes(blendshapes)
        mesh = obj.data
        try:
            return build_frame_payload(obj, self._append_parts)
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
        """把几何和颜色提交到所有已启用 viewer。

        vertices: 顶点坐标数组。
        faces: 三角面索引数组。
        vertex_colors: 可选顶点色数组。
        """
        if self.viewer is not None:
            self.viewer.update(vertices, faces, vertex_colors=vertex_colors)
        if self.side_viewer is not None:
            self.side_viewer.update(vertices, faces, vertex_colors=vertex_colors)

    def update_visualizer(self, blendshapes: np.ndarray | list[float]) -> dict[str, np.ndarray]:
        """提取一帧并刷新已启用的 viewer。

        blendshapes: 一帧 SRanipal 权重。
        """
        frame = self.extract_frame(blendshapes)
        self.render(
            frame["vertices"],
            frame["faces"],
            vertex_colors=frame.get("vertex_colors"),
        )
        return frame

    set_key_shapes = set_blendshapes
    get_keypoints = staticmethod(get_mesh_data)
    get_lip = staticmethod(get_lip_vertices)
    get_tongue = staticmethod(get_tongue_vertices)
    get_cheek = staticmethod(get_cheek_vertices)
    get_key_tongue = staticmethod(get_tongue_tip)
    get_key_cheek = staticmethod(get_cheek_keypoints)
