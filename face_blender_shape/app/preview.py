"""预览流程编排。"""

from __future__ import annotations

import time
from pathlib import Path

from face_blender_shape.io.blendshape_csv import load_blendshape_csv
from face_blender_shape.runtime.blender_runtime import FaceBlenderRuntime
from face_blender_shape.core.viewer_defaults import (
    DEFAULT_OPEN3D_HEIGHT,
    DEFAULT_OPEN3D_WIDTH,
    DEFAULT_PLAYBACK_FPS,
    DEFAULT_VIEW_SCALE,
)


def preview_sequence(
    path: str | Path,
    fps: float = DEFAULT_PLAYBACK_FPS,
    *,
    view_scale: float = DEFAULT_VIEW_SCALE,
    window_width: int = DEFAULT_OPEN3D_WIDTH,
    window_height: int = DEFAULT_OPEN3D_HEIGHT,
    head_object_name: str | None = None,
    extra_mesh_names: tuple[str, ...] | None = None,
    dual_view: bool = False,
    fbx_path: str | None = None,
) -> None:
    """按 CSV 序列逐帧驱动预览。

    path: 输入 CSV 路径。
    fps: 播放帧率。
    view_scale: 取景比例。
    window_width: 主窗口宽度。
    window_height: 主窗口高度。
    head_object_name: 头部对象名。
    extra_mesh_names: 附加网格对象名元组。
    dual_view: 是否启用侧视窗口。
    fbx_path: 自定义 FBX 路径。
    """
    frames = load_blendshape_csv(path)
    runtime = FaceBlenderRuntime(
        enable_viewer=True,
        enable_side_viewer=dual_view,
        view_scale=view_scale,
        window_width=window_width,
        window_height=window_height,
        head_object_name=head_object_name,
        extra_mesh_names=extra_mesh_names,
        fbx_path=fbx_path,
    )
    frame_delay = 1.0 / fps if fps > 0 else 0.0

    for index, blendshapes in enumerate(frames, start=1):
        print(f"frame {index}/{len(frames)}")
        runtime.update_visualizer(blendshapes)
        if frame_delay > 0:
            time.sleep(frame_delay)
