"""读取 SRanipal CSV 并写出 ARKit CSV。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from face_blender_shape.blendshape_mapping import ARKIT_SHAPE_NAMES
from face_blender_shape.core.blendshape_schema import FRAME_WIDTH
from face_blender_shape.core.paths import resolve_input_csv_path

ARKIT_FRAME_WIDTH = len(ARKIT_SHAPE_NAMES)


def load_blendshape_csv(path: str | Path) -> np.ndarray:
    """读取 SRanipal CSV 为二维浮点矩阵。

    path: 输入 CSV 路径。
    """
    resolved = resolve_input_csv_path(path)
    data = np.loadtxt(resolved, delimiter=",")
    data = np.atleast_2d(data)
    if data.shape[1] != FRAME_WIDTH:
        raise ValueError(f"Expected {FRAME_WIDTH} columns, got {data.shape[1]}")
    return data.astype(float, copy=False)


def save_arkit_blendshape_csv(
    path: str | Path,
    frames: np.ndarray,
    *,
    write_header: bool = False,
) -> None:
    """把多帧 ARKit 权重写入 CSV。

    path: 输出 CSV 路径。
    frames: ARKit 权重矩阵，形状应为 (N, 51)。
    write_header: 是否写入表头。
    """
    arr = np.asarray(frames, dtype=float)
    arr = np.atleast_2d(arr)
    if arr.shape[1] != ARKIT_FRAME_WIDTH:
        raise ValueError(f"ARKit 帧须为 {ARKIT_FRAME_WIDTH} 列，当前为 {arr.shape[1]} 列")

    out = Path(path).expanduser()
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ValueError(
            f"无法创建输出目录「{out.parent}」: {exc}\n"
            "请改用可写路径，例如项目内 data/out_arkit.csv，"
            "或 ~/Desktop/out_arkit.csv；不要使用占位路径 /path/to/...。"
        ) from exc

    try:
        handle = out.open("w", encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"无法写入文件「{out}」: {exc}\n请确认路径可写且父目录已存在。") from exc

    with handle:
        if write_header:
            handle.write(",".join(ARKIT_SHAPE_NAMES) + "\n")
        np.savetxt(handle, arr, delimiter=",", fmt="%.6f")
