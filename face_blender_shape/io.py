"""读取符合项目列数约定的 blendshape CSV，解析为浮点数帧矩阵。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from face_blender_shape.blendshape_mapping import ARKIT_SHAPE_NAMES
from face_blender_shape.constants import FRAME_WIDTH
from face_blender_shape.paths import resolve_input_csv_path

# ARKit 一帧列数，与 ARKIT_SHAPE_NAMES 顺序一致。
ARKIT_FRAME_WIDTH = len(ARKIT_SHAPE_NAMES)


def load_blendshape_csv(path: str | Path) -> np.ndarray:
    """
    从 CSV 读取多帧 blendshape 系数矩阵。
    path: CSV 路径；相对路径时按 resolve_input_csv_path 规则解析。
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
    """
    将多帧 ARKit blendshape 权重写入 CSV。
    path: 输出文件路径。
    frames: 形状 (N, 51)，每行一帧；列顺序须与 ARKIT_SHAPE_NAMES 一致。
    write_header: 为 True 时首行为逗号分隔的形态名，便于核对列或与表格软件对齐。
    """
    arr = np.asarray(frames, dtype=float)
    arr = np.atleast_2d(arr)
    if arr.shape[1] != ARKIT_FRAME_WIDTH:
        raise ValueError(
            f"ARKit 帧须为 {ARKIT_FRAME_WIDTH} 列，当前为 {arr.shape[1]} 列"
        )
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
        f = out.open("w", encoding="utf-8")
    except OSError as exc:
        raise ValueError(
            f"无法写入文件「{out}」: {exc}\n请确认路径可写且父目录已存在。"
        ) from exc
    with f:
        if write_header:
            f.write(",".join(ARKIT_SHAPE_NAMES) + "\n")
        np.savetxt(f, arr, delimiter=",", fmt="%.6f")
