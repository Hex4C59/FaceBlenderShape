from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from face_blender_shape.constants import BLENDSHAPE_NAMES, FRAME_WIDTH
from face_blender_shape.paths import resolve_input_csv_path, resolve_output_path


def load_blendshape_csv(path: str | Path) -> NDArray[np.float64]:
    resolved = resolve_input_csv_path(path)
    with open(resolved, "r", encoding="utf-8") as f:
        first = f.readline().split(",")[0].strip()
    skiprows = 0
    try:
        float(first)
    except ValueError:
        skiprows = 1
    data = np.loadtxt(resolved, delimiter=",", skiprows=skiprows)
    data = np.atleast_2d(data)

    if data.shape[1] != FRAME_WIDTH:
        raise ValueError(f"Expected {FRAME_WIDTH} columns, got {data.shape[1]}")

    return data.astype(float, copy=False)


def save_keypoints_npz(
    input_path: str | Path,
    *,
    blendshapes: NDArray[np.float64],
    vertices: NDArray[np.float64],
    faces: NDArray[np.int64],
    lip: NDArray[np.float64],
    tongue_tip: NDArray[np.float64],
    cheek_keypoints: NDArray[np.float64],
    keypoints: NDArray[np.float64],
    output_path: str | Path | None = None,
) -> Path:
    """将 blendshape 序列与每帧网格、关键点写入压缩 npz。

    参数:
        input_path: 原始 CSV 路径，用于推导默认输出文件名。
        blendshapes: 形状 (帧数, 通道数) 的 blendshape 系数。
        vertices: 形状 (帧数, V, 3) 的顶点坐标。
        faces: 三角面索引，形状 (F, 3)。
        lip: 唇部相关顶点坐标序列。
        tongue_tip: 舌尖坐标序列。
        cheek_keypoints: 脸颊关键点序列。
        keypoints: 通用关键点序列。
        output_path: 显式输出路径；为 None 时与 input_path 同目录、改后缀 .npz。
    """
    resolved_output = resolve_output_path(
        input_path=input_path, output_path=output_path, suffix=".npz"
    )
    np.savez_compressed(
        resolved_output,
        blendshapes=blendshapes,
        vertices=vertices,
        faces=faces,
        lip=lip,
        tongue_tip=tongue_tip,
        cheek_keypoints=cheek_keypoints,
        keypoints=keypoints,
        blendshape_names=np.array(BLENDSHAPE_NAMES),
    )
    return resolved_output
