from __future__ import annotations

from pathlib import Path

import numpy as np

from face_blender_shape.constants import FRAME_WIDTH
from face_blender_shape.paths import resolve_input_csv_path


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
