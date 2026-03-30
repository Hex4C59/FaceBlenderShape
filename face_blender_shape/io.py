from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from face_blender_shape.constants import FRAME_WIDTH


def load_blendshape_csv(path: str | Path) -> NDArray[np.float64]:
    """加载 blendshape CSV 并返回二维浮点数组。

    参数:
        path: 输入 CSV 路径；相对路径相对进程当前工作目录，支持 ~ 展开。
    """
    # 统一为 Path；expanduser 将 ~/xxx 展开为用户主目录下的路径。
    csv_path = Path(path).expanduser()

    # 只读首行：取第一个单元格，用于判断首行是列名表头还是数值数据。
    with open(csv_path, "r", encoding="utf-8") as file:
        # split(",")[0] 为第一列；strip 去掉首尾空白与换行。
        first = file.readline().split(",")[0].strip()

    # 首格能转成 float 则视为数据行，从第 0 行开始读；否则视为表头，跳过 1 行。
    skiprows = 0
    try:
        float(first)
    except ValueError:
        skiprows = 1

    # 按逗号分隔读入为 float 矩阵；skiprows 与上文表头判断一致。
    data = np.loadtxt(csv_path, delimiter=",", skiprows=skiprows)
    # 仅一行数据时 loadtxt 得到一维向量，升为 (1, N) 以便统一按「帧 × 通道」处理。
    data = np.atleast_2d(data)

    if data.shape[1] != FRAME_WIDTH:
        raise ValueError(f"Expected {FRAME_WIDTH} columns, got {data.shape[1]}")

    # 统一为浮点 dtype；copy=False 在已是 float 时尽量避免多余拷贝。
    return data.astype(float, copy=False)
